import numpy as np
import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.depthanythingv2 import DepthAnythingFeature
from model.backbone.vit import VisionTransformer, MODEL_CONFIGS

from utils.utils import coords_grid, Padder, bilinear_sampler

import timm


########################################
# forward warping functions
########################################

def get_gaussian_weights(x, y, x1, x2, y1, y2, z=1.0):
    # z 0.0 ~ 1.0
    w11 = z * torch.exp(-((x - x1) ** 2 + (y - y1) ** 2))
    w12 = z * torch.exp(-((x - x1) ** 2 + (y - y2) ** 2))
    w21 = z * torch.exp(-((x - x2) ** 2 + (y - y1) ** 2))
    w22 = z * torch.exp(-((x - x2) ** 2 + (y - y2) ** 2))

    return w11, w12, w21, w22


def sample_one(img, shiftx, shifty, weight):
    """
    Input:
        -img (N, C, H, W)
        -shiftx, shifty (N, c, H, W)
    """

    N, C, H, W = img.size()

    # flatten all (all restored as Tensors)
    flat_shiftx = shiftx.view(-1)
    flat_shifty = shifty.view(-1)
    flat_basex = torch.arange(0, H, requires_grad=False).view(-1, 1)[None, None].to(img.device).long().repeat(N, C,1,W).view(-1)
    flat_basey = torch.arange(0, W, requires_grad=False).view(1, -1)[None, None].to(img.device).long().repeat(N, C,H,1).view(-1)
    flat_weight = weight.view(-1)
    flat_img = img.contiguous().view(-1)

    # The corresponding positions in I1
    idxn = torch.arange(0, N, requires_grad=False).view(N, 1, 1, 1).to(img.device).long().repeat(1, C, H, W).view(-1)
    idxc = torch.arange(0, C, requires_grad=False).view(1, C, 1, 1).to(img.device).long().repeat(N, 1, H, W).view(-1)
    idxx = flat_shiftx.long() + flat_basex
    idxy = flat_shifty.long() + flat_basey

    # recording the inside part the shifted
    mask = idxx.ge(0) & idxx.lt(H) & idxy.ge(0) & idxy.lt(W)

    # Mask off points out of boundaries
    ids = (idxn * C * H * W + idxc * H * W + idxx * W + idxy)
    ids_mask = torch.masked_select(ids, mask).clone().to(img.device)

    # Note here! accmulate fla must be true for proper bp
    img_warp = torch.zeros([N * C * H * W, ]).to(img.device)
    img_warp.put_(ids_mask, torch.masked_select(flat_img * flat_weight, mask), accumulate=True)

    one_warp = torch.zeros([N * C * H * W, ]).to(img.device)
    one_warp.put_(ids_mask, torch.masked_select(flat_weight, mask), accumulate=True)

    return img_warp.view(N, C, H, W), one_warp.view(N, C, H, W)


def fwarp(img, flo):

    """
        -img: image (N, C, H, W)
        -flo: optical flow (N, 2, H, W)
        elements of flo is in [0, H] and [0, W] for dx, dy
        https://github.com/lyh-18/EQVI/blob/EQVI-master/models/forward_warp_gaussian.py
        https://github.com/JihyongOh/XVFI/blob/main/XVFInet.py#L237
    """

    # (x1, y1)		(x1, y2)
    # +---------------+
    # |				  |
    # |	o(x, y) 	  |
    # |				  |
    # |				  |
    # |				  |
    # |				  |
    # +---------------+
    # (x2, y1)		(x2, y2)

    N, C, _, _ = img.size()

    # translate start-point optical flow to end-point optical flow
    y = flo[:, 0:1:, :]
    x = flo[:, 1:2, :, :]

    x = x.repeat(1, C, 1, 1)
    y = y.repeat(1, C, 1, 1)

    # Four point of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
    x1 = torch.floor(x)
    x2 = x1 + 1
    y1 = torch.floor(y)
    y2 = y1 + 1

    # firstly, get gaussian weights
    w11, w12, w21, w22 = get_gaussian_weights(x, y, x1, x2, y1, y2)

    # secondly, sample each weighted corner
    img11, o11 = sample_one(img, x1, y1, w11)
    img12, o12 = sample_one(img, x1, y2, w12)
    img21, o21 = sample_one(img, x2, y1, w21)
    img22, o22 = sample_one(img, x2, y2, w22)

    imgw = img11 + img12 + img21 + img22
    o = o11 + o12 + o21 + o22

    return imgw, o


class resconv(nn.Module):
    def __init__(self, inp, oup, k=3, s=1):
        super(resconv, self).__init__()
        self.conv = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(inp, oup, kernel_size=k, stride=s, padding=k//2, bias=True),
            nn.GELU(),
            nn.Conv2d(oup, oup, kernel_size=3, stride=1, padding=1, bias=True),
        )
        if inp != oup or s != 1:
            self.skip_conv = nn.Conv2d(inp, oup, kernel_size=1, stride=s, padding=0, bias=True)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.skip_conv(x)

class ResNet18Deconv(nn.Module):
    def __init__(self, inp, oup):
        super(ResNet18Deconv, self).__init__()
        self.feature_dims = [64, 128, 256, 512]
        self.ds1 = resconv(inp, 64, k=7, s=2)
        self.conv1 = timm.create_model("resnet18.a3_in1k", pretrained=True, features_only=True).layer1
        self.conv2 = timm.create_model("resnet18.a3_in1k", pretrained=True, features_only=True).layer2
        self.conv3 = timm.create_model("resnet18.a3_in1k", pretrained=True, features_only=True).layer3
        self.conv4 = timm.create_model("resnet18.a3_in1k", pretrained=True, features_only=True).layer4
        self.up_4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_3 = resconv(256, 256, k=3, s=1)
        self.up_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_2 = resconv(128, 128, k=3, s=1)
        self.up_2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_1 = resconv(64, oup, k=3, s=1)

    def forward(self, x):
        out_1 = self.ds1(x)
        out_1 = self.conv1(out_1)
        out_2 = self.conv2(out_1)
        out_3 = self.conv3(out_2)
        out_4 = self.conv4(out_3)
        out_3 = self.proj_3(out_3 + self.up_4(out_4))
        out_2 = self.proj_2(out_2 + self.up_3(out_3))
        out_1 = self.proj_1(out_1 + self.up_2(out_2))
        return [out_1, out_2, out_3, out_4]

class ViTWarpFWarp(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.da_feature = self.freeze_(DepthAnythingFeature(encoder=args.dav2_backbone))
        self.pretrain_dim = self.da_feature.model_configs[args.dav2_backbone]['features']
        self.network_dim = MODEL_CONFIGS[args.network_backbone]['features']
        self.refine_net = VisionTransformer(args.network_backbone, self.network_dim, patch_size=8)
        self.fnet = ResNet18Deconv(self.pretrain_dim//2 + 3, 64)
        self.fmap_conv = nn.Conv2d(self.pretrain_dim//2 + 64, self.network_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.hidden_conv = nn.Conv2d(self.network_dim*2, self.network_dim, kernel_size=1, stride=1, padding=0, bias=True)
        #self.warp_linear = nn.Conv2d(3*self.network_dim+2, self.network_dim, 1, 1, 0, bias=True)
        self.warp_linear = nn.Conv2d(5*self.network_dim+2, self.network_dim, 1, 1, 0, bias=True)
        self.refine_transform = nn.Conv2d(self.network_dim//2*3, self.network_dim, 1, 1, 0, bias=True)
        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(self.network_dim, 2*self.network_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*self.network_dim, 4*9, 1, padding=0, bias=True)
        )
        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(self.network_dim, 2*self.network_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*self.network_dim, 6, 1, padding=0, bias=True)
        )

    def freeze_(self, model):
        model = model.eval()
        for p in model.parameters():
            p.requires_grad = False
        for p in model.buffers():
            p.requires_grad = False
        return model

    def upsample_data(self, flow, info, mask):
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 2, 2, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(2 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)
        
        return up_flow.reshape(N, 2, 2*H, 2*W), up_info.reshape(N, C, 2*H, 2*W)

    def normalize_image(self, img):
        '''
        @img: (B,C,H,W) in range 0-255, RGB order
        '''
        tf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
        return tf(img/255.0).contiguous()

    def forward(self, image1, image2, iters=None, flow_gt=None):
        """ Estimate optical flow between pair of frames """
        if iters is None:
            iters = self.args.iters
        image1 = self.normalize_image(image1)
        image2 = self.normalize_image(image2)
        padder = Padder(image1.shape, factor=112)
        image1 = padder.pad(image1)
        image2 = padder.pad(image2)
        flow_predictions = []
        info_predictions = [] 
        N, _, H, W = image1.shape
        # initial feature
        da_feature1 = self.da_feature(image1)
        da_feature2 = self.da_feature(image2)
        fmap1_feats = self.fnet(torch.cat([da_feature1['out'], image1], dim=1))
        fmap2_feats = self.fnet(torch.cat([da_feature2['out'], image2], dim=1))
        da_feature1_2x = F.interpolate(da_feature1['out'], scale_factor=0.5, mode='bilinear', align_corners=True)
        da_feature2_2x = F.interpolate(da_feature2['out'], scale_factor=0.5, mode='bilinear', align_corners=True)
        fmap1_2x = self.fmap_conv(torch.cat([fmap1_feats[0], da_feature1_2x], dim=1))
        fmap2_2x = self.fmap_conv(torch.cat([fmap2_feats[0], da_feature2_2x], dim=1))
        net = self.hidden_conv(torch.cat([fmap1_2x, fmap2_2x], dim=1))
        flow_2x = torch.zeros(N, 2, H//2, W//2).to(image1.device)
        for itr in range(iters):
            flow_2x = flow_2x.detach()
            coords2 = (coords_grid(N, H//2, W//2, device=image1.device) + flow_2x).detach()
            warp_2x = bilinear_sampler(fmap2_2x, coords2.permute(0, 2, 3, 1))
            # TODO implement forward warping
            fwarp_2x = self.forward_warp(fmap1_2x, flow_2x)
            #refine_inp = self.warp_linear(torch.cat([fmap1_2x, warp_2x, net, flow_2x], dim=1))
            refine_inp = self.warp_linear(torch.cat([fmap1_2x, warp_2x, fwarp_2x, fmap2_2x, net, flow_2x], dim=1))
            refine_outs = self.refine_net(refine_inp)
            net = self.refine_transform(torch.cat([refine_outs['out'], net], dim=1))
            flow_update = self.flow_head(net)
            weight_update = .25 * self.upsample_weight(net)
            flow_2x = flow_2x + flow_update[:, :2]
            info_2x = flow_update[:, 2:]
            # upsample predictions
            flow_up, info_up = self.upsample_data(flow_2x, info_2x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

        for i in range(len(info_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            info_predictions[i] = padder.unpad(info_predictions[i])
        
        if flow_gt is not None:
            nf_predictions = []
            for i in range(len(info_predictions)):                 
                raw_b = info_predictions[i][:, 2:]
                log_b = torch.zeros_like(raw_b)
                weight = info_predictions[i][:, :2]
                log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=self.args.var_max)
                log_b[:, 1] = torch.clamp(raw_b[:, 1], min=self.args.var_min, max=0)
                term2 = ((flow_gt - flow_predictions[i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
                term1 = weight - math.log(2) - log_b
                nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
                nf_predictions.append(nf_loss)
            output = {'flow': flow_predictions, 'info': info_predictions, 'nf': nf_predictions}
        else:
            output = {'flow': flow_predictions, 'info': info_predictions}    
        
        return output

    def forward_warp(self, img, flo):
        Ft1, norm1 = fwarp(img, flo)
        Ft1[norm1 > 0] = Ft1[norm1 > 0] / norm1[norm1 > 0]
        return Ft1