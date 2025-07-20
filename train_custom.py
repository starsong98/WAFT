import sys
import os
import argparse
import numpy as np
import torch
import time
import torch.optim as optim
from tqdm import tqdm

from config.parser import parse_args
from model import fetch_model
from dataloader.loader import fetch_dataloader
from utils.utils import load_ckpt
from utils.ddp_utils import *
from criterion.loss import sequence_loss

import wandb

from evaluate_custom import validate_inloop

os.system("export KMP_INIT_AT_FORK=FALSE")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def train(args, rank=0, world_size=1, use_ddp=False):
    """ Full training loop """
    device_id = rank
    model = fetch_model(args).to(device_id)
    if rank == 0:
        avg_loss = AverageMeter()
        avg_epe = AverageMeter()
        wandb.init(
            #project=args.name
            project='WAFT_debug',
            name=args.name
        )
    if args.restore_ckpt is not None:
        load_ckpt(model, args.restore_ckpt)
        print(f"restore ckpt from {args.restore_ckpt}")

    # setup models
    if use_ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], static_graph=True)
    model.train()
    train_loader = fetch_dataloader(args, rank=rank, world_size=world_size, use_ddp=use_ddp)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    #VAL_FREQ = 10000
    VAL_FREQ = args.val_freq if (hasattr(args, "val_freq") and args.val_freq is not None) else 10000
    epoch = 0
    cnt_overheat = 0
    should_keep_training = True
    
    # Create overall progress bar for total steps (only on rank 0)
    if rank == 0:
        overall_pbar = tqdm(total=args.num_steps, desc="Training Progress", unit="step")
    
    while should_keep_training:
        # shuffle sampler
        #train_loader.sampler.set_epoch(epoch)
        # Only set epoch for distributed sampler
        if use_ddp and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        epoch += 1
        
        # Create epoch progress bar (only on rank 0)
        if rank == 0:
            epoch_pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", leave=False)
        else:
            epoch_pbar = train_loader
            
        for i_batch, data_blob in enumerate(epoch_pbar):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda(non_blocking=True) for x in data_blob]
            output = model(image1, image2, flow_gt=flow)
            loss = sequence_loss(output, flow, valid, args.gamma)
            
            # logger
            if rank == 0:
                if valid.sum() > 0:
                    avg_loss.update(loss.item())
                    epe = (((flow - output['flow'][-1])**2).sum(dim=1)).sqrt()
                    epe = (epe * valid).sum() / valid.sum()
                    avg_epe.update(epe.item())
                    
                # Update progress bar description with current metrics
                epoch_pbar.set_postfix({
                    'Loss': f'{avg_loss.avg:.4f}',
                    'EPE': f'{avg_epe.avg:.4f}',
                    'Step': f'{total_steps}/{args.num_steps}'
                })
                
                if total_steps % 100 == 0:
                #if total_steps % 10 == 0:
                    #wandb.log({"loss": avg_loss.avg, "epe": avg_epe.avg})
                    wandb.log({"loss": avg_loss.avg, "epe": avg_epe.avg}, step=total_steps)
                    wandb.log({"lr": scheduler.get_last_lr()[0]}, step=total_steps)
                    avg_loss.reset()
                    avg_epe.reset()
                    cnt_overheat = 0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) 
            optimizer.step()
            scheduler.step()
            
            if total_steps % VAL_FREQ == VAL_FREQ - 1 and rank == 0:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                if use_ddp:
                    torch.save(model.module.state_dict(), PATH)
                else:
                    torch.save(model.state_dict(), PATH)
                overall_pbar.set_description(f"Training Progress - Saved checkpoint at step {total_steps+1}")

                # validation loop
                # TODO later
                if use_ddp:
                    model_to_validate = model.module
                else:
                    model_to_validate = model
                model.eval()
                val_dict = validate_inloop(args, model_to_validate)
                wandb.log(val_dict, step=total_steps)
                model.train()
            
            if total_steps > args.num_steps:
                should_keep_training = False
                break
            
            total_steps += 1
            
            # Update overall progress bar
            if rank == 0:
                overall_pbar.update(1)

        # Close epoch progress bar
        if rank == 0:
            epoch_pbar.close()

    # Close overall progress bar
    if rank == 0:
        overall_pbar.close()

    PATH = 'checkpoints/%s.pth' % args.name
    if rank == 0:
        if use_ddp:
            torch.save(model.module.state_dict(), PATH)
        else:
            torch.save(model.state_dict(), PATH)
        wandb.finish()
        
    return PATH

def main(rank, world_size, args, use_ddp):
    if use_ddp:
        print(f"Using DDP [{rank=} {world_size=}]")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        setup_ddp(rank, world_size)

    train(args, rank=rank, world_size=world_size, use_ddp=use_ddp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--seed', help='seed', default=42, type=int)
    parser.add_argument('--restore_ckpt', help='restore checkpoint', default=None, type=str)
    args = parse_args(parser)
    smp, world_size = init_ddp()
    if world_size > 1:
        spwn_ctx = mp.spawn(main, nprocs=world_size, args=(world_size, args, True), join=False)
        spwn_ctx.join()
    else:
        main(0, 1, args, False)
    print("Done!")