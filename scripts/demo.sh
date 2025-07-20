#!/bin/bash
# trying out demo.py in the baseline configs?

# chairs
#python demo.py --cfg config/chairs.json --ckpt checkpoints/tar-c.pth --dataset chairs

# sintel
#python demo.py --cfg config/tar-c-t-sintel.json --ckpt checkpoints/tar-c-t-sintel.pth --dataset sintel

# spring
#python demo.py --cfg config/tar-c-t-spring-1080p.json --ckpt checkpoints/tar-c-t-spring-1080p.pth --dataset spring
#python demo.py --cfg config/tar-c-t-spring-540p.json --ckpt checkpoints/tar-c-t-spring-540p.pth --dataset spring --scale -1

# tartanair dataset needs to be downloaded

# kitti
#python demo.py --cfg config/tar-c-t-kitti.json --ckpt checkpoints/tar-c-t-kitti.pth --dataset kitti
#python demo.py --cfg config/tar-c-t.json --ckpt checkpoints/tar-c-t.pth --dataset kitti
#python demo.py --cfg config/tar.json --ckpt checkpoints/tar.pth --dataset kitti
#python demo.py --cfg config/tar-c-t-spring-1080p.json --ckpt checkpoints/tar-c-t-spring-1080p.pth --dataset kitti
#python demo.py --cfg config/tar-c-t-sintel.json --ckpt checkpoints/tar-c-t-sintel.pth --dataset kitti

# custom demo
#python demo_custom.py --cfg config/tar-c-t-spring-1080p.json --ckpt checkpoints/tar-c-t-spring-1080p.pth --dataset custom --scale -1
#python demo_custom.py --cfg config/tar-c-t-spring-540p.json --ckpt checkpoints/tar-c-t-spring-540p.pth --dataset custom --scale -2
#python demo_custom.py --cfg config/tar-c-t-kitti.json --ckpt checkpoints/tar-c-t-kitti.pth --dataset custom --scale -2
#python demo_custom.py --cfg config/tar-c-t-sintel.json --ckpt checkpoints/tar-c-t-sintel.pth --dataset custom --scale -2

# custom demo with option to save scales in the output folder
#python demo_custom.py --cfg config/tar-c-t-spring-1080p.json --ckpt checkpoints/tar-c-t-spring-1080p.pth --dataset custom
#python demo_custom.py --cfg config/s177/tar-c-t-spring-1080p_10iter.json --ckpt checkpoints/tar-c-t-spring-1080p.pth --dataset custom
#python demo_custom.py --cfg config/s177/tar-c-t-spring-1080p_20iter.json --ckpt checkpoints/tar-c-t-spring-1080p.pth --dataset custom
#python demo_custom.py --cfg config/s177/tar-c-t-spring-540p_20iter.json --ckpt checkpoints/tar-c-t-spring-540p.pth --dataset custom

#python demo_custom.py --cfg config/tar-c-t-kitti.json --ckpt checkpoints/tar-c-t-kitti.pth --dataset custom
#python demo_custom.py --cfg config/tar-c-t-kitti.json --ckpt checkpoints/tar-c-t-kitti.pth --dataset custom --scale -1
#python demo_custom.py --cfg config/tar-c-t-kitti.json --ckpt checkpoints/tar-c-t-kitti.pth --dataset custom --scale -2

#python demo_custom.py --cfg config/tar-c-t-sintel.json --ckpt checkpoints/tar-c-t-sintel.pth --dataset custom
#python demo_custom.py --cfg config/tar-c-t-sintel.json --ckpt checkpoints/tar-c-t-sintel.pth --dataset custom --scale -1
#python demo_custom.py --cfg config/tar-c-t-sintel.json --ckpt checkpoints/tar-c-t-sintel.pth --dataset custom --scale -2

#python demo_custom.py --cfg config/tar-c-t.json --ckpt checkpoints/tar-c-t.pth --dataset custom
#python demo_custom.py --cfg config/tar-c-t.json --ckpt checkpoints/tar-c-t.pth --dataset custom --scale -1
#python demo_custom.py --cfg config/tar-c-t.json --ckpt checkpoints/tar-c-t.pth --dataset custom --scale -2

# forward warping, standard datasets
#python demo_custom.py --cfg config/tar-c.json --ckpt checkpoints/tar-c.pth --dataset chairs
#python demo_custom.py --cfg config/chairs.json --ckpt checkpoints/chairs.pth --dataset chairs
#python demo_custom.py --cfg config/tar.json --ckpt checkpoints/tar.pth --dataset chairs