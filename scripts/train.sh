#!/bin/bash
# trying out training throughput

# Set the environment variable
export WANDB_API_KEY=3e47c12c726946688f60a01031af30854ae44216
export CUDA_VISIBLE_DEVICES=0,1
#export CUDA_VISIBLE_DEVICES=0

# original chairs training
#python train.py --cfg config/chairs.json

# my tryout
#python train.py --cfg config/s177/chairs.json
#python train_custom.py --cfg config/s177/chairs.json    # bs=4 --> 10.2 GiB/GPU, 3.1~3.2 batch/s, ETA approx. 36h
#python train_custom.py --cfg config/s177/chairs.json    # bs=8 --> 18.6 GiB/GPU, 1.9~2.0 batch/s, ETA approx. 29h
#python train_custom.py --cfg config/s177/chairs_gpu0.json    # bs=4, 1GPU -->

# this thing already uses WandB. good to know.

# debugging
#python train_custom.py --cfg config/s177/chairs_gpu0_debug.json

# no gradient accumulation, flyingchairs stage
#python train_custom.py --cfg config/s177/chairs_gpu01_nogradacc.json

# debugging - gradient accumulation
#python train_custom_2.py --cfg config/s177/chairs_gpu0_debug.json

# gradient accumulation, flyingchairs stage
python train_custom_2.py --cfg config/s177/chairs_gpu01_gradacc.json