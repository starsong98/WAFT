import os
import sys
from model.vitwarp_v8 import ViTWarpV8
# my additions
from model.vitwarp_fwarp import ViTWarpFWarp

def fetch_model(args):
    if args.algorithm == 'vitwarp':
        model = ViTWarpV8(args)
    # my own addition(s)
    elif args.algorithm == 'vitwarp_fwarp':
        model = ViTWarpFWarp(args)
    else:
        raise ValueError("Unknown algorithm: {}".format(args.algorithm))
    return model