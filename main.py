import argparse
import os
import numpy as np
import random
import torch

from data_generator import SpikedCovarianceDataset

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="train", help="Set Experiment Mode")
parser.add_argument("--model", default="ae", help="Set Experiment Model")
parser.add_argument("--ckptfldr", default="v0", help="Folder for Saving Files")
parser.add_argument("--cuda", action="store_true", help="Use CUDA")
parser.add_argument("--gpus", default="0,1", help="GPU Device ID to use separated by commas")
parser.add_argument("--seed", default="0", help="Random seed to allow replication of results")

## Parameters for Data Generation
parser.add_argument("--r", type=int, default=10, help="Representation Dimension of Original Signal")
parser.add_argument("--d", type=int, default=40, help="Representation Dimension of Generated Input")
parser.add_argument("--sigma", type=float, default=1.0, help="Standard Deviation of original signal")

## Parameters for Training
## Add more and change as required
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--steps", type=int, default=100, help="Number of Steps for Training")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

if args.mode=='train':
    # Train Model
    ## Call data generator

    if args.model=='ae':
        ## Call autoencoder trainer
    elif args.model=='cl':
        ## Call contrastive learning trainer

elif args.mode=='test':
    # Test performances
    ## Call test data generator

    if args.model=='ae':
        ## Call appropriate parameter and model extractor
    elif args.model=='cl':
        ## Call appropriate parameter and model extractor

    ## Test matrix U-star

    ## Test final prediction accuracy

    ## Print Results or Visualize
    ## Suggestion : For complicated visualizations, if possible create a separate mode

## Add more modes if required
