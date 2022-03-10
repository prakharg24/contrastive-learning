import argparse
import os
import numpy as np
import random
import torch

from contrastive_learning import contrastive_training
from data_generator import SpikedCovarianceDataset
from test_utils import sinedistance_eigenvectors

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="train", help="Set Experiment Mode")
parser.add_argument("--model", default="cl", help="Set Experiment Model")
parser.add_argument("--ckptfldr", default="v0", help="Folder for Saving Files")
parser.add_argument("--cuda", action="store_true", help="Use CUDA")
parser.add_argument("--gpus", default="0,1", help="GPU Device ID to use separated by commas")
parser.add_argument("--seed", default=0, help="Random seed to allow replication of results")


## Parameters for Data Generation
parser.add_argument("--r", type=int, default=10, help="Representation Dimension of Original Signal")
parser.add_argument("--d", type=int, default=40, help="Representation Dimension of Generated Input")
parser.add_argument("--sigma", type=float, default=1.0, help="Standard Deviation of original signal")

## Parameters for Training
## Add more and change as required
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=32, help="Batch Size for Training")
parser.add_argument("--epochs", type=int, default=1000, help="Number of Steps for Training")
parser.add_argument("--num_unsup_datapoints", type=int, default=1000, help="Number of data points of unsupervised learning")
parser.add_argument("--lam", type=float, default=1e-3, help="Weight of regularization term")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
np.random.seed(int(args.seed))
random.seed(int(args.seed))
torch.manual_seed(int(args.seed))

## Call data generator
generator = SpikedCovarianceDataset(args.r, args.d, args.sigma, label_mode='classification')

if args.mode=='train':
    X, y = generator.get_next_batch(batch_size=args.num_unsup_datapoints)
    if args.model=='ae':
        ## Call autoencoder trainer
        pass
    elif args.model=='cl':
        ## Call contrastive learning trainer
        model = contrastive_training(args.r, args.d, X, loss_fn="NTXENT",
                                     batch_size=args.batch_size, num_epochs=args.epochs, lr=args.lr, lam=args.lam)
        ## Save Model
        torch.save(model, os.path.join(args.ckptfldr, 'cl_baseline.pth'))

elif args.mode=='test':
    X, y = generator.get_next_batch(batch_size=args.num_unsup_datapoints)
    if args.model=='ae':
        ## Call appropriate parameter and model extractor
        pass
    elif args.model=='cl':
        ## Load Model
        model = torch.load(os.path.join(args.ckptfldr, 'cl_baseline.pth'))
        ## Call appropriate parameter extractor
        wcl = model.linear.weight.cpu().detach().numpy()

    ## Test matrix U-star
    sinedistance_score = sinedistance_eigenvectors(generator.get_ustar(), wcl)
    print("Sine Distance to U* : ", sinedistance_score)

    ## Test final prediction accuracy

    ## Print Results or Visualize
    ## Suggestion : For complicated visualizations, if possible create a separate mode

## Add more modes if required