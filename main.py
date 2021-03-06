import argparse
import os
import numpy as np
import random
import torch

from contrastive_learning import contrastive_training
from autoencoder import auto_encoder
from data_generator import SpikedCovarianceDataset
from test_utils import sinedistance_eigenvectors, downstream_score


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    ## Call data generator
    generator = SpikedCovarianceDataset(args.r, args.d, args.sigma, args.noise_sigma, args.het_bound, label_mode=args.dwn_mode)
    X_train, y_train, r_train = generator.get_next_batch(batch_size=args.train_size)
    X_test, y_test, r_test = generator.get_next_batch(batch_size=args.test_size)

    if args.mode=='train' or args.mode=='both':
        if args.model=='ae':
            model = auto_encoder(args.d, args.r_model, X_train,
                                 batch_size=args.batch_size, num_epochs=args.epochs, lr=args.lr,
                                 single_layer=True, requires_relu=False,
                                 lam=args.lam, patience=args.patience,
                                 mask_percentage=args.mask_percentage, cuda=args.cuda)
            torch.save(model, os.path.join(args.ckptfldr, 'ae_baseline.pt'))

        elif args.model=='cl':
            ## Call contrastive learning trainer
            model = contrastive_training(args.r_model, args.d, X_train, generator.get_ustar(), loss_fn="NTXENT",
                                         batch_size=args.batch_size, num_epochs=args.epochs,
                                         lr=args.lr, lam=args.lam, patience=args.patience, cuda=args.cuda)
            ## Save Model
            torch.save(model, os.path.join(args.ckptfldr, 'cl_baseline.pth'))

    if args.mode=='test' or args.mode=='both':
        if args.model=='ae':
            model = torch.load(os.path.join(args.ckptfldr, 'ae_baseline.pt'))
            if model.single_layer:
                weight_matrix = model.encoder.weight.cpu().detach().numpy()
            else:
                weight_matrix = np.identity(args.d)
                for layer in model.encoder:
                    weight_matrix = np.matmul(weight_matrix, layer.weight.cpu().detach().numpy())

        elif args.model=='cl':
            ## Load Model
            model = torch.load(os.path.join(args.ckptfldr, 'cl_baseline.pth'))
            ## Call appropriate parameter extractor
            weight_matrix = model.linear.weight.cpu().detach().numpy()

        ## Test matrix U-star
        gold_ustar = generator.get_ustar()
        sinedistance_score = sinedistance_eigenvectors(gold_ustar, weight_matrix)
        print("Sine Distance to U* : %f" % sinedistance_score)

        weight_matrix = weight_matrix.T
        representations_train = np.matmul(X_train, weight_matrix)
        representations_test = np.matmul(X_test, weight_matrix)
        score = downstream_score(args.dwn_mode, args.dwn_model,
                         representations_train, y_train,
                         representations_test, y_test)

        baseline_rep_train = np.matmul(X_train, gold_ustar)
        baseline_rep_test = np.matmul(X_test, gold_ustar)

        baseline_score = downstream_score(args.dwn_mode, args.dwn_model,
                         baseline_rep_train, y_train,
                         baseline_rep_test, y_test)

        return sinedistance_score, score - baseline_score

    if args.mode=='gold':
        downstream_score(args.dwn_mode, args.dwn_model,
                         r_train, y_train,
                         r_test, y_test)


def obtain_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="both", help="Set Experiment Mode")
    parser.add_argument("--model", default="ae", help="Set Experiment Model")
    parser.add_argument("--ckptfldr", default="v0", help="Folder for Saving Files")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--gpus", default="0,1", help="GPU Device ID to use separated by commas")
    parser.add_argument("--seed", default=0, help="Random seed to allow replication of results")

    ## Parameters for Data Generation
    parser.add_argument("--r", type=int, default=10, help="Representation Dimension of Original Signal")
    parser.add_argument("--d", type=int, default=40, help="Representation Dimension of Generated Input")
    parser.add_argument("--sigma", type=float, default=1., help="Standard Deviation of original signal")
    parser.add_argument("--noise_sigma", type=float, default=1., help="Standard Deviation of noise signal")
    parser.add_argument("--het_bound", type=float, default=2., help="Heteroskedastic noise ratio bound")

    ## Parameters for Training
    parser.add_argument("--r_model", type=int, default=10, help="Representation Dimension of Model Output")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size for Training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of Steps for Training")
    parser.add_argument("--train_size", type=int, default=20000, help="Number of data points of unsupervised learning")
    parser.add_argument("--test_size", type=int, default=1000, help="Number of data points of testing")
    parser.add_argument("--lam", type=float, default=1e-3, help="Weight of regularization term")
    parser.add_argument("--patience", type=int, default=50, help="Patience for early stopping")
    parser.add_argument("--mask_percentage", type=float, default=0.5, help="Percentage of masking for data augmentation")

    ## Parameters for Downstream Task
    parser.add_argument("--dwn_mode", default="reg", help="Classification mode for downstream labels")
    parser.add_argument("--dwn_model", default="linear", help="Use SVM model for downstream classification")

    args = parser.parse_args()
    return args
if __name__ == "__main__":

    args = obtain_args()
    run(args)
    # sinedistance_score, score = run(args)
    # print(sinedistance_score, score)
