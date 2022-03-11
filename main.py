import argparse
import os
import numpy as np
import random
import torch

from contrastive_learning import contrastive_training
from autoencoder import auto_encoder
from data_generator import SpikedCovarianceDataset
from test_utils import sinedistance_eigenvectors, classification_score, regression_score
from downstream import svm_classifier, linear_regressor

downstream_task_name = {'cls': 'Classification', 'reg': 'Regression'}

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
parser.add_argument("--sigma", type=float, default=100., help="Standard Deviation of original signal")

## Parameters for Training
## Add more and change as required
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=32, help="Batch Size for Training")
parser.add_argument("--epochs", type=int, default=1000, help="Number of Steps for Training")
parser.add_argument("--train_size", type=int, default=10000, help="Number of data points of unsupervised learning")
parser.add_argument("--test_size", type=int, default=1000, help="Number of data points of testing")
parser.add_argument("--lam", type=float, default=1e-3, help="Weight of regularization term")

## Parameters for Downstream Task
parser.add_argument("--dwn_mode", default="cls", help="Classification mode for downstream labels")
parser.add_argument("--dwn_model", default="svm", help="Use SVM model for downstream classification")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
np.random.seed(int(args.seed))
random.seed(int(args.seed))
torch.manual_seed(int(args.seed))

## Call data generator
generator = SpikedCovarianceDataset(args.r, args.d, args.sigma, label_mode=args.dwn_mode)
X_train, y_train, r_train = generator.get_next_batch(batch_size=args.train_size)
X_test, y_test, r_test = generator.get_next_batch(batch_size=args.test_size)

if args.mode=='train':
    if args.model=='ae':
        model = auto_encoder(args.d, args.r, X_train, batch_size=args.batch_size, num_epochs=args.epochs, lr=args.lr)
        torch.save(model, os.path.join(args.ckptfldr, 'ae_baseline.pt'))

    elif args.model=='cl':
        ## Call contrastive learning trainer
        model = contrastive_training(args.r, args.d, X_train, loss_fn="NTXENT",
                                     batch_size=args.batch_size, num_epochs=args.epochs, lr=args.lr, lam=args.lam)
        ## Save Model
        torch.save(model, os.path.join(args.ckptfldr, 'cl_baseline.pth'))

elif args.mode=='test':
    if args.model=='ae':
        model = torch.load(os.path.join(args.ckptfldr, 'ae_baseline.pt'))
        weight_matrix = model.encoder.weight.cpu().detach().numpy()

    elif args.model=='cl':
        ## Load Model
        model = torch.load(os.path.join(args.ckptfldr, 'cl_baseline.pth'))
        ## Call appropriate parameter extractor
        weight_matrix = model.linear.weight.cpu().detach().numpy()

    ## Test matrix U-star
    sinedistance_score = sinedistance_eigenvectors(generator.get_ustar(), weight_matrix)
    print("Sine Distance to U* : %f" % sinedistance_score)

    ## Test final prediction accuracy
    representations_train = np.matmul(X_train, weight_matrix.T)
    representations_test = np.matmul(X_test, weight_matrix.T)

    if args.dwn_mode=='cls':
        if args.dwn_model=='svm':
            predicted_labels = svm_classifier(representations_train, y_train, representations_test)
        else:
            raise Exception("Classification Model specified is not Implemented")

        final_score = classification_score(y_test, predicted_labels, mode='acc')

    elif args.dwn_mode=='reg':
        if args.dwn_model=='linear':
            predicted_labels = linear_regressor(representations_train, y_train, representations_test)
        else:
            raise Exception("Regression Model specified is not Implemented")

        final_score = regression_score(y_test, predicted_labels, mode='rmse')

    print("%s Task Score : %f" % (downstream_task_name[args.dwn_mode], final_score))

elif args.mode=='gold':

    if args.dwn_mode=='cls':
        if args.dwn_model=='svm':
            predicted_labels = svm_classifier(r_train, y_train, r_test)
        else:
            raise Exception("Classification Model specified is not Implemented")

        final_score = classification_score(y_test, predicted_labels, mode='acc')

    elif args.dwn_mode=='reg':
        if args.dwn_model=='linear':
            predicted_labels = linear_regressor(r_train, y_train, r_test)
        else:
            raise Exception("Regression Model specified is not Implemented")

        final_score = regression_score(y_test, predicted_labels, mode='rmse')

    print("%s Task Score : %f" % (downstream_task_name[args.dwn_mode], final_score))
## Add more modes if required
