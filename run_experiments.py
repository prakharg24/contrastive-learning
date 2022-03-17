from main import run
import argparse
import csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="both", help="Set Experiment Mode")
    parser.add_argument("--ckptfldr", default="v0", help="Folder for Saving Files")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--gpus", default="0,1", help="GPU Device ID to use separated by commas")
    parser.add_argument("--seed", default=0, help="Random seed to allow replication of results")

    ## Parameters for Data Generation
    parser.add_argument("--r", type=int, default=10, help="Representation Dimension of Original Signal")
    parser.add_argument("--d", type=int, default=40, help="Representation Dimension of Generated Input")
    parser.add_argument("--sigma", type=float, default=1., help="Standard Deviation of original signal")
    parser.add_argument("--noise_sigma", type=float, default=0.5, help="Standard Deviation of original signal")

    ## Parameters for Training
    ## Add more and change as required
    parser.add_argument("--r_model", type=int, default=10, help="Representation Dimension of Model Output")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size for Training")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of Steps for Training")
    parser.add_argument("--train_size", type=int, default=1000, help="Number of data points of unsupervised learning")
    parser.add_argument("--test_size", type=int, default=1000, help="Number of data points of testing")
    parser.add_argument("--lam", type=float, default=1e-3, help="Weight of regularization term")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")

    ## Parameters for Downstream Task
    parser.add_argument("--dwn_mode", default="reg", help="Classification mode for downstream labels")
    parser.add_argument("--dwn_model", default="linear", help="Use SVM model for downstream classification")
    ## Parameters for Downstream Task
    parser.add_argument("--experiment", default="increase_noise", help="what experiment to run")

    args = parser.parse_args()
    sinedistance_scores = {}
    downstream_scores = {}
    for model in ["cl"]:
        sinedistance_scores[model] = [model]
        downstream_scores[model] = [model]
        if args.experiment == "increase_noise":
            noise_sigmas = ["Noise Sigma", 0.01, 0.1, 0.2, 0.5, 1, 2]
            for noise_sigma in noise_sigmas[1:]:
                args.model = model
                args.noise_sigma = noise_sigma
                sinedistance_score, score = run(args)
                sinedistance_scores[model].append(sinedistance_score)
                downstream_scores[model].append(score)
    with open(f"{args.experiment}_sinedistance.csv", "w+", newline='') as f:
        writer = csv.writer(f)
        if args.experiment == "increase_noise":
            writer.writerow(noise_sigmas)
            writer.writerow(sinedistance_scores["ae"])
            writer.writerow(sinedistance_scores["cl"])
    with open(f"{args.experiment}_downstream_score.csv", "w+", newline='') as f:
        writer = csv.writer(f)
        if args.experiment == "increase_noise":
            writer.writerow(noise_sigmas)
            writer.writerow(downstream_scores["ae"])
            writer.writerow(downstream_scores["cl"])
