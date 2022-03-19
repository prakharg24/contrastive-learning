from main import run, obtain_args
import argparse
import csv


if __name__ == "__main__":
    args = obtain_args()
    # args.experiment = "increase_noise"
    #args.experiment = "increase_dimension_d"
    #args.experiment = "increase_dimension_r"
    #args.experiment = "increase_dimension_d_and_r"
    args.experiment = "increase_n"

    sinedistance_scores = {}
    downstream_scores = {}
    for model in ["ae", "cl"]:
        sinedistance_scores[model] = [model]
        downstream_scores[model] = [model]

        if args.experiment == "increase_dimension_d":
            dimensions = ["Dimension", 10, 30, 50, 70, 90, 110]
            for dimension in dimensions[1:]:
                args.model = model
                args.d = dimension
                sinedistance_score, score = run(args)
                sinedistance_scores[model].append(sinedistance_score)
                downstream_scores[model].append(score)
        if args.experiment == "increase_n":
            n_values = ["n", 1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 50000]
            for n in n_values[1:]:
                args.model = model
                args.train_size = n
                sinedistance_score, score = run(args)
                sinedistance_scores[model].append(sinedistance_score)
                downstream_scores[model].append(score)
        if args.experiment == "increase_noise":
            noise_sigmas = ["Noise Sigma", 0.01, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5]
            for noise_sigma in noise_sigmas[1:]:
                args.model = model
                args.noise_sigma = noise_sigma
                sinedistance_score, score = run(args)
                sinedistance_scores[model].append(sinedistance_score)
                downstream_scores[model].append(score)
    with open(f"{args.experiment}_sinedistance.csv", "w+", newline='') as f:
        writer = csv.writer(f)

        if args.experiment == "increase_dimension_d":
            writer.writerow(dimensions)
        if args.experiment == "increase_n":
            writer.writerow(n_values)
        if args.experiment == "increase_noise":
            writer.writerow(noise_sigmas)
        writer.writerow(sinedistance_scores["ae"])
        writer.writerow(sinedistance_scores["cl"])
    with open(f"{args.experiment}_downstream_score.csv", "w+", newline='') as f:
        writer = csv.writer(f)
        if args.experiment == "increase_dimension_d":
            writer.writerow(dimensions)
        if args.experiment == "increase_n":
            writer.writerow(n_values)
        if args.experiment == "increase_noise":
            writer.writerow(noise_sigmas)
        writer.writerow(downstream_scores["ae"])
        writer.writerow(downstream_scores["cl"])
