from main import run, obtain_args
import argparse
import csv


def run_all_exp(args):
    # args.experiment = "increase_noise"
    # args.experiment = "increase_dimension_d"
    # args.cuda = True
    # args.experiment = "increase_dimension_r"
    # args.experiment = "increase_dimension_d_and_r"
    args.experiment = "increase_n"

    # args.experiment = "increase_mask"

    sinedistance_scores = {}
    downstream_scores = {}
    for model in ["cl_shallow", "cl_deep"]:
        sinedistance_scores[model] = [model]
        downstream_scores[model] = [model]

        if args.experiment == "increase_mask":
            mask_percentages = ["Mask Percentage", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            for mask_percentage in mask_percentages[1:]:
                args.model = model
                args.mask_percentage = mask_percentage
                sinedistance_score, score = run(args)
                sinedistance_scores[model].append(sinedistance_score)
                downstream_scores[model].append(score)

        if args.experiment == "increase_dimension_d":
            dimensions = ["Dimension", 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 130, 150]
            for dimension in dimensions[1:]:
                args.model = "cl"
                args.d = dimension
                if model == "cl_shallow":
                    args.single_layer = True
                else:
                    args.single_layer = False
                # if model == "cl_flip_fix":
                #     args.flip = True
                #     args.fix = True
                # else:
                #     args.flip = False
                #     args.fix = False
                # if model == "cl_shallow":
                #     args.single_layer = True
                # else:
                #     args.single_layer = False
                sinedistance_score, score = run(args)
                sinedistance_scores[model].append(sinedistance_score)
                downstream_scores[model].append(score)
        if args.experiment == "increase_dimension_d_and_r":
            dimensions = ["Dimension", 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
            for dimension in dimensions[1:]:
                args.model = model
                args.d = dimension
                args.r = dimension//5
                sinedistance_score, score = run(args)
                sinedistance_scores[model].append(sinedistance_score)
                downstream_scores[model].append(score)
        if args.experiment == "increase_n":
            n_values = ["n", 1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000]
            for n in n_values[1:]:
                args.model = "cl"
                args.train_size = n
                if model == "cl_shallow":
                    args.single_layer = True
                else:
                    args.single_layer = False
                sinedistance_score, score = run(args)
                sinedistance_scores[model].append(sinedistance_score)
                downstream_scores[model].append(score)
        if args.experiment == "increase_noise":
            noise_sigmas = ["Noise Sigma", 0.01, 0.1, 0.3, 0.5, 0.8, 0.9, 1, 1.1, 1.3, 1.5, 2, 3]
            for noise_sigma in noise_sigmas[1:]:
                args.model = "cl"
                args.noise_sigma = noise_sigma
                if model == "cl_shallow":
                    args.single_layer = True
                else:
                    args.single_layer = False
                sinedistance_score, score = run(args)
                sinedistance_scores[model].append(sinedistance_score)
                downstream_scores[model].append(score)
    with open(f"{args.ckptfldr}/{args.experiment}_sinedistance_{args.seed}.csv", "w+", newline='') as f:
        writer = csv.writer(f)
        if args.experiment == "increase_mask":
            writer.writerow(mask_percentages)
        if args.experiment == "increase_dimension_d":
            writer.writerow(dimensions)
        if args.experiment == "increase_n":
            writer.writerow(n_values)
        if args.experiment == "increase_noise":
            writer.writerow(noise_sigmas)
        writer.writerow(sinedistance_scores["cl_shallow"])
        writer.writerow(sinedistance_scores["cl_deep"])

    with open(f"{args.ckptfldr}/{args.experiment}_downstream_score_{args.seed}.csv", "w+", newline='') as f:
        writer = csv.writer(f)
        if args.experiment == "increase_mask":
            writer.writerow(mask_percentages)
        if args.experiment == "increase_dimension_d":
            writer.writerow(dimensions)
        if args.experiment == "increase_n":
            writer.writerow(n_values)
        if args.experiment == "increase_noise":
            writer.writerow(noise_sigmas)
        writer.writerow(downstream_scores["cl_shallow"])
        writer.writerow(downstream_scores["cl_deep"])
if __name__ == "__main__":
    args = obtain_args()
    args.ckptfldr = "v5"
    # seeds = range(8, 10)
    seeds = range(10)

    for seed in seeds:
        args.seed = seed
        run_all_exp(args)
