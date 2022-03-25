from main import run, obtain_args
import argparse
import csv
import math

def cl_theoretical_upper_bound(d,r,n):
    return ((r**1.5/d)*math.log(d)) + math.sqrt(d*r/n)

def run_all_exp(args):
    # args.experiment = "increase_noise"
    # args.experiment = "increase_noise_homogenous"
    # args.experiment = "increase_dimension_d"
    # args.experiment = "increase_dimension_r"
    # args.experiment = "increase_dimension_r_model"
    # args.experiment = "increase_dimension_d_and_r"
    # args.experiment = "increase_n"
    # args.experiment = "bound_cl_with_d"
    args.experiment = "bound_cl_with_r"
    # args.experiment = "bound_cl_with_n"
    bound_experiments = ["bound_cl_with_d", "bound_cl_with_r", "bound_cl_with_n"]

    sinedistance_scores = {}
    downstream_scores = {}
    
    sinedistance_upper_bounds = {}
    for model in ["ae", "cl"]:
        sinedistance_scores[model] = [model]
        downstream_scores[model] = [model]
        sinedistance_upper_bounds[model] = [model]

        if args.experiment == "increase_dimension_d":
            dimensions = ["Dimension", 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
            for dimension in dimensions[1:]:
                args.model = model
                args.d = dimension
                sinedistance_score, score = run(args)
                sinedistance_scores[model].append(sinedistance_score)
                downstream_scores[model].append(score)
        if args.experiment == "increase_dimension_r_model":
            dimensions = ["Model Representation", 2, 5, 10, 20, 40]
            for dimension in dimensions[1:]:
                args.model = model
                args.r_model = dimension
                sinedistance_score, score = run(args)
                sinedistance_scores[model].append(sinedistance_score)
                downstream_scores[model].append(score)
        if args.experiment == "increase_n":
            n_values = ["n", 1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000]
            for n in n_values[1:]:
                args.model = model
                args.train_size = n
                sinedistance_score, score = run(args)
                sinedistance_scores[model].append(sinedistance_score)
                downstream_scores[model].append(score)
        if args.experiment == "increase_noise":
            noise_sigmas = ["Noise Sigma", 0.01, 0.1, 0.2, 0.5, 1., 2., 3., 4., 5.]
            for noise_sigma in noise_sigmas[1:]:
                args.model = model
                args.noise_sigma = noise_sigma
                sinedistance_score, score = run(args)
                sinedistance_scores[model].append(sinedistance_score)
                downstream_scores[model].append(score)
        if args.experiment == "increase_noise_homogenous":
            noise_sigmas = ["Noise Sigma (Homogenous)", 0.01, 0.1, 0.2, 0.5, 1., 2., 3., 4., 5.]
            for noise_sigma in noise_sigmas[1:]:
                args.model = model
                args.noise_sigma = noise_sigma
                args.het_bound = 1.
                sinedistance_score, score = run(args)
                sinedistance_scores[model].append(sinedistance_score)
                downstream_scores[model].append(score)
        if args.experiment == "bound_cl_with_d":
            dimensions = ["Dimension", 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
            for dimension in dimensions[1:]:
                if model == "ae": break
                args.model = model
                args.d = dimension
                sinedistance_score, _ = run(args)
                sinedistance_upper_bound = cl_theoretical_upper_bound(args.d, args.r_model, args.train_size)
                sinedistance_scores[model].append(sinedistance_score)
                sinedistance_upper_bounds[model].append(sinedistance_upper_bound)
        if args.experiment == "bound_cl_with_r":
            dimensions = ["Model Representation", 2, 5, 10, 20, 40]
            for dimension in dimensions[1:]:
                args.model = model
                args.r_model = dimension
                sinedistance_score, _ = run(args)
                sinedistance_upper_bound = cl_theoretical_upper_bound(args.d, args.r_model, args.train_size)
                sinedistance_scores[model].append(sinedistance_score)
                sinedistance_upper_bounds[model].append(sinedistance_upper_bound)
        if args.experiment == "bound_cl_with_n":
            n_values = ["n", 1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000]
            for n in n_values[1:]:
                args.model = model
                args.train_size = n
                sinedistance_score, _ = run(args)
                sinedistance_upper_bound = cl_theoretical_upper_bound(args.d, args.r_model, args.train_size)
                sinedistance_scores[model].append(sinedistance_score)
                sinedistance_upper_bounds[model].append(sinedistance_upper_bound)
    with open(f"{args.ckptfldr}/{args.experiment}_sinedistance_{args.seed}.csv", "w+", newline='') as f:
        writer = csv.writer(f)

        if args.experiment == "increase_dimension_d":
            writer.writerow(dimensions)
        if args.experiment == "increase_dimension_r_model":
            writer.writerow(dimensions)
        if args.experiment == "increase_n":
            writer.writerow(n_values)
        if args.experiment == "increase_noise":
            writer.writerow(noise_sigmas)
        if args.experiment == "increase_noise_homogenous":
            writer.writerow(noise_sigmas)
        if args.experiment == "bound_cl_with_d":
            writer.writerow(dimensions)
        if args.experiment == "bound_cl_with_r":
            writer.writerow(dimensions)
        if args.experiment == "bound_cl_with_n":
            writer.writerow(n_values)
        if args.experiment not in bound_experiments:
            writer.writerow(sinedistance_scores["ae"])
            writer.writerow(sinedistance_scores["cl"])
        else:
            writer.writerow(sinedistance_scores["cl"])
            writer.writerow(sinedistance_upper_bounds["cl"])
    if args.experiment not in bound_experiments:
        with open(f"{args.ckptfldr}/{args.experiment}_downstream_score_{args.seed}.csv", "w+", newline='') as f:
            writer = csv.writer(f)
            if args.experiment == "increase_dimension_d":
                writer.writerow(dimensions)
            if args.experiment == "increase_dimension_r_model":
                writer.writerow(dimensions)
            if args.experiment == "increase_n":
                writer.writerow(n_values)
            if args.experiment == "increase_noise":
                writer.writerow(noise_sigmas)
            if args.experiment == "increase_noise_homogenous":
                writer.writerow(noise_sigmas)
            writer.writerow(downstream_scores["ae"])
            writer.writerow(downstream_scores["cl"])

if __name__ == "__main__":
    args = obtain_args()
    seeds = range(10)

    for seed in seeds:
        args.seed = seed
        run_all_exp(args)
