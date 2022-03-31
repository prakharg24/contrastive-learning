import csv
import matplotlib.pyplot as plt
import numpy as np

def plot_graph(experiment, metric, seeds):

    # model_names = {'ae': 'Auto Encoders', 'cl': 'Contrastive Learning'}
    model_names = {'cl_shallow': 'Contrastive Learning Single Layer', "cl_deep": "Contrastive Learning Non-Linear"}
    # model_names = {'cl_flip_fix': 'Contrastive Learning Flipped and Fixed Mask', "cl_no_flip_no_fixed": "Contrastive Learning Non-Flipped and Non-Fixed Mask"}

    # model_colors = {'ae': 'orange', 'cl': 'blue'}
    model_colors = {'cl_shallow': 'green', 'cl_deep': 'red'}
    # model_colors = {'cl_flip_fix': 'blue', 'cl_no_flip_no_fixed': 'green'}

    # result_dict = {'ae': [], 'cl': []}
    # result_dict = {'cl_flip_fix': [], 'cl_no_flip_no_fixed': []}
    result_dict = {'cl_shallow': [], 'cl_deep': []}

    for seed in seeds:
        rows = []
        with open(f"v5/{experiment}_{metric}_{seed}.csv") as f:
            csvreader = csv.reader(f)
            header = next(csvreader)
            for row in csvreader:
                rows.append(row)

        result_dict[rows[0][0]].append([float(i) for i in rows[0][1:]])
        result_dict[rows[1][0]].append([float(i) for i in rows[1][1:]])
        xvals = [float(i) for i in header[1:]]

    for model in result_dict:
        all_results = np.array(result_dict[model])
        avg_results = np.mean(all_results, axis=0)
        var_upper_bound = np.max(all_results, axis=0)
        var_lower_bound = np.min(all_results, axis=0)

        plt.plot(xvals, avg_results, label=model_names[model], color=model_colors[model], marker='o')
        plt.fill_between(xvals, var_lower_bound, var_upper_bound, color=model_colors[model], alpha=0.3)

    plt.xlabel(header[0])
    plt.ylabel("Downstream Risk")
    plt.legend()
    plt.savefig(f"{experiment}_{metric}.png")
    plt.show()
    plt.clf()

if __name__ == "__main__":
    seeds = range(10)
    # experiment = "increase_noise"

    # experiment = "increase_dimension_d"
    experiment = "increase_n"
    # experiment = "increase_dimension_d"
    # experiment = "increase_dimension_d_and_r"

    # metric = "sinedistance"
    # plot_graph(experiment, metric, seeds)
    metric = "downstream_score"
    plot_graph(experiment, metric, seeds)
