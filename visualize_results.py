import csv
import matplotlib.pyplot as plt
import numpy as np

def plot_graph(experiment, metric, seeds):

    model_names = {'ae': 'Auto Encoders', 'cl': 'Contrastive Learning'}
    model_colors = {'ae': 'orange', 'cl': 'blue'}
    result_dict = {'ae': [], 'cl': []}
    for seed in seeds:
        rows = []
        with open(f"v0/{experiment}_{metric}_{seed}.csv") as f:
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
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(f"{experiment}_{metric}.png")
    plt.clf()

if __name__ == "__main__":
    seeds = range(10)

    experiment = "increase_n"
    # experiment = "increase_dimension_d"
    # experiment = "increase_dimension_d_and_r"

    metric = "sinedistance"
    plot_graph(experiment, metric, seeds)
    metric = "downstream_score"
    plot_graph(experiment, metric, seeds)
