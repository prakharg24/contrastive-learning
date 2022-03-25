import csv
import matplotlib.pyplot as plt
import numpy as np

def plot_graph(experiment, metric, seeds):

    bound_experiments = ["bound_cl_with_d", "bound_cl_with_r", "bound_cl_with_n"]
    bound_metrics = ["sinedistance"]
    if experiment in bound_experiments:
        if metric not in bound_metrics: return
        test_bound = True    

    model_names = {'ae': 'Auto Encoders', 'cl': 'Contrastive Learning'}
    model_colors = {'ae': 'orange', 'cl': 'blue'}
    result_dict = {'ae': [], 'cl': []}
    bound_dict = {'empirical': [], 'theoretical': []}
    for seed in seeds:
        rows = []
        with open(f"v0/{experiment}_{metric}_{seed}.csv") as f:
            csvreader = csv.reader(f)
            header = next(csvreader)
            for row in csvreader:
                rows.append(row)

        if experiment not in bound_experiments:
            result_dict[rows[0][0]].append([float(i) for i in rows[0][1:]])
            result_dict[rows[1][0]].append([float(i) for i in rows[1][1:]])
        if experiment in bound_experiments: 
            bound_dict["empirical"].append([float(i) for i in rows[0][1:]])
            bound_dict["theoretical"].append([float(i) for i in rows[1][1:]])
        xvals = [float(i) for i in header[1:]]

    if experiment not in bound_experiments:
        for model in result_dict:
            all_results = np.array(result_dict[model])
            avg_results = np.mean(all_results, axis=0)
            var_upper_bound = np.max(all_results, axis=0)
            var_lower_bound = np.min(all_results, axis=0)

            plt.plot(xvals, avg_results, label=model_names[model], color=model_colors[model], marker='o')
            plt.fill_between(xvals, var_lower_bound, var_upper_bound, color=model_colors[model], alpha=0.3)
    else:
        all_results = np.array(bound_dict["empirical"])
        avg_results = np.mean(all_results, axis=0)
        var_upper_bound = np.max(all_results, axis=0)
        var_lower_bound = np.min(all_results, axis=0)
        theoretical_upper_bound = np.mean(np.array(bound_dict["theoretical"]), axis=0) #Theoretical bound, will be same across all seeds

        plt.plot(xvals, avg_results, label="Empirical sinedistance", color="blue", marker='o')
        plt.fill_between(xvals, var_lower_bound, var_upper_bound, color="blue", alpha=0.3)
        plt.plot(xvals, theoretical_upper_bound, label="Theoretical upper bound", color="orange", marker='+')


    plt.xlabel(header[0])
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(f"{experiment}_{metric}.png")
    plt.clf()

if __name__ == "__main__":
    seeds = range(10)

    # experiment = "increase_n"
    # experiment = "increase_dimension_d"
    # experiment = "increase_dimension_d_and_r"
    # experiment = "increase_noise"
    # experiment = "increase_noise_homogenous"
    # experiment = "increase_dimension_r_model"
    # experiment = "bound_cl_with_d"
    experiment = "bound_cl_with_r"
    # experiment = "bound_cl_with_n"

    metric = "sinedistance"
    plot_graph(experiment, metric, seeds)
    metric = "downstream_score"
    plot_graph(experiment, metric, seeds)
