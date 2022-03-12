import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--outfile", default="output.txt", help="File with Output")

args = parser.parse_args()

all_output = open(args.outfile, 'r').read().split("\n")

all_results = []
for line_number in range(0, len(all_output), 5):
    if all_output[line_number]=="":
        continue
    hyperparameters = all_output[line_number].split(",")
    d, train_size = int(hyperparameters[0]), int(hyperparameters[1])
    sine_distance = float(all_output[line_number+1].split(":")[-1].strip())
    cls_acc = float(all_output[line_number+2].split(":")[-1].strip())
    reg_rmse = float(all_output[line_number+4].split(":")[-1].strip())
    all_results.append(d, train_size, sine_distance, cls_acc, reg_rmse)

d_results = all_results[:11]
train_size_results = all_results[11:]
## Plot graphs
# plt.plot()
