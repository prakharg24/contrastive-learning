import csv
import matplotlib.pyplot as plt
if __name__ == "__main__":
    experiment = "increase_noise"
    # metric = "sinedistance"
    metric = "downstream_score"
    rows = []
    with open(f"{experiment}_{metric}.csv") as f:
        csvreader = csv.reader(f)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)
    plt.plot(header[1:], [float(i) for i in rows[0][1:]], label=str(rows[0][0]))
    plt.plot(header[1:], [float(i) for i in rows[1][1:]], label=str(rows[1][0]))
    plt.xlabel(header[0])
    plt.ylabel(metric)
    plt.legend()
    plt.show()