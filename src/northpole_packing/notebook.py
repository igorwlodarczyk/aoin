import matplotlib.pyplot as plt
from northpole_packing.tree import load_trees_from_string
from northpole_packing.visualization import plot_results


def plot_ga(log_path: str, title: str = "GA"):
    epoch = []
    best = []
    avg = []
    worst = []

    with open(log_path, "r") as f:
        for line in f:
            data = line.split(";")
            epoch_l = int(data[0]) + 1
            best_l = float(data[1])
            avg_l = float(data[2])
            worst_l = float(data[3])

            epoch.append(epoch_l)
            best.append(best_l)
            avg.append(avg_l)
            worst.append(worst_l)

    plt.figure(figsize=(10, 6))
    plt.plot(epoch, best, label="Best")
    plt.plot(epoch, avg, label="Average")
    plt.plot(epoch, worst, label="Worst")

    plt.xlabel("Epoch")
    plt.ylabel("Fitness")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_sa(log_path: str, title: str = "SA"):
    iteration = []
    temperature = []
    best = []
    last_line = None
    with open(log_path, "r") as f:
        for line in f:
            data = line.split(";")
            iter_l = int(data[0])
            temp_l = float(data[1])
            best_l = float(data[2])

            iteration.append(iter_l)
            temperature.append(temp_l)
            best.append(best_l)
            last_line = line

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(iteration, temperature, label="Temperature", alpha=0.8, color="red")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Temperature")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.plot(iteration, best, label="Best value", linestyle="--", alpha=0.8)
    ax2.set_ylabel("Best value")
    ax2.tick_params(axis="y")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    trees_str = last_line.split(";")[-1]
    trees = load_trees_from_string(trees_str)
    plot_results(trees)


