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

def plot_hc(log_path: str, title: str = "HC"):
    iteration = []
    best = []
    last_line = None
    with open(log_path, "r") as f:
        for line in f:
            data = line.split(";")
            iter_l = int(data[0])
            best_l = float(data[1])
            iteration.append(iter_l)
            best.append(best_l)
            last_line = line
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(iteration, best)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best solution")
    ax.set_title(title)
    fig.legend()
    plt.show()
    trees_str = last_line.split(";")[-1]
    trees = load_trees_from_string(trees_str)
    plot_results(trees)

def plot_ts(log_path: str, title: str = "TS"):
    iteration = []
    best = []
    last_line = None
    with open(log_path, "r") as f:
        for line in f:
            data = line.split(";")
            iter_l = int(data[0])
            best_l = float(data[2])
            iteration.append(iter_l)
            best.append(best_l)
            last_line = line
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(iteration, best)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best solution")
    ax.set_title(title)
    fig.legend()
    plt.show()
    trees_str = last_line.split(";")[-1]
    trees = load_trees_from_string(trees_str)
    plot_results(trees)

def _read_best_series(log_path: str, best_col: int, iter_col: int = 0):
    iters = []
    best = []
    with open(log_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = line.strip().split(";")
            if len(data) <= max(best_col, iter_col):
                continue

            try:
                it = int(float(data[iter_col]))
                b = float(data[best_col])
            except ValueError:
                continue

            iters.append(it)
            best.append(b)

    steps = list(range(1, len(best) + 1))
    return steps, best


def compare_solutions(sa_path: str, hc_path: str, ts_path: str, title: str = "Best solution over iterations"):
    sa_x, sa_best = _read_best_series(sa_path, best_col=2, iter_col=0)

    hc_x, hc_best = _read_best_series(hc_path, best_col=1, iter_col=0)

    ts_x, ts_best = _read_best_series(ts_path, best_col=2, iter_col=0)

    plt.figure(figsize=(10, 6))
    plt.plot(sa_x, sa_best, label="SA (best)")
    plt.plot(hc_x, hc_best, label="HC (best)")
    plt.plot(ts_x, ts_best, label="TS (best)")

    plt.xlabel("Iteration (step)")
    plt.ylabel("Best solution")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_sa2(log_path: str, title: str = "SA"):
    iteration = []
    temperature = []
    best = []
    current = []
    last_line = None

    with open(log_path, "r") as f:
        for line in f:
            data = line.split(";")
            iter_l = int(data[0])
            temp_l = float(data[1])
            best_l = float(data[2])
            current_l = float(data[3])

            iteration.append(iter_l)
            temperature.append(temp_l)
            best.append(best_l)
            current.append(current_l)

            last_line = line

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(iteration, temperature, label="Temperature", alpha=0.8, color="red")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Temperature")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.plot(iteration, best, label="Best value", linestyle="--", alpha=0.8)
    ax2.plot(iteration, current, label="Current value", linestyle=":", alpha=0.8)
    ax2.set_ylabel("Objective value")
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