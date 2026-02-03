import time
from pathlib import Path
from copy import deepcopy
from northpole_packing.tree import load_trees_from_string, calculate_side_length
from northpole_packing.visualization import plot_results
from northpole_packing.initialization import greedy_initialization
from northpole_packing.simulated_annealing import SimulatedAnnealing

output_dir = Path("/Users/igor/PWR/aoin/kaggle_entry")
output_dir.mkdir(exist_ok=True)

base_solution_file = "/Users/igor/PWR/aoin/data/output_saa_77b06160110c44d9a202bc3bd4a804c6.log"
with open(base_solution_file, "r") as base:
    treestr = base.readlines()[-1].split(";")[-1]
    base_solution = load_trees_from_string(trees_str=treestr)
    base_solution_cost = calculate_side_length(base_solution)

def greedy_trim_solution(trees, num_trees):
    copy_trees = deepcopy(trees)
    if len(trees) == num_trees:
        return copy_trees

    if num_trees > len(trees):
        return greedy_initialization(num_trees=num_trees, init_trees=copy_trees)
    else:
        while len(copy_trees) != num_trees:
            best_side = float("inf")
            best_remove_idx = None

            for i in range(len(copy_trees)):
                candidate = copy_trees[:i] + copy_trees[i + 1:]
                side = calculate_side_length(candidate)

                if side < best_side:
                    best_side = side
                    best_remove_idx = i
            copy_trees.pop(best_remove_idx)

        return copy_trees


current_score = 0

for num_tree in range(1, 201):
    solution_dir = output_dir / str(num_tree)
    if solution_dir.exists():
        with open(solution_dir / "log.csv", "r") as f:
            score = float(f.readlines()[-1].split(";")[2])
            current_score += score ** 2
    else:
        solution_dir.mkdir(exist_ok=True)

        start_time = time.time()
        print(f"num_tree={num_tree}")
        trees = greedy_trim_solution(base_solution, num_trees=num_tree)
        sa = SimulatedAnnealing(
            output_log_path=solution_dir / "log.csv",
            alpha=0.99986,
            start_temp=5000,
            min_t=3.78e3,
            init_trees=trees
        )
        best_solution, best_solution_cost = sa.solve()
        plot_results(best_solution, output_file=solution_dir / "output")

        if len(best_solution) > len(base_solution):
            base_solution = best_solution
        end_time = time.time()
        print(f"time: {round(end_time - start_time, 2)} s")
        current_score += float(best_solution_cost) ** 2
    print("current_score: ", current_score / num_tree)

final_score = current_score / 200
with open(output_dir / "score.txt", "w") as f:
    f.write(str(final_score))