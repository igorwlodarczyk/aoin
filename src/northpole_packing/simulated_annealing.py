import copy
import random
import numpy as np
import math
import time

from northpole_packing.initialization import greedy_initialization
from northpole_packing.const import PRECISION
from northpole_packing.tree import (
    ChristmasTree,
    has_collision_with_candidate,
    calculate_side_length,
    convert_trees_to_string,
)


class SimulatedAnnealing:
    def __init__(
        self, output_log_path, num_trees=100, alpha=0.992, min_t=0.001, start_temp=1000
    ):
        self.output_log_path = output_log_path
        self.num_trees = num_trees
        self.alpha = alpha
        self.min_t = min_t
        self.start_temp = start_temp

    @staticmethod
    def generate_neighbor_sol(trees, max_attempts=1000):
        for attempt in range(max_attempts):
            trees_candidate = copy.deepcopy(trees)
            tree = random.choice(trees_candidate)
            trees_candidate.remove(tree)

            x, y, angle = tree.get_params()
            param_to_change = random.choice(["x", "y", "angle"])
            if param_to_change == "x":
                x += np.random.normal(0, 0.2)
                x = round(x, PRECISION)
            elif param_to_change == "y":
                y += np.random.normal(0, 0.2)
                y = round(y, PRECISION)
            elif param_to_change == "angle":
                angle = (angle + np.random.normal(0, 30)) % 360
                angle = round(angle, 1)

            new_tree = ChristmasTree(str(x), str(y), str(angle))
            if not has_collision_with_candidate(trees_candidate, new_tree):
                trees_candidate.append(new_tree)
                return trees_candidate
        return trees

    def solve(self):
        start_time = time.time()
        best_solution = greedy_initialization(num_trees=self.num_trees)
        end_time = time.time()
        print(f"Initialized starting solution using greedy algorithm: {round(end_time - start_time, 2)} s.")
        best_solution_cost = calculate_side_length(best_solution)
        current_solution = copy.deepcopy(best_solution)
        current_cost = best_solution_cost
        current_temp = self.start_temp

        iter = 1
        with open(self.output_log_path, "w") as output_log:
            while current_temp > self.min_t:
                neighbor_solutions = [
                    self.generate_neighbor_sol(current_solution) for _ in range(10)
                ]
                neighbor_sol = min(neighbor_solutions, key=calculate_side_length)
                neighbor_cost = calculate_side_length(neighbor_sol)
                delta = float(neighbor_cost - current_cost)
                if delta < 0 or random.random() < math.exp(-delta / current_temp):
                    current_solution = neighbor_sol
                    current_cost = neighbor_cost
                    if current_cost < best_solution_cost:
                        best_solution = copy.deepcopy(current_solution)
                        best_solution_cost = current_cost

                best_solution_str = convert_trees_to_string(best_solution)
                output_log.write(
                    f"{iter};{current_temp};{best_solution_cost};{current_cost};{best_solution_str}\n"
                )
                output_log.flush()
                current_temp *= self.alpha
                iter += 1
            return best_solution, best_solution_cost
