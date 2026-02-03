import copy
import random
import math
import numpy as np
from decimal import Decimal


from northpole_packing.initialization import greedy_initialization
from northpole_packing.const import PRECISION
from northpole_packing.tree import (
    ChristmasTree,
    has_collision_with_candidate,
    calculate_side_length,
    convert_trees_to_string,
)


def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


class HillClimbing:
    def __init__(self, output_log_path, num_trees=100, max_iter: int = 150_000):
        self.output_log_path = output_log_path
        self.num_trees = num_trees
        self.max_iter = max_iter

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

    @staticmethod
    def generate_neighbor_sol_greedy(trees, max_attempts=1000):
        n = Decimal(len(trees))
        cx = float(sum(t.center_x for t in trees) / n)
        cy = float(sum(t.center_y for t in trees) / n)

        for attempt in range(max_attempts):
            trees_candidate = copy.deepcopy(trees)
            tree = random.choice(trees_candidate)
            trees_candidate.remove(tree)

            x, y, angle = tree.get_params()
            initial_dist = distance((x, y), (cx, cy))
            old_x, old_y = x, y
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
            if param_to_change != "angle":
                new_dist = distance((old_x, old_y), (cx, cy))
                if new_dist > initial_dist:
                    continue
            new_tree = ChristmasTree(str(x), str(y), str(angle))
            if not has_collision_with_candidate(trees_candidate, new_tree):
                trees_candidate.append(new_tree)
                return trees_candidate
        return trees

    def solve(self):
        best_solution = greedy_initialization(num_trees=self.num_trees)
        best_solution_cost = calculate_side_length(best_solution)
        with open(self.output_log_path, "w") as output_log:
            for iter in range(self.max_iter):
                neighbor_solutions = [
                    self.generate_neighbor_sol(best_solution) for _ in range(5)
                ]
                neighbor_solutions_greedy = [
                    self.generate_neighbor_sol_greedy(best_solution) for _ in range(5)
                ]
                neighbor_solutions.extend(neighbor_solutions_greedy)
                neighbor_sol, neighbor_sol_cost = min(
                    zip(
                        neighbor_solutions,
                        [calculate_side_length(trees) for trees in neighbor_solutions],
                    ),
                    key=lambda s: s[1],
                )
                if neighbor_sol_cost <= best_solution_cost:
                    best_solution = neighbor_sol
                    best_solution_cost = neighbor_sol_cost
                best_solution_str = convert_trees_to_string(best_solution)
                output_log.write(
                    f"{iter + 1};{best_solution_cost};{best_solution_str}\n"
                )
                output_log.flush()
        return best_solution, best_solution_cost
