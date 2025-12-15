import random
import copy
import math
import numpy as np

from northpole_packing.tree import (
    ChristmasTree,
    has_collision,
    calculate_side_length,
    convert_trees_to_string,
)
from northpole_packing.initialization import (
    initialize_trees_random,
    find_starting_rectangle_size,
)
from northpole_packing.const import PRECISION


def gaussian_step(tree, grid_size, px=1, sigma=0.1, angle_delta=30.0):
    x, y, angle = tree.get_params()
    if random.uniform(0, 1) < px:
        new_x = x + np.random.normal(0, sigma)
        if new_x > grid_size or new_x < -grid_size:
            new_x = x
    else:
        new_x = x
    if random.uniform(0, 1) < px:
        new_y = y + np.random.normal(0, sigma)
        if new_y > grid_size or new_y < -grid_size:
            new_y = y
    else:
        new_y = y

    new_angle = (angle + random.uniform(-angle_delta, angle_delta)) % 360
    new_angle = round(new_angle, 1)
    new_x = round(new_x, PRECISION)
    new_y = round(new_y, PRECISION)
    new_tree = ChristmasTree(str(new_x), str(new_y), str(new_angle))
    return new_tree


def generate_neighbor_trees_solution(trees, grid_size, max_attempts=1000):
    for attempt in range(max_attempts):
        trees_candidate = copy.deepcopy(trees)
        tree = random.choice(trees_candidate)
        trees_candidate.remove(tree)
        new_tree = gaussian_step(tree, grid_size)
        trees_candidate.append(new_tree)
        if not has_collision(trees_candidate):
            return trees_candidate
    return trees


def generate_neighbor_trees_solution_multiplied(
    trees, grid_size, max_attempts=1000, multiplier=10
):
    neighbor_sol = copy.deepcopy(trees)
    for _ in range(multiplier):
        neighbor_sol = generate_neighbor_trees_solution(
            neighbor_sol, grid_size, max_attempts
        )
    return neighbor_sol


class SimulatedAnnealing:
    def __init__(self, num_trees, alpha=0.992, min_t=0.001, start_temp=1000):
        self.num_trees = num_trees
        self.alpha = alpha
        self.min_t = min_t
        self.start_temp = start_temp

    def solve(self, output_log_file_path):
        grid_size = find_starting_rectangle_size(self.num_trees) / 2
        best_solution = initialize_trees_random(self.num_trees)
        best_solution_cost = calculate_side_length(best_solution)
        best_solution_compactness = self.calculate_compactness(best_solution)
        current_solution = copy.deepcopy(best_solution)
        current_cost = best_solution_cost
        current_temp = self.start_temp
        with open(output_log_file_path, "w") as output_log:
            while current_temp > self.min_t:
                best_solution_str = convert_trees_to_string(best_solution)
                output_log.write(
                    f"{current_temp};{best_solution_cost};{best_solution_str}\n"
                )
                neighbor_sol = generate_neighbor_trees_solution(
                    current_solution, grid_size
                )
                neighbor_cost = calculate_side_length(neighbor_sol)
                delta = float(neighbor_cost - current_cost)
                if delta == 0:
                    neighbor_sol_compactness = self.calculate_compactness(neighbor_sol)
                    if neighbor_sol_compactness < best_solution_compactness:
                        current_solution = neighbor_sol
                        current_cost = neighbor_cost
                        if current_cost < best_solution_cost:
                            best_solution = copy.deepcopy(current_solution)
                            best_solution_cost = current_cost
                        continue

                if delta < 0 or (
                    random.random() < math.exp(-delta / current_temp) and False
                ):
                    current_solution = neighbor_sol
                    current_cost = neighbor_cost
                    if current_cost < best_solution_cost:
                        best_solution = copy.deepcopy(current_solution)
                        best_solution_cost = current_cost
                current_temp *= self.alpha

        return best_solution, best_solution_cost

    @staticmethod
    def calculate_compactness(trees):
        xs = np.array([t.get_params()[0] for t in trees], dtype=float)
        ys = np.array([t.get_params()[1] for t in trees], dtype=float)
        cx, cy = xs.mean(), ys.mean()
        compactness = float(np.sum((xs - cx) ** 2 + (ys - cy) ** 2))
        return compactness
