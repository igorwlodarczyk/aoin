import math
import random
from decimal import Decimal

from northpole_packing.tree import (
    ChristmasTree,
    has_collision,
    has_collision_with_candidate,
)
from northpole_packing.const import PRECISION, SCALE_FACTOR


def initialize_trees_random(num_trees, max_attempts=1000, box_pct=0.95):
    def calculate_min_rectangle_size_for_single_tree():
        tree = ChristmasTree("0", "0", "0")
        minx, miny, maxx, maxy = tree.polygon.bounds
        width = float(Decimal(maxx - minx) / SCALE_FACTOR)
        height = float(Decimal(maxy - miny) / SCALE_FACTOR)
        return width, height

    def find_starting_rectangle_size(num_trees):
        tree_width, tree_height = calculate_min_rectangle_size_for_single_tree()
        min_square_height = float("inf")
        for c in range(1, num_trees + 1):
            r = math.ceil(num_trees / c)
            W = c * tree_width
            H = tree_height * r
            square_height = max(H, W)
            if square_height < min_square_height:
                min_square_height = square_height
        return min_square_height

    start_rectangle_height = find_starting_rectangle_size(num_trees)
    grid_size = (start_rectangle_height / 2) * box_pct
    for attempt in range(max_attempts):
        trees = []
        for tree in range(num_trees):
            for tree_attempt in range(max_attempts):
                x = round(random.uniform(-grid_size, grid_size), PRECISION)
                y = round(random.uniform(-grid_size, grid_size), PRECISION)
                angle = round(random.uniform(0, 360), 1)

                tree_candidate = ChristmasTree(
                    center_x=str(x), center_y=str(y), angle=str(angle)
                )
                trees.append(tree_candidate)

                if has_collision(trees):
                    trees.pop()
                else:
                    break
            if tree + 1 != len(trees):
                break
        if len(trees) == num_trees:
            return trees
    raise Exception


def greedy_initialization(
    num_trees: int,
    tree_angles: list | None = None,
    starting_dist=0.4,
    step=0.1,
    search_depth: int = 5,
    optimize: bool = True,
):
    def calculate_side_length_from_bounds(minx, miny, maxx, maxy):
        width = maxx - minx
        height = maxy - miny
        side_length = max(width, height)
        return side_length

    def recalculate_bounds(minx, miny, maxx, maxy, tree_candidate):
        candidate_minx, candidate_miny, candidate_maxx, candidate_maxy = (
            tree_candidate.get_bounds()
        )
        return (
            min(candidate_minx, minx),
            min(candidate_miny, miny),
            max(candidate_maxx, maxx),
            max(candidate_maxy, maxy),
        )

    if tree_angles is None:
        tree_angles = [random.uniform(0, 360) for _ in range(num_trees)]

    trees = []
    first_tree = ChristmasTree("0", "0", str(tree_angles[0]))
    trees.append(first_tree)

    minx, miny, maxx, maxy = first_tree.get_bounds()

    for idx, tree_angle in enumerate(tree_angles[1:]):
        tree_placed = False
        dist = starting_dist
        trees_to_explore = trees
        if optimize:
            if len(trees_to_explore) > 30:
                center_x, center_y = (minx + maxx / 2), (miny + maxy) / 2
                sorted_trees = sorted(
                    trees_to_explore,
                    key=lambda s: (s.center_x - center_x) ** 2
                    + (s.center_y - center_y) ** 2,
                )

                k = int(len(sorted_trees) * 0.6)
                trees_to_explore = sorted_trees[-k:]
        while not tree_placed:
            tree_x = None
            tree_y = None
            min_side = float("inf")
            tmp_dist = dist
            for _ in range(search_depth):
                points = [
                    (tmp_dist, 0),
                    (-tmp_dist, 0),
                    (0, tmp_dist),
                    (0, -tmp_dist),
                    (tmp_dist, tmp_dist),
                    (tmp_dist, -tmp_dist),
                    (-tmp_dist, tmp_dist),
                    (-tmp_dist, -tmp_dist),
                ]

                for tree in trees_to_explore:
                    x, y, _ = tree.get_params()
                    for point_x, point_y in points:
                        x_candidate = x + point_x
                        y_candidate = y + point_y

                        tree_candidate = ChristmasTree(
                            str(x_candidate), str(y_candidate), str(tree_angle)
                        )

                        if not has_collision_with_candidate(trees, tree_candidate):
                            side = calculate_side_length_from_bounds(
                                *recalculate_bounds(
                                    minx, miny, maxx, maxy, tree_candidate
                                )
                            )
                            if side < min_side:
                                min_side = side
                                tree_x = x_candidate
                                tree_y = y_candidate

                tmp_dist += step

            if tree_x is not None and tree_y is not None:
                tree = ChristmasTree(str(tree_x), str(tree_y), str(tree_angle))
                trees.append(tree)
                minx, miny, maxx, maxy = recalculate_bounds(
                    minx, miny, maxx, maxy, tree
                )
                tree_placed = True
            else:
                dist += step * search_depth
    return trees
