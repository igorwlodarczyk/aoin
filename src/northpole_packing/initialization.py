import math
import random
from decimal import Decimal

from northpole_packing.tree import ChristmasTree, has_collision, calculate_side_length
from northpole_packing.const import PRECISION, SCALE_FACTOR


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


def initialize_trees_random(num_trees, max_attempts=1000, box_pct=0.95):
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
