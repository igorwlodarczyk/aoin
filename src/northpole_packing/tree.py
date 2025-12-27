from decimal import Decimal
from itertools import combinations
from shapely.geometry import Polygon
from shapely import affinity
from shapely.ops import unary_union

from northpole_packing.const import SCALE_FACTOR


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x="0", center_y="0", angle="0"):
        """Initializes the Christmas tree with a specific position and rotation."""
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)
        self.minx = None
        self.miny = None
        self.maxx = None
        self.maxy = None

        trunk_w = Decimal("0.15")
        trunk_h = Decimal("0.2")
        base_w = Decimal("0.7")
        mid_w = Decimal("0.4")
        top_w = Decimal("0.25")
        tip_y = Decimal("0.8")
        tier_1_y = Decimal("0.5")
        tier_2_y = Decimal("0.25")
        base_y = Decimal("0.0")
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                # Start at Tip
                (Decimal("0.0") * SCALE_FACTOR, tip_y * SCALE_FACTOR),
                # Right side - Top Tier
                (top_w / Decimal("2") * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
                (top_w / Decimal("4") * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
                # Right side - Middle Tier
                (mid_w / Decimal("2") * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
                (mid_w / Decimal("4") * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
                # Right side - Bottom Tier
                (base_w / Decimal("2") * SCALE_FACTOR, base_y * SCALE_FACTOR),
                # Right Trunk
                (trunk_w / Decimal("2") * SCALE_FACTOR, base_y * SCALE_FACTOR),
                (trunk_w / Decimal("2") * SCALE_FACTOR, trunk_bottom_y * SCALE_FACTOR),
                # Left Trunk
                (
                    -(trunk_w / Decimal("2")) * SCALE_FACTOR,
                    trunk_bottom_y * SCALE_FACTOR,
                ),
                (-(trunk_w / Decimal("2")) * SCALE_FACTOR, base_y * SCALE_FACTOR),
                # Left side - Bottom Tier
                (-(base_w / Decimal("2")) * SCALE_FACTOR, base_y * SCALE_FACTOR),
                # Left side - Middle Tier
                (-(mid_w / Decimal("4")) * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
                (-(mid_w / Decimal("2")) * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
                # Left side - Top Tier
                (-(top_w / Decimal("4")) * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
                (-(top_w / Decimal("2")) * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
            ]
        )
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated,
            xoff=float(self.center_x * SCALE_FACTOR),
            yoff=float(self.center_y * SCALE_FACTOR),
        )

    def get_params(self):
        return float(self.center_x), float(self.center_y), float(self.angle)

    def get_bounds(self):
        if (
            self.minx is None
            or self.miny is None
            or self.maxx is None
            or self.maxy is None
        ):
            tree_bounds = self.polygon.bounds
            self.minx = Decimal(tree_bounds[0]) / SCALE_FACTOR
            self.miny = Decimal(tree_bounds[1]) / SCALE_FACTOR
            self.maxx = Decimal(tree_bounds[2]) / SCALE_FACTOR
            self.maxy = Decimal(tree_bounds[3]) / SCALE_FACTOR

        return self.minx, self.miny, self.maxx, self.maxy


def has_collision(trees):
    for t1, t2 in combinations(trees, 2):
        if t1.polygon.intersects(t2.polygon) and not t1.polygon.touches(t2.polygon):
            return True
    return False


def has_collision_with_candidate(trees, candidate):
    cand_poly = candidate.polygon
    for tree in trees:
        poly = tree.polygon
        if cand_poly.intersects(poly) and not cand_poly.touches(poly):
            return True
    return False


def calculate_side_length(trees):
    all_polygons = [t.polygon for t in trees]
    bounds = unary_union(all_polygons).bounds

    minx = Decimal(bounds[0]) / SCALE_FACTOR
    miny = Decimal(bounds[1]) / SCALE_FACTOR
    maxx = Decimal(bounds[2]) / SCALE_FACTOR
    maxy = Decimal(bounds[3]) / SCALE_FACTOR

    width = maxx - minx
    height = maxy - miny
    side_length = max(width, height)
    return side_length


def convert_trees_to_string(trees):
    return "|".join(",".join(map(str, tree.get_params())) for tree in trees)


def load_trees_from_string(trees_str):
    trees = []
    for tree_str in trees_str.split("|"):
        tree_obj = ChristmasTree(*tree_str.split(","))
        trees.append(tree_obj)
    return trees
