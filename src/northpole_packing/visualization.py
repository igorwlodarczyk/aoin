import matplotlib.pyplot as plt
import shutil
import glob
from matplotlib.patches import Rectangle
from decimal import Decimal
from shapely.ops import unary_union
from pathlib import Path
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from northpole_packing.const import SCALE_FACTOR
from northpole_packing.tree import load_trees_from_string, calculate_side_length


def plot_results(side_length, placed_trees, output_file=None, mono_color=False):
    """Plots the arrangement of trees and the bounding square."""
    num_trees = len(placed_trees)
    _, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.viridis([i / num_trees for i in range(num_trees)])

    all_polygons = [t.polygon for t in placed_trees]
    bounds = unary_union(all_polygons).bounds

    for i, tree in enumerate(placed_trees):
        # Rescale for plotting
        x_scaled, y_scaled = tree.polygon.exterior.xy
        x = [Decimal(val) / SCALE_FACTOR for val in x_scaled]
        y = [Decimal(val) / SCALE_FACTOR for val in y_scaled]
        if mono_color:
            color = "green"
        else:
            color = colors[i]
        ax.plot(x, y, color=color)
        ax.fill(x, y, alpha=0.5, color=color)

    minx = Decimal(bounds[0]) / SCALE_FACTOR
    miny = Decimal(bounds[1]) / SCALE_FACTOR
    maxx = Decimal(bounds[2]) / SCALE_FACTOR
    maxy = Decimal(bounds[3]) / SCALE_FACTOR

    width = maxx - minx
    height = maxy - miny

    square_x = minx if width >= height else minx - (side_length - width) / 2
    square_y = miny if height >= width else miny - (side_length - height) / 2
    bounding_square = Rectangle(
        (float(square_x), float(square_y)),
        float(side_length),
        float(side_length),
        fill=False,
        edgecolor="red",
        linewidth=2,
        linestyle="--",
    )
    ax.add_patch(bounding_square)

    padding = 0.5
    ax.set_xlim(
        float(square_x - Decimal(str(padding))),
        float(square_x + side_length + Decimal(str(padding))),
    )
    ax.set_ylim(
        float(square_y - Decimal(str(padding))),
        float(square_y + side_length + Decimal(str(padding))),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    plt.title(f"{num_trees} Trees: {side_length:.12f}")
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()
    plt.close()


def create_progress_video(output_file):
    tmp_dir = Path("tmp_video_dir")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    tmp_dir.mkdir()
    with open(output_file, "r") as output:
        log = output.read()

    previous_best_tree_str = None
    tree_idx = 1
    for line in log.splitlines():
        trees_str = line.split(";")[-1]
        if previous_best_tree_str is None or (
            previous_best_tree_str != trees_str and previous_best_tree_str is not None
        ):
            trees = load_trees_from_string(trees_str)
            side = calculate_side_length(trees)
            plot_results(side, trees, tmp_dir / str(tree_idx), mono_color=True)
            previous_best_tree_str = trees_str
            tree_idx += 1

    paths = sorted(
        glob.glob(f"{tmp_dir}/*.png"),
        key=lambda entry: int(entry.split("/")[-1].replace(".png", "")),
    )
    fps = 5
    clip = ImageSequenceClip(paths, fps=fps)
    clip.write_videofile("out.mp4", codec="libx264", audio=False, fps=fps)
    shutil.rmtree(tmp_dir)
