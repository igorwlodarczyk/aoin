from pathlib import Path
import csv

from northpole_packing.tree import load_trees_from_string

RESULTS_DIR = Path("/Users/igor/PWR/aoin/kaggle_entry")
OUT_PATH = RESULTS_DIR / "submission.csv"

PREC = 6

def fmt_s(val: float, prec: int = PREC) -> str:
    return f"s{val:.{prec}f}"

dirs = sorted(
    [entry for entry in RESULTS_DIR.iterdir() if entry.is_dir()],
    key=lambda p: int(p.name),
)

with open(OUT_PATH, "w", newline="") as out_f:
    writer = csv.writer(out_f)
    writer.writerow(["id", "x", "y", "deg"])

    for d in dirs:
        log_path = d / "log.csv"
        if not log_path.exists():
            continue

        last_line = log_path.read_text().splitlines()[-1]
        treestr = last_line.split(";")[-1].strip()

        trees = load_trees_from_string(treestr)

        instance_id = int(d.name)
        for i, tree in enumerate(trees):
            x, y, deg = tree.get_params()
            row_id = f"{instance_id:03d}_{i}"
            writer.writerow([row_id, fmt_s(x), fmt_s(y), fmt_s(deg)])

print(f"Wrote: {OUT_PATH}")
