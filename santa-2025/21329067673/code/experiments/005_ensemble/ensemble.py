"""
Ensemble approach: Collect best configuration for each N from all available CSVs
"""
import os
import glob
import math
import pandas as pd
import numpy as np
from numba import njit
from tqdm import tqdm
from decimal import Decimal, getcontext
from shapely.geometry import Polygon
from shapely import affinity

getcontext().prec = 25

# Tree template
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]

@njit
def make_polygon_template():
    x = np.array(TX, np.float64)
    y = np.array(TY, np.float64)
    return x, y

@njit
def score_group(xs, ys, degs, tx, ty):
    n = xs.size
    V = tx.size
    mnx = 1e300
    mny = 1e300
    mxx = -1e300
    mxy = -1e300
    for i in range(n):
        r = degs[i] * math.pi / 180.0
        c = math.cos(r)
        s = math.sin(r)
        xi = xs[i]
        yi = ys[i]
        for j in range(V):
            X = c * tx[j] - s * ty[j] + xi
            Y = s * tx[j] + c * ty[j] + yi
            if X < mnx:
                mnx = X
            if X > mxx:
                mxx = X
            if Y < mny:
                mny = Y
            if Y > mxy:
                mxy = Y
    side = max(mxx - mnx, mxy - mny)
    return side * side / n

def strip(a):
    return np.array([float(str(v).replace("s", "")) for v in a], np.float64)

def create_tree_polygon(x, y, deg):
    initial_polygon = Polygon(list(zip(TX, TY)))
    rotated = affinity.rotate(initial_polygon, deg, origin=(0, 0))
    return affinity.translate(rotated, xoff=x, yoff=y)

def check_overlaps(xs, ys, degs):
    """Check if any trees overlap using Shapely"""
    trees = [create_tree_polygon(x, y, d) for x, y, d in zip(xs, ys, degs)]
    for i in range(len(trees)):
        for j in range(i+1, len(trees)):
            if trees[i].intersects(trees[j]) and not trees[i].touches(trees[j]):
                intersection = trees[i].intersection(trees[j])
                if intersection.area > 1e-15:
                    return True
    return False

# Find all CSV files
csv_dirs = [
    "/home/nonroot/snapshots/santa-2025/21116303805/code/preoptimized/",
    "/home/nonroot/snapshots/santa-2025/21116303805/code/experiments/",
    "/home/nonroot/snapshots/santa-2025/21328309254/code/experiments/",
    "/home/code/experiments/",
]

files = []
for d in csv_dirs:
    if os.path.exists(d):
        files += glob.glob(d + "**/*.csv", recursive=True)

files = sorted(set(files))
print(f"Found {len(files)} CSV files")

tx, ty = make_polygon_template()
best = {n: {"score": 1e300, "data": None, "src": None} for n in range(1, 201)}

for fp in tqdm(files, desc="scanning"):
    try:
        df = pd.read_csv(fp)
    except Exception:
        continue
    if not {"id", "x", "y", "deg"}.issubset(df.columns):
        continue
    df = df.copy()
    df["N"] = df["id"].astype(str).str.split("_").str[0].astype(int)
    for n, g in df.groupby("N"):
        if n < 1 or n > 200:
            continue
        xs = strip(g["x"].to_numpy())
        ys = strip(g["y"].to_numpy())
        ds = strip(g["deg"].to_numpy())
        
        # Check if we have the right number of trees
        if len(xs) != n:
            continue
            
        sc = score_group(xs, ys, ds, tx, ty)
        if sc < best[n]["score"]:
            # Validate no overlaps
            if not check_overlaps(xs, ys, ds):
                best[n]["score"] = sc
                best[n]["data"] = g[["id", "x", "y", "deg"]].copy()
                best[n]["src"] = fp

# Build ensemble
rows = []
total_score = 0
for n in range(1, 201):
    if best[n]["data"] is not None:
        rows.append(best[n]["data"])
        total_score += best[n]["score"]
        print(f"N={n}: score={best[n]['score']:.6f} from {best[n]['src']}")
    else:
        print(f"N={n}: NO VALID DATA FOUND!")

if rows:
    ensemble_df = pd.concat(rows, ignore_index=True)
    ensemble_df.to_csv("submission_ensemble.csv", index=False)
    print(f"\nTotal ensemble score: {total_score:.6f}")
    print(f"Saved to submission_ensemble.csv")
else:
    print("No valid data found!")
