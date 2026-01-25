"""
Simple Ensemble - Uses baseline and scans for valid improvements
Skips any malformed data
"""
import pandas as pd
import numpy as np
from numba import njit
import math
import glob
from tqdm import tqdm
from shapely.geometry import Polygon
from shapely import affinity

# Tree shape
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]

@njit
def make_polygon_template():
    tw=0.15; th=0.2; bw=0.7; mw=0.4; ow=0.25
    tip=0.8; t1=0.5; t2=0.25; base=0.0; tbot=-th
    x=np.array([0,ow/2,ow/4,mw/2,mw/4,bw/2,tw/2,tw/2,-tw/2,-tw/2,-bw/2,-mw/4,-mw/2,-ow/4,-ow/2],np.float64)
    y=np.array([tip,t1,t1,t2,t2,base,base,tbot,tbot,base,base,t2,t2,t1,t1],np.float64)
    return x,y

@njit
def score_group(xs, ys, degs, tx, ty):
    n = xs.size
    V = tx.size
    mnx = 1e300; mny = 1e300; mxx = -1e300; mxy = -1e300
    for i in range(n):
        r = degs[i] * math.pi / 180.0
        c = math.cos(r); s = math.sin(r)
        xi = xs[i]; yi = ys[i]
        for j in range(V):
            X = c * tx[j] - s * ty[j] + xi
            Y = s * tx[j] + c * ty[j] + yi
            if X < mnx: mnx = X
            if X > mxx: mxx = X
            if Y < mny: mny = Y
            if Y > mxy: mxy = Y
    side = max(mxx - mnx, mxy - mny)
    return side * side / n

def strip(a):
    return np.array([float(str(v).replace("s", "")) for v in a], np.float64)

def make_tree_polygon(x, y, deg):
    try:
        x = float(str(x).replace('s', ''))
        y = float(str(y).replace('s', ''))
        deg = float(str(deg).replace('s', ''))
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(deg)):
            return None
        initial_polygon = Polygon(list(zip(TX, TY)))
        rotated = affinity.rotate(initial_polygon, deg, origin=(0, 0))
        return affinity.translate(rotated, xoff=x, yoff=y)
    except:
        return None

def has_overlap(data):
    """Check if configuration has overlaps"""
    try:
        trees = []
        for _, r in data.iterrows():
            t = make_tree_polygon(r['x'], r['y'], r['deg'])
            if t is None:
                return True  # Invalid data = treat as overlap
            trees.append(t)
        
        for i in range(len(trees)):
            for j in range(i+1, len(trees)):
                if trees[i].intersects(trees[j]) and not trees[i].touches(trees[j]):
                    intersection = trees[i].intersection(trees[j])
                    if intersection.area > 1e-15:
                        return True
        return False
    except:
        return True  # Any error = treat as overlap

# Load baseline
baseline_path = "/home/nonroot/snapshots/santa-2025/21116303805/code/preoptimized/santa-2025.csv"
baseline = pd.read_csv(baseline_path)
tx, ty = make_polygon_template()

# Initialize with baseline
best = {}
for n in range(1, 201):
    base_data = baseline[baseline['id'].str.startswith(f'{n:03d}_')]
    xs = strip(base_data["x"].to_numpy())
    ys = strip(base_data["y"].to_numpy())
    ds = strip(base_data["deg"].to_numpy())
    sc = score_group(xs, ys, ds, tx, ty)
    best[n] = {"score": float(sc), "data": base_data.copy(), "src": "baseline"}

baseline_total = sum(best[n]["score"] for n in range(1, 201))
print(f"Baseline total score: {baseline_total:.6f}")

# Override N=1 with optimal value
manual_data = pd.DataFrame({
    "id": ["001_0"],
    "x": ["s0.0"],
    "y": ["s0.0"],
    "deg": ["s45.0"]
})
xs = strip(manual_data["x"].to_numpy())
ys = strip(manual_data["y"].to_numpy())
ds = strip(manual_data["deg"].to_numpy())
sc = score_group(xs, ys, ds, tx, ty)
best[1]["score"] = float(sc)
best[1]["data"] = manual_data
best[1]["src"] = "manual"

# Find all CSV files - but limit to known good directories
csv_dirs = [
    "/home/nonroot/snapshots/santa-2025/21116303805/code/preoptimized/",
    "/home/nonroot/snapshots/santa-2025/21116303805/code/experiments/",
    "/home/nonroot/snapshots/santa-2025/21328309254/code/experiments/",
    "/home/code/experiments/",
]

all_csvs = []
for d in csv_dirs:
    all_csvs.extend(glob.glob(d + "**/*.csv", recursive=True))
all_csvs = list(set(all_csvs))
print(f"Found {len(all_csvs)} CSV files in known good directories")

# Scan for improvements
improvements = 0
for fp in tqdm(all_csvs, desc="Scanning"):
    try:
        df = pd.read_csv(fp)
    except Exception:
        continue
    if not {"id", "x", "y", "deg"}.issubset(df.columns):
        continue
    
    df = df.copy()
    try:
        df["N"] = df["id"].astype(str).str.split("_").str[0].astype(int)
    except:
        continue
    
    for n, g in df.groupby("N"):
        if n < 1 or n > 200:
            continue
        if len(g) != n:
            continue
        
        try:
            xs = strip(g["x"].to_numpy())
            ys = strip(g["y"].to_numpy())
            ds = strip(g["deg"].to_numpy())
            
            if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys)) and np.all(np.isfinite(ds))):
                continue
            
            sc = score_group(xs, ys, ds, tx, ty)
            
            # Only consider if better than current best
            if sc < best[n]["score"] - 1e-10:
                g_clean = g.drop(columns=["N"]).copy()
                if not has_overlap(g_clean):
                    best[n]["score"] = float(sc)
                    best[n]["data"] = g_clean
                    best[n]["src"] = fp
                    improvements += 1
        except:
            continue

print(f"\nFound {improvements} valid improvements over baseline")

# Calculate total score
total_score = sum(best[n]["score"] for n in range(1, 201))
print(f"Total validated score: {total_score:.6f}")
print(f"Improvement from baseline: {baseline_total - total_score:.6f}")

# Build output
rows = []
for n in range(1, 201):
    data = best[n]["data"]
    for _, row in data.iterrows():
        rows.append({
            "id": row["id"],
            "x": row["x"],
            "y": row["y"],
            "deg": row["deg"]
        })

df_out = pd.DataFrame(rows)
df_out.to_csv("submission_simple.csv", index=False)
print(f"Saved to submission_simple.csv")

# Copy to submission
import shutil
shutil.copy("submission_simple.csv", "/home/submission/submission.csv")
print("Copied to /home/submission/submission.csv")

# Save metrics
import json
metrics = {
    "cv_score": total_score,
    "baseline_score": baseline_total,
    "improvement": baseline_total - total_score
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
