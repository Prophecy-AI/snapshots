"""
Comprehensive Ensemble - Scans ALL 3330 CSV files
Picks best score per N, validates with strict overlap detection
Falls back to baseline if overlap detected
"""
import pandas as pd
import numpy as np
from numba import njit
import math
import glob
import os
from tqdm import tqdm

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

# Find ALL CSV files
print("Finding all CSV files...")
all_csvs = glob.glob("/home/nonroot/snapshots/santa-2025/**/*.csv", recursive=True)
all_csvs += glob.glob("/home/code/experiments/**/*.csv", recursive=True)
all_csvs = list(set(all_csvs))
print(f"Found {len(all_csvs)} CSV files")

tx, ty = make_polygon_template()
best = {n: {"score": 1e300, "data": None, "src": None} for n in range(1, 201)}

# Scan all files
for fp in tqdm(all_csvs, desc="Scanning"):
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
        if len(g) != n:
            continue  # Invalid config
        
        xs = strip(g["x"].to_numpy())
        ys = strip(g["y"].to_numpy())
        ds = strip(g["deg"].to_numpy())
        sc = score_group(xs, ys, ds, tx, ty)
        
        if sc < best[n]["score"]:
            best[n]["score"] = float(sc)
            best[n]["data"] = g.drop(columns=["N"]).copy()
            best[n]["src"] = fp

# Override N=1 with optimal value
print("\nOverriding N=1 with optimal (0, 0, 45°)...")
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

# Calculate total score
total_score = sum(best[n]["score"] for n in range(1, 201))
print(f"\nTotal ensemble score: {total_score:.6f}")

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
df_out.to_csv("submission_comprehensive.csv", index=False)
print(f"Saved to submission_comprehensive.csv")

# Now validate with Shapely
print("\nValidating with Shapely...")
from shapely.geometry import Polygon
from shapely import affinity

TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]

def make_tree_polygon(x, y, deg):
    x = float(str(x).replace('s', ''))
    y = float(str(y).replace('s', ''))
    deg = float(str(deg).replace('s', ''))
    initial_polygon = Polygon(list(zip(TX, TY)))
    rotated = affinity.rotate(initial_polygon, deg, origin=(0, 0))
    return affinity.translate(rotated, xoff=x, yoff=y)

def has_overlap(trees):
    for i in range(len(trees)):
        for j in range(i+1, len(trees)):
            if trees[i].intersects(trees[j]) and not trees[i].touches(trees[j]):
                intersection = trees[i].intersection(trees[j])
                if intersection.area > 0:
                    return True, intersection.area
    return False, 0

# Load baseline for fallback
baseline = pd.read_csv("/home/nonroot/snapshots/santa-2025/21116303805/code/preoptimized/santa-2025.csv")

overlap_ns = []
for n in range(1, 201):
    data = best[n]["data"]
    trees = [make_tree_polygon(r['x'], r['y'], r['deg']) for _, r in data.iterrows()]
    has_ovl, area = has_overlap(trees)
    if has_ovl:
        overlap_ns.append((n, area, best[n]["src"]))
        # Fall back to baseline
        base_data = baseline[baseline['id'].str.startswith(f'{n:03d}_')]
        best[n]["data"] = base_data
        best[n]["src"] = "baseline_fallback"
        # Recalculate score
        xs = strip(base_data["x"].to_numpy())
        ys = strip(base_data["y"].to_numpy())
        ds = strip(base_data["deg"].to_numpy())
        best[n]["score"] = float(score_group(xs, ys, ds, tx, ty))

if overlap_ns:
    print(f"\nFound {len(overlap_ns)} N values with overlaps (falling back to baseline):")
    for n, area, src in overlap_ns[:10]:
        print(f"  N={n}: area={area:.2e}, src={src}")

# Recalculate total
total_score = sum(best[n]["score"] for n in range(1, 201))
print(f"\nFinal validated score: {total_score:.6f}")

# Save final output
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
df_out.to_csv("submission_validated.csv", index=False)
print(f"Saved validated submission to submission_validated.csv")

# Save metrics
import json
metrics = {
    "cv_score": total_score,
    "baseline_score": 70.676102,
    "improvement": 70.676102 - total_score,
    "overlap_fallbacks": len(overlap_ns)
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
