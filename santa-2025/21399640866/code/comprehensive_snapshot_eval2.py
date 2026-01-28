import numpy as np
import pandas as pd
from numba import njit
import math
import os
import glob
from collections import defaultdict

TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

@njit
def compute_bbox_score(xs, ys, angles, tx, ty):
    n = len(xs)
    V = len(tx)
    mnx = 1e300
    mny = 1e300
    mxx = -1e300
    mxy = -1e300
    for i in range(n):
        r = angles[i] * math.pi / 180.0
        c = math.cos(r)
        s = math.sin(r)
        xi = xs[i]
        yi = ys[i]
        for j in range(V):
            X = c * tx[j] - s * ty[j] + xi
            Y = s * tx[j] + c * ty[j] + yi
            if X < mnx: mnx = X
            if X > mxx: mxx = X
            if Y < mny: mny = Y
            if Y > mxy: mxy = Y
    side = max(mxx - mnx, mxy - mny)
    return side * side / n

def strip(v):
    return float(str(v).replace("s", ""))

def df_to_arrays(df):
    xs = np.array([strip(v) for v in df['x']])
    ys = np.array([strip(v) for v in df['y']])
    angles = np.array([strip(v) for v in df['deg']])
    return xs, ys, angles

# Load current best
baseline_df = pd.read_csv('/home/submission/submission.csv')
baseline_df['N'] = baseline_df['id'].str.split('_').str[0].astype(int)

# Calculate baseline per-N scores
baseline_scores = {}
for n in range(1, 201):
    g = baseline_df[baseline_df['N'] == n]
    xs, ys, angles = df_to_arrays(g)
    baseline_scores[n] = compute_bbox_score(xs, ys, angles, TX, TY)

baseline_total = sum(baseline_scores.values())
print(f"Baseline total: {baseline_total:.6f}")

# Find ALL CSV files recursively
import subprocess
result = subprocess.run(['find', '/home/nonroot/snapshots', '-name', '*.csv'], 
                       capture_output=True, text=True)
all_csvs = [f.strip() for f in result.stdout.split('\n') if f.strip()]
print(f"Found {len(all_csvs)} CSV files in snapshots")

# Track best per-N across all sources
best_per_n = {n: (baseline_scores[n], 'baseline', None) for n in range(1, 201)}
improvements = []

# Evaluate each CSV
for i, csv_path in enumerate(all_csvs):
    if i % 500 == 0:
        print(f"Processing {i}/{len(all_csvs)}...")
    
    try:
        df = pd.read_csv(csv_path)
        if 'id' not in df.columns:
            continue
        df['N'] = df['id'].str.split('_').str[0].astype(int)
        
        for n in range(1, 201):
            g = df[df['N'] == n]
            if len(g) != n:
                continue
            
            xs, ys, angles = df_to_arrays(g)
            score = compute_bbox_score(xs, ys, angles, TX, TY)
            
            if score < best_per_n[n][0] - 0.0001:  # Significant improvement
                old_score = best_per_n[n][0]
                best_per_n[n] = (score, csv_path, g.copy())
                improvements.append((n, old_score - score, csv_path))
    except Exception as e:
        continue

# Summary
print("\n" + "="*60)
print("Improvements found over baseline:")
improvements.sort(key=lambda x: -x[1])
total_improvement = 0
for n, imp, source in improvements[:30]:
    print(f"  N={n}: improvement {imp:.6f} from {source.split('/')[-3]}")
    total_improvement += imp

print(f"\nTotal potential improvement: {total_improvement:.6f}")
print(f"New total would be: {baseline_total - total_improvement:.6f}")

# Count unique N values with improvements
unique_ns = set(n for n, _, _ in improvements)
print(f"Unique N values with improvements: {len(unique_ns)}")
