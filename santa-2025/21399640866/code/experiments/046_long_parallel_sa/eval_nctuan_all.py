"""
Evaluate ALL CSV files in the nctuan dataset and find best per-N solutions.
"""
import numpy as np
import pandas as pd
from numba import njit
import math
import glob
import os

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

# Load baseline
baseline_df = pd.read_csv('/home/submission/submission.csv')
baseline_df['N'] = baseline_df['id'].str.split('_').str[0].astype(int)

baseline_scores = {}
for n in range(1, 201):
    g = baseline_df[baseline_df['N'] == n]
    xs, ys, angles = df_to_arrays(g)
    baseline_scores[n] = compute_bbox_score(xs, ys, angles, TX, TY)

baseline_total = sum(baseline_scores.values())
print(f"Baseline total: {baseline_total:.6f}")

# Find all CSV files in nctuan dataset
csv_files = glob.glob('/home/code/research/datasets/nctuan/**/*.csv', recursive=True)
print(f"\nFound {len(csv_files)} CSV files")

# Track best per-N scores
best_per_n = {n: baseline_scores[n] for n in range(1, 201)}
best_per_n_source = {n: 'baseline' for n in range(1, 201)}

# Evaluate each CSV file
for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        if 'id' not in df.columns or 'x' not in df.columns:
            continue
        
        df['N'] = df['id'].str.split('_').str[0].astype(int)
        
        for n in range(1, 201):
            g = df[df['N'] == n]
            if len(g) != n:
                continue
            
            xs, ys, angles = df_to_arrays(g)
            score = compute_bbox_score(xs, ys, angles, TX, TY)
            
            if score < best_per_n[n] - 0.0001:
                best_per_n[n] = score
                best_per_n_source[n] = os.path.basename(csv_file)
    except Exception as e:
        continue

# Find improvements
improvements = []
for n in range(1, 201):
    diff = baseline_scores[n] - best_per_n[n]
    if diff > 0.0001:
        improvements.append((n, diff, best_per_n[n], baseline_scores[n], best_per_n_source[n]))

print(f"\nImprovements found: {len(improvements)}")
if improvements:
    improvements.sort(key=lambda x: -x[1])
    for n, diff, new_score, old_score, source in improvements[:30]:
        print(f"  N={n}: {old_score:.6f} -> {new_score:.6f} (improvement: {diff:.6f}) from {source}")
    total_improvement = sum(diff for _, diff, _, _, _ in improvements)
    print(f"\nTotal improvement potential: {total_improvement:.6f}")
    
    new_total = sum(best_per_n.values())
    print(f"New total score: {new_total:.6f}")
else:
    print("No improvements found in nctuan dataset")
