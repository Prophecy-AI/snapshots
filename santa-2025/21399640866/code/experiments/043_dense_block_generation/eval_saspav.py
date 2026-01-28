import numpy as np
import pandas as pd
from numba import njit
import math

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

# Load datasets
saspav_df = pd.read_csv('/home/code/research/datasets/saspav_latest/santa-2025.csv')
baseline_df = pd.read_csv('/home/submission/submission.csv')

saspav_df['N'] = saspav_df['id'].str.split('_').str[0].astype(int)
baseline_df['N'] = baseline_df['id'].str.split('_').str[0].astype(int)

# Calculate scores
saspav_scores = {}
baseline_scores = {}

for n in range(1, 201):
    g_s = saspav_df[saspav_df['N'] == n]
    g_b = baseline_df[baseline_df['N'] == n]
    
    xs_s, ys_s, angles_s = df_to_arrays(g_s)
    xs_b, ys_b, angles_b = df_to_arrays(g_b)
    
    saspav_scores[n] = compute_bbox_score(xs_s, ys_s, angles_s, TX, TY)
    baseline_scores[n] = compute_bbox_score(xs_b, ys_b, angles_b, TX, TY)

saspav_total = sum(saspav_scores.values())
baseline_total = sum(baseline_scores.values())

print(f"Saspav total: {saspav_total:.6f}")
print(f"Baseline total: {baseline_total:.6f}")
print(f"Difference: {saspav_total - baseline_total:+.6f}")

# Find improvements
improvements = []
for n in range(1, 201):
    diff = baseline_scores[n] - saspav_scores[n]
    if diff > 0.0001:
        improvements.append((n, diff, saspav_scores[n], baseline_scores[n]))

print(f"\nImprovements found: {len(improvements)}")
if improvements:
    improvements.sort(key=lambda x: -x[1])
    for n, diff, saspav, baseline in improvements[:20]:
        print(f"  N={n}: {baseline:.6f} -> {saspav:.6f} (improvement: {diff:.6f})")
    total_improvement = sum(diff for _, diff, _, _ in improvements)
    print(f"\nTotal improvement potential: {total_improvement:.6f}")
