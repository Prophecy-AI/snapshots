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

# Evaluate both kumar submissions
for fname in ['submission (19).csv', 'submission (20).csv']:
    print(f"\n{'='*60}")
    print(f"Evaluating: {fname}")
    
    df = pd.read_csv(f'/home/code/research/datasets/kumar_latest/{fname}')
    df['N'] = df['id'].str.split('_').str[0].astype(int)
    
    scores = {}
    for n in range(1, 201):
        g = df[df['N'] == n]
        xs, ys, angles = df_to_arrays(g)
        scores[n] = compute_bbox_score(xs, ys, angles, TX, TY)
    
    total = sum(scores.values())
    print(f"Total: {total:.6f}")
    print(f"Difference from baseline: {total - baseline_total:+.6f}")
    
    # Find improvements
    improvements = []
    for n in range(1, 201):
        diff = baseline_scores[n] - scores[n]
        if diff > 0.0001:
            improvements.append((n, diff, scores[n], baseline_scores[n]))
    
    print(f"Improvements found: {len(improvements)}")
    if improvements:
        improvements.sort(key=lambda x: -x[1])
        for n, diff, new_score, old_score in improvements[:10]:
            print(f"  N={n}: {old_score:.6f} -> {new_score:.6f} (improvement: {diff:.6f})")
        total_improvement = sum(diff for _, diff, _, _ in improvements)
        print(f"Total improvement potential: {total_improvement:.6f}")
