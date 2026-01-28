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

# Load current best
df = pd.read_csv('/home/submission/submission.csv')
df['N'] = df['id'].str.split('_').str[0].astype(int)

# Calculate per-N scores
scores = {}
for n in range(1, 201):
    g = df[df['N'] == n]
    xs, ys, angles = df_to_arrays(g)
    scores[n] = compute_bbox_score(xs, ys, angles, TX, TY)

total = sum(scores.values())
print(f"Current total score: {total:.6f}")
print(f"Target score: 68.866853")
print(f"Gap: {total - 68.866853:.6f} ({(total - 68.866853)/68.866853*100:.2f}%)")

# Analyze score distribution
print("\n" + "="*60)
print("Score by N range:")
for start in [1, 11, 51, 101, 151]:
    end = start + 49 if start > 1 else 10
    subset = {n: s for n, s in scores.items() if start <= n <= end}
    print(f"  N={start}-{end}: {sum(subset.values()):.4f} ({sum(subset.values())/total*100:.1f}%)")

# Calculate theoretical minimum (perfect packing)
tree_area = 0.24625
print("\n" + "="*60)
print("Efficiency analysis (actual / theoretical minimum):")
for n in [1, 2, 5, 10, 20, 50, 100, 200]:
    min_score = tree_area  # S^2/N = tree_area when perfectly packed
    actual = scores[n]
    print(f"  N={n}: {actual:.6f} / {min_score:.6f} = {actual/min_score:.2f}x")

# Find N values with worst efficiency
print("\n" + "="*60)
print("Top 20 N values with worst efficiency (most room for improvement):")
efficiency = [(n, scores[n]/tree_area) for n in range(1, 201)]
efficiency.sort(key=lambda x: -x[1])
for n, eff in efficiency[:20]:
    print(f"  N={n}: {scores[n]:.6f} ({eff:.2f}x theoretical)")
