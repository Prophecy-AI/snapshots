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

# Calculate per-N scores and side lengths
results = []
for n in range(1, 201):
    g = baseline_df[baseline_df['N'] == n]
    xs, ys, angles = df_to_arrays(g)
    score = compute_bbox_score(xs, ys, angles, TX, TY)
    side = math.sqrt(score * n)  # S^2/N = score, so S = sqrt(score * N)
    results.append({
        'N': n,
        'score': score,
        'side': side,
        'contribution': score  # Each N contributes score to total
    })

df = pd.DataFrame(results)
df = df.sort_values('score', ascending=False)

print("Top 30 worst-performing N values (highest S²/N):")
print(df.head(30).to_string(index=False))

print("\n" + "="*60)
print("Score by N range:")
for start in [1, 51, 101, 151]:
    end = start + 49
    subset = df[(df['N'] >= start) & (df['N'] <= end)]
    print(f"  N={start}-{end}: {subset['score'].sum():.4f} ({subset['score'].sum()/df['score'].sum()*100:.1f}%)")

print("\n" + "="*60)
print("Total score:", df['score'].sum())

# Theoretical minimum for perfect packing
# Tree area is approximately 0.24625 (calculated from polygon)
tree_area = 0.24625
print(f"\nTree area: {tree_area}")
print("Theoretical minimum (if trees could pack perfectly):")
for n in [1, 10, 50, 100, 200]:
    min_side = math.sqrt(n * tree_area)
    min_score = min_side * min_side / n
    actual = df[df['N'] == n]['score'].values[0]
    print(f"  N={n}: min_score={min_score:.6f}, actual={actual:.6f}, ratio={actual/min_score:.2f}x")
