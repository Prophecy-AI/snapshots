import numpy as np
import pandas as pd
from numba import njit
import math
import os
import subprocess
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon

getcontext().prec = 25

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
print(f"Current submission total: {baseline_total:.6f}")
print(f"Target: 68.866853")
print(f"Gap: {baseline_total - 68.866853:.6f}")

# Check what the best public kernel scores are
print("\n" + "="*60)
print("Checking public kernel scores...")

# Load jazivxt team-optimization-blend
try:
    df = pd.read_csv('/home/code/research/kernels/jazivxt_team-optimization-blend/submission.csv')
    if 'id' in df.columns:
        df['N'] = df['id'].str.split('_').str[0].astype(int)
        total = 0
        for n in range(1, 201):
            g = df[df['N'] == n]
            if len(g) == n:
                xs, ys, angles = df_to_arrays(g)
                total += compute_bbox_score(xs, ys, angles, TX, TY)
        print(f"jazivxt team-optimization-blend: {total:.6f}")
except Exception as e:
    print(f"jazivxt: {e}")

# Check saspav
try:
    df = pd.read_csv('/home/code/research/datasets/saspav_latest/santa-2025.csv')
    if 'id' in df.columns:
        df['N'] = df['id'].str.split('_').str[0].astype(int)
        total = 0
        for n in range(1, 201):
            g = df[df['N'] == n]
            if len(g) == n:
                xs, ys, angles = df_to_arrays(g)
                total += compute_bbox_score(xs, ys, angles, TX, TY)
        print(f"saspav santa-2025.csv: {total:.6f}")
except Exception as e:
    print(f"saspav: {e}")

# Check what's in the submission directory
print("\n" + "="*60)
print("Files in /home/submission/:")
for f in os.listdir('/home/submission'):
    print(f"  {f}")
