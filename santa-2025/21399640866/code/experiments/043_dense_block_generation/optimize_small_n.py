"""
Optimize small N values (1-10) with exhaustive search.
For N=1, we just need to find the optimal rotation angle.
For N=2-10, we try many configurations.
"""
import numpy as np
import pandas as pd
from numba import njit
import math
import time
from itertools import product

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

@njit
def check_overlap(xs, ys, angles, tx, ty):
    """Check if any trees overlap (simplified - just check bounding boxes)."""
    n = len(xs)
    V = len(tx)
    
    # Get bounding boxes for each tree
    bboxes = np.zeros((n, 4))  # minx, miny, maxx, maxy
    for i in range(n):
        r = angles[i] * math.pi / 180.0
        c = math.cos(r)
        s = math.sin(r)
        xi = xs[i]
        yi = ys[i]
        mnx = 1e300
        mny = 1e300
        mxx = -1e300
        mxy = -1e300
        for j in range(V):
            X = c * tx[j] - s * ty[j] + xi
            Y = s * tx[j] + c * ty[j] + yi
            if X < mnx: mnx = X
            if X > mxx: mxx = X
            if Y < mny: mny = Y
            if Y > mxy: mxy = Y
        bboxes[i, 0] = mnx
        bboxes[i, 1] = mny
        bboxes[i, 2] = mxx
        bboxes[i, 3] = mxy
    
    # Check for overlapping bounding boxes
    for i in range(n):
        for j in range(i+1, n):
            # Check if bboxes overlap
            if (bboxes[i, 0] < bboxes[j, 2] and bboxes[i, 2] > bboxes[j, 0] and
                bboxes[i, 1] < bboxes[j, 3] and bboxes[i, 3] > bboxes[j, 1]):
                return True
    return False

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

# Get baseline scores
baseline_scores = {}
baseline_configs = {}
for n in range(1, 201):
    g = baseline_df[baseline_df['N'] == n]
    xs, ys, angles = df_to_arrays(g)
    baseline_scores[n] = compute_bbox_score(xs, ys, angles, TX, TY)
    baseline_configs[n] = (xs.copy(), ys.copy(), angles.copy())

print("Baseline scores for N=1-10:")
for n in range(1, 11):
    print(f"  N={n}: {baseline_scores[n]:.6f}")

print("\n" + "="*60)
print("Optimizing N=1 (exhaustive angle search)...")

# N=1: Just find optimal angle
best_score_1 = float('inf')
best_angle_1 = 0
for angle_int in range(0, 36000):  # 0.01 degree increments
    angle = angle_int / 100.0
    xs = np.array([0.0])
    ys = np.array([0.0])
    angles = np.array([angle])
    score = compute_bbox_score(xs, ys, angles, TX, TY)
    if score < best_score_1:
        best_score_1 = score
        best_angle_1 = angle

print(f"N=1: Best angle = {best_angle_1:.2f}°, score = {best_score_1:.6f}")
print(f"     Baseline = {baseline_scores[1]:.6f}, improvement = {baseline_scores[1] - best_score_1:.6f}")

# Check current N=1 configuration
xs, ys, angles = baseline_configs[1]
print(f"     Current config: x={xs[0]:.4f}, y={ys[0]:.4f}, angle={angles[0]:.4f}")

# The tree is symmetric, so we only need to check 0-90 degrees
# Let's verify the optimal angle
print("\nVerifying optimal angle range:")
for angle in [0, 15, 30, 45, 60, 75, 90]:
    xs = np.array([0.0])
    ys = np.array([0.0])
    angles = np.array([float(angle)])
    score = compute_bbox_score(xs, ys, angles, TX, TY)
    print(f"  angle={angle}°: score={score:.6f}")

# The tree at 45° should give the minimum bounding box for a single tree
# because the tree is roughly symmetric around the vertical axis
