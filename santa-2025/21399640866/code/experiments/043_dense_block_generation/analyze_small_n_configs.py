"""
Analyze current configurations for N=2-5 and try to improve them.
"""
import numpy as np
import pandas as pd
from numba import njit
import math
from shapely.geometry import Polygon
from shapely import affinity

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

def get_tree_polygon(x, y, angle):
    """Create a Shapely polygon for a tree."""
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = affinity.rotate(poly, angle, origin=(0, 0))
    poly = affinity.translate(poly, x, y)
    return poly

def check_overlap_shapely(xs, ys, angles):
    """Check for overlaps using Shapely."""
    n = len(xs)
    polys = [get_tree_polygon(xs[i], ys[i], angles[i]) for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
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

# Analyze N=2
print("="*60)
print("Analyzing N=2")
g = baseline_df[baseline_df['N'] == 2]
xs, ys, angles = df_to_arrays(g)
print(f"Current config:")
for i in range(len(xs)):
    print(f"  Tree {i}: x={xs[i]:.6f}, y={ys[i]:.6f}, angle={angles[i]:.6f}")
print(f"Score: {compute_bbox_score(xs, ys, angles, TX, TY):.6f}")

# Try to find a better configuration for N=2
# Two trees can interlock if one is rotated 180° relative to the other
print("\nSearching for better N=2 configuration...")
best_score = compute_bbox_score(xs, ys, angles, TX, TY)
best_config = (xs.copy(), ys.copy(), angles.copy())

# Try different angle pairs
for angle1 in range(0, 360, 5):
    for angle2 in range(0, 360, 5):
        # Try different relative positions
        for dx in np.linspace(-0.5, 0.5, 21):
            for dy in np.linspace(-1.0, 1.0, 41):
                xs_test = np.array([0.0, dx])
                ys_test = np.array([0.0, dy])
                angles_test = np.array([float(angle1), float(angle2)])
                
                # Check for overlap
                if check_overlap_shapely(xs_test, ys_test, angles_test):
                    continue
                
                score = compute_bbox_score(xs_test, ys_test, angles_test, TX, TY)
                if score < best_score - 0.0001:
                    best_score = score
                    best_config = (xs_test.copy(), ys_test.copy(), angles_test.copy())
                    print(f"  Found better: score={score:.6f}, angles=({angle1}, {angle2}), pos=({dx:.2f}, {dy:.2f})")

print(f"\nBest N=2 score: {best_score:.6f}")
print(f"Improvement: {compute_bbox_score(xs, ys, angles, TX, TY) - best_score:.6f}")
