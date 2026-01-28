"""
Create submission with backward iteration improvements.
"""
import pandas as pd
import numpy as np
from numba import njit
import math
from shapely import Polygon
from shapely.affinity import rotate, translate
import json

# Tree geometry
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def get_tree_polygon(x, y, angle):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = rotate(poly, angle, origin=(0, 0), use_radians=False)
    poly = translate(poly, x, y)
    return poly

def check_overlaps(xs, ys, angles):
    """Check if any trees overlap."""
    n = len(xs)
    if n <= 1:
        return False
    polygons = [get_tree_polygon(xs[i], ys[i], angles[i]) for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if polygons[i].intersects(polygons[j]):
                if not polygons[i].touches(polygons[j]):
                    area = polygons[i].intersection(polygons[j]).area
                    if area > 1e-12:
                        return True
    return False

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

# Load current best submission
baseline_df = pd.read_csv('/home/submission/submission.csv')
baseline_df['N'] = baseline_df['id'].str.split('_').str[0].astype(int)

# Load all configurations
configs = {}
for n in range(1, 201):
    g = baseline_df[baseline_df['N'] == n]
    xs = np.array([strip(v) for v in g['x']])
    ys = np.array([strip(v) for v in g['y']])
    angles = np.array([strip(v) for v in g['deg']])
    configs[n] = (xs, ys, angles)

# Apply backward iteration improvements for N=122 and N=121
# N=122: Remove tree from N=123 configuration
xs_123, ys_123, angles_123 = configs[123]
best_score_122 = float('inf')
best_config_122 = None
for i in range(123):
    xs_new = np.delete(xs_123, i)
    ys_new = np.delete(ys_123, i)
    angles_new = np.delete(angles_123, i)
    if not check_overlaps(list(xs_new), list(ys_new), list(angles_new)):
        score = compute_bbox_score(xs_new, ys_new, angles_new, TX, TY)
        if score < best_score_122:
            best_score_122 = score
            best_config_122 = (xs_new.copy(), ys_new.copy(), angles_new.copy())

if best_config_122:
    configs[122] = best_config_122
    print(f"N=122: Updated to score {best_score_122:.6f}")

# N=121: Remove tree from updated N=122 configuration
xs_122, ys_122, angles_122 = configs[122]
best_score_121 = float('inf')
best_config_121 = None
for i in range(122):
    xs_new = np.delete(xs_122, i)
    ys_new = np.delete(ys_122, i)
    angles_new = np.delete(angles_122, i)
    if not check_overlaps(list(xs_new), list(ys_new), list(angles_new)):
        score = compute_bbox_score(xs_new, ys_new, angles_new, TX, TY)
        if score < best_score_121:
            best_score_121 = score
            best_config_121 = (xs_new.copy(), ys_new.copy(), angles_new.copy())

if best_config_121:
    configs[121] = best_config_121
    print(f"N=121: Updated to score {best_score_121:.6f}")

# Create submission
rows = []
for n in range(1, 201):
    xs, ys, angles = configs[n]
    for i in range(n):
        rows.append({
            'id': f'{n:03d}_{i}',
            'x': f's{xs[i]:.20f}',
            'y': f's{ys[i]:.20f}',
            'deg': f's{angles[i]:.20f}'
        })

result_df = pd.DataFrame(rows)
result_df.to_csv('submission.csv', index=False)

# Verify total score
total = 0
for n in range(1, 201):
    xs, ys, angles = configs[n]
    score = compute_bbox_score(xs, ys, angles, TX, TY)
    total += score

print(f"\nTotal score: {total:.6f}")
print(f"Original: 70.309159")
print(f"Improvement: {70.309159 - total:.6f}")

# Copy to submission folder
result_df.to_csv('/home/submission/submission.csv', index=False)
print("\nSubmission saved to /home/submission/submission.csv")
