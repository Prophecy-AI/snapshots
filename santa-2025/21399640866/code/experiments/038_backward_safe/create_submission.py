"""
Create submission starting from SAFE baseline (exp_033_fix_overlaps which passed Kaggle)
and applying backward iteration improvements for N=121 and N=122.
"""
import pandas as pd
import numpy as np
from numba import njit
import math
from shapely import Polygon
from shapely.affinity import rotate, translate
from decimal import Decimal, getcontext
getcontext().prec = 30

TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])
SCALE = 10**18

def get_tree_polygon(x, y, angle):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = rotate(poly, angle, origin=(0, 0), use_radians=False)
    poly = translate(poly, x, y)
    return poly

def check_overlaps_strict(xs, ys, angles):
    """Check for overlaps with high precision."""
    n = len(xs)
    if n <= 1:
        return []
    
    # Create polygons with integer coordinates for precision
    polygons = []
    for i in range(n):
        poly = get_tree_polygon(xs[i], ys[i], angles[i])
        coords = [(int(Decimal(str(x)) * SCALE), int(Decimal(str(y)) * SCALE)) 
                  for x, y in poly.exterior.coords]
        polygons.append(Polygon(coords))
    
    overlaps = []
    for i in range(n):
        for j in range(i+1, n):
            if polygons[i].intersects(polygons[j]):
                if not polygons[i].touches(polygons[j]):
                    inter = polygons[i].intersection(polygons[j])
                    if inter.area > 0:
                        overlaps.append((i, j, inter.area / (SCALE * SCALE)))
    return overlaps

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

# Load the SAFE submission that passed Kaggle (exp_033_fix_overlaps)
print("Loading safe baseline (exp_033_fix_overlaps)...")
safe_df = pd.read_csv('/home/code/experiments/033_fix_overlaps/submission.csv')
safe_df['N'] = safe_df['id'].str.split('_').str[0].astype(int)

# Load all configurations from safe submission
configs = {}
for n in range(1, 201):
    g = safe_df[safe_df['N'] == n]
    xs = np.array([strip(v) for v in g['x']])
    ys = np.array([strip(v) for v in g['y']])
    angles = np.array([strip(v) for v in g['deg']])
    configs[n] = (xs, ys, angles)

# Compute baseline scores
baseline_scores = {}
for n in range(1, 201):
    xs, ys, angles = configs[n]
    baseline_scores[n] = compute_bbox_score(xs, ys, angles, TX, TY)

print(f"Safe baseline total score: {sum(baseline_scores.values()):.6f}")

# Apply backward iteration improvements for N=122 and N=121
# N=122: Remove tree from N=123 configuration
print("\nApplying backward iteration improvements...")
xs_123, ys_123, angles_123 = configs[123]
best_score_122 = float('inf')
best_config_122 = None
for i in range(123):
    xs_new = np.delete(xs_123, i)
    ys_new = np.delete(ys_123, i)
    angles_new = np.delete(angles_123, i)
    overlaps = check_overlaps_strict(list(xs_new), list(ys_new), list(angles_new))
    if not overlaps:
        score = compute_bbox_score(xs_new, ys_new, angles_new, TX, TY)
        if score < best_score_122:
            best_score_122 = score
            best_config_122 = (xs_new.copy(), ys_new.copy(), angles_new.copy())

if best_config_122 and best_score_122 < baseline_scores[122]:
    print(f"N=122: {baseline_scores[122]:.6f} -> {best_score_122:.6f} (improvement: {baseline_scores[122] - best_score_122:.6f})")
    configs[122] = best_config_122
    baseline_scores[122] = best_score_122
else:
    print(f"N=122: No improvement found")

# N=121: Remove tree from updated N=122 configuration
xs_122, ys_122, angles_122 = configs[122]
best_score_121 = float('inf')
best_config_121 = None
for i in range(122):
    xs_new = np.delete(xs_122, i)
    ys_new = np.delete(ys_122, i)
    angles_new = np.delete(angles_122, i)
    overlaps = check_overlaps_strict(list(xs_new), list(ys_new), list(angles_new))
    if not overlaps:
        score = compute_bbox_score(xs_new, ys_new, angles_new, TX, TY)
        if score < best_score_121:
            best_score_121 = score
            best_config_121 = (xs_new.copy(), ys_new.copy(), angles_new.copy())

if best_config_121 and best_score_121 < baseline_scores[121]:
    print(f"N=121: {baseline_scores[121]:.6f} -> {best_score_121:.6f} (improvement: {baseline_scores[121] - best_score_121:.6f})")
    configs[121] = best_config_121
    baseline_scores[121] = best_score_121
else:
    print(f"N=121: No improvement found")

new_total = sum(baseline_scores.values())
print(f"\nNew total score: {new_total:.6f}")

# Validate ALL N values for overlaps
print("\nValidating all N values for overlaps...")
all_valid = True
for n in range(1, 201):
    xs, ys, angles = configs[n]
    overlaps = check_overlaps_strict(list(xs), list(ys), list(angles))
    if overlaps:
        print(f"N={n}: OVERLAPS DETECTED: {overlaps[:3]}")
        all_valid = False

if all_valid:
    print("All N values validated: NO OVERLAPS")
else:
    print("WARNING: Some N values have overlaps!")

# Create submission
print("\nCreating submission...")
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
result_df.to_csv('/home/submission/submission.csv', index=False)

print(f"Submission saved with score: {new_total:.6f}")
