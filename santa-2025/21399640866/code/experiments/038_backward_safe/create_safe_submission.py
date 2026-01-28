"""
Create submission using EXACT safe baseline coordinates, only modifying N=121 and N=122.
"""
import pandas as pd
import numpy as np
from numba import njit
import math
from shapely import Polygon
from shapely.affinity import rotate, translate

TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def get_tree_polygon(x, y, angle):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = rotate(poly, angle, origin=(0, 0), use_radians=False)
    poly = translate(poly, x, y)
    return poly

def check_overlaps(xs, ys, angles):
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

# Load the SAFE submission that passed Kaggle
print("Loading safe baseline...")
safe_df = pd.read_csv('/home/code/experiments/033_fix_overlaps/submission.csv')
safe_df['N'] = safe_df['id'].str.split('_').str[0].astype(int)

# Load all configurations from safe submission
configs = {}
raw_configs = {}  # Keep raw string values
for n in range(1, 201):
    g = safe_df[safe_df['N'] == n].sort_values('id')
    xs = np.array([strip(v) for v in g['x']])
    ys = np.array([strip(v) for v in g['y']])
    angles = np.array([strip(v) for v in g['deg']])
    configs[n] = (xs, ys, angles)
    raw_configs[n] = (g['x'].values.tolist(), g['y'].values.tolist(), g['deg'].values.tolist())

# Compute baseline scores
baseline_scores = {}
for n in range(1, 201):
    xs, ys, angles = configs[n]
    baseline_scores[n] = compute_bbox_score(xs, ys, angles, TX, TY)

print(f"Safe baseline total score: {sum(baseline_scores.values()):.6f}")

# Apply backward iteration improvements for N=122 and N=121
print("\nApplying backward iteration improvements...")

# N=122: Remove tree from N=123 configuration
xs_123, ys_123, angles_123 = configs[123]
best_score_122 = float('inf')
best_config_122 = None
best_idx_122 = None
for i in range(123):
    xs_new = np.delete(xs_123, i)
    ys_new = np.delete(ys_123, i)
    angles_new = np.delete(angles_123, i)
    if not check_overlaps(list(xs_new), list(ys_new), list(angles_new)):
        score = compute_bbox_score(xs_new, ys_new, angles_new, TX, TY)
        if score < best_score_122:
            best_score_122 = score
            best_config_122 = (xs_new.copy(), ys_new.copy(), angles_new.copy())
            best_idx_122 = i

if best_config_122 and best_score_122 < baseline_scores[122]:
    print(f"N=122: {baseline_scores[122]:.6f} -> {best_score_122:.6f} (removed tree {best_idx_122})")
    configs[122] = best_config_122
    # Update raw_configs for N=122 - remove tree at best_idx_122 from N=123
    raw_x_123, raw_y_123, raw_deg_123 = raw_configs[123]
    raw_configs[122] = (
        raw_x_123[:best_idx_122] + raw_x_123[best_idx_122+1:],
        raw_y_123[:best_idx_122] + raw_y_123[best_idx_122+1:],
        raw_deg_123[:best_idx_122] + raw_deg_123[best_idx_122+1:]
    )
    baseline_scores[122] = best_score_122
else:
    print(f"N=122: No improvement found")

# N=121: Remove tree from updated N=122 configuration
xs_122, ys_122, angles_122 = configs[122]
best_score_121 = float('inf')
best_config_121 = None
best_idx_121 = None
for i in range(122):
    xs_new = np.delete(xs_122, i)
    ys_new = np.delete(ys_122, i)
    angles_new = np.delete(angles_122, i)
    if not check_overlaps(list(xs_new), list(ys_new), list(angles_new)):
        score = compute_bbox_score(xs_new, ys_new, angles_new, TX, TY)
        if score < best_score_121:
            best_score_121 = score
            best_config_121 = (xs_new.copy(), ys_new.copy(), angles_new.copy())
            best_idx_121 = i

if best_config_121 and best_score_121 < baseline_scores[121]:
    print(f"N=121: {baseline_scores[121]:.6f} -> {best_score_121:.6f} (removed tree {best_idx_121})")
    configs[121] = best_config_121
    # Update raw_configs for N=121 - remove tree at best_idx_121 from N=122
    raw_x_122, raw_y_122, raw_deg_122 = raw_configs[122]
    raw_configs[121] = (
        raw_x_122[:best_idx_121] + raw_x_122[best_idx_121+1:],
        raw_y_122[:best_idx_121] + raw_y_122[best_idx_121+1:],
        raw_deg_122[:best_idx_121] + raw_deg_122[best_idx_121+1:]
    )
    baseline_scores[121] = best_score_121
else:
    print(f"N=121: No improvement found")

new_total = sum(baseline_scores.values())
print(f"\nNew total score: {new_total:.6f}")

# Create submission using raw string values (preserving precision)
print("\nCreating submission...")
rows = []
for n in range(1, 201):
    raw_x, raw_y, raw_deg = raw_configs[n]
    for i in range(n):
        rows.append({
            'id': f'{n:03d}_{i}',
            'x': raw_x[i],
            'y': raw_y[i],
            'deg': raw_deg[i]
        })

result_df = pd.DataFrame(rows)
result_df.to_csv('submission.csv', index=False)
result_df.to_csv('/home/submission/submission.csv', index=False)

print(f"Submission saved with score: {new_total:.6f}")

# Verify the submission
print("\nVerifying submission...")
verify_df = pd.read_csv('/home/submission/submission.csv')
verify_df['N'] = verify_df['id'].str.split('_').str[0].astype(int)

total_verify = 0
for n in range(1, 201):
    g = verify_df[verify_df['N'] == n]
    xs = np.array([strip(v) for v in g['x']])
    ys = np.array([strip(v) for v in g['y']])
    angles = np.array([strip(v) for v in g['deg']])
    sc = compute_bbox_score(xs, ys, angles, TX, TY)
    total_verify += sc

print(f"Verified total score: {total_verify:.6f}")
