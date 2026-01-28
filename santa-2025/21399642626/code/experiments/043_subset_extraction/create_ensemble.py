"""
Create ensemble with improvements from subset extraction.
"""

import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import warnings
warnings.filterwarnings('ignore')

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def parse_coord(val):
    if isinstance(val, str):
        if val.startswith('s'):
            return float(val[1:])
        return float(val)
    return float(val)

def get_tree_vertices(x, y, angle_deg):
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rx = TX * cos_a - TY * sin_a
    ry = TX * sin_a + TY * cos_a
    return rx + x, ry + y

def get_tree_polygon(x, y, angle_deg):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = affinity.rotate(poly, angle_deg, origin=(0, 0))
    poly = affinity.translate(poly, x, y)
    return poly

def compute_bbox_size(trees):
    all_x, all_y = [], []
    for x, y, angle in trees:
        vx, vy = get_tree_vertices(x, y, angle)
        all_x.extend(vx)
        all_y.extend(vy)
    if not all_x:
        return float('inf')
    return max(max(all_x) - min(all_x), max(all_y) - min(all_y))

def compute_score(trees, n):
    if not trees or len(trees) != n:
        return float('inf')
    size = compute_bbox_size(trees)
    return (size ** 2) / n

def check_overlap(trees, threshold=1e-20):
    """Strict overlap check."""
    polygons = [get_tree_polygon(x, y, a) for x, y, a in trees]
    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > threshold:
                    return True, f"Trees {i} and {j} overlap (area={intersection.area:.2e})"
    return False, "OK"

def load_baseline(path):
    df = pd.read_csv(path)
    df['n'] = df['id'].apply(lambda x: int(x.split('_')[0]))
    df['x'] = df['x'].apply(parse_coord)
    df['y'] = df['y'].apply(parse_coord)
    df['deg'] = df['deg'].apply(parse_coord)
    
    result = {}
    for n in range(1, 201):
        n_df = df[df['n'] == n]
        if len(n_df) == n:
            result[n] = [(row['x'], row['y'], row['deg']) for _, row in n_df.iterrows()]
    return result

# Load baseline
print("Loading baseline (exp_039)...")
baseline_path = "/home/code/experiments/039_per_n_analysis/safe_ensemble.csv"
baseline = load_baseline(baseline_path)
baseline_scores = {n: compute_score(baseline[n], n) for n in range(1, 201)}
total_baseline = sum(baseline_scores.values())
print(f"Baseline total: {total_baseline:.6f}")

# Create ensemble with improvements
ensemble = {n: baseline[n] for n in range(1, 201)}
improvements_applied = []

# N=121: Extract from N=122 by removing tree 0
source_n = 122
target_n = 121
remove_idx = 0
subset = [t for i, t in enumerate(baseline[source_n]) if i != remove_idx]
subset_score = compute_score(subset, target_n)
has_overlap, msg = check_overlap(subset)

print(f"\nN={target_n}: Checking improvement from N={source_n}")
print(f"  Baseline score: {baseline_scores[target_n]:.6f}")
print(f"  Subset score: {subset_score:.6f}")
print(f"  Improvement: {baseline_scores[target_n] - subset_score:.6f}")
print(f"  Overlap check: {msg}")

if not has_overlap and subset_score < baseline_scores[target_n] - 0.0001:
    ensemble[target_n] = subset
    improvements_applied.append((target_n, baseline_scores[target_n] - subset_score, source_n))
    print(f"  ✅ Applied!")
else:
    print(f"  ❌ Not applied (overlap or insufficient improvement)")

# N=122: Extract from N=123 by removing tree 0
source_n = 123
target_n = 122
remove_idx = 0
subset = [t for i, t in enumerate(baseline[source_n]) if i != remove_idx]
subset_score = compute_score(subset, target_n)
has_overlap, msg = check_overlap(subset)

print(f"\nN={target_n}: Checking improvement from N={source_n}")
print(f"  Baseline score: {baseline_scores[target_n]:.6f}")
print(f"  Subset score: {subset_score:.6f}")
print(f"  Improvement: {baseline_scores[target_n] - subset_score:.6f}")
print(f"  Overlap check: {msg}")

if not has_overlap and subset_score < baseline_scores[target_n] - 0.0001:
    ensemble[target_n] = subset
    improvements_applied.append((target_n, baseline_scores[target_n] - subset_score, source_n))
    print(f"  ✅ Applied!")
else:
    print(f"  ❌ Not applied (overlap or insufficient improvement)")

# Calculate new total
new_total = sum(compute_score(ensemble[n], n) for n in range(1, 201))
print(f"\n" + "="*60)
print(f"SUMMARY")
print(f"="*60)
print(f"Baseline total: {total_baseline:.6f}")
print(f"New total: {new_total:.6f}")
print(f"Total improvement: {total_baseline - new_total:.6f}")
print(f"Improvements applied: {len(improvements_applied)}")

for n, imp, source in improvements_applied:
    print(f"  N={n}: {imp:.6f} (from N={source})")

# Validate all N values
print("\nValidating all N values...")
all_valid = True
for n in range(1, 201):
    has_overlap, msg = check_overlap(ensemble[n])
    if has_overlap:
        print(f"  N={n}: OVERLAP - {msg}")
        all_valid = False

if all_valid:
    print("All N values validated - no overlaps!")
else:
    print("WARNING: Some N values have overlaps!")

# Save ensemble
if all_valid and improvements_applied:
    rows = []
    for n in range(1, 201):
        for i, (x, y, deg) in enumerate(ensemble[n]):
            rows.append({
                'id': f'{n:03d}_{i}',
                'x': f's{x:.20e}',
                'y': f's{y:.20e}',
                'deg': f's{deg:.20e}'
            })
    
    df_out = pd.DataFrame(rows)
    df_out.to_csv('ensemble_043.csv', index=False)
    print(f"\nSaved ensemble to ensemble_043.csv")
    
    # Copy to submission folder
    import shutil
    shutil.copy('ensemble_043.csv', '/home/submission/submission.csv')
    print("Copied to /home/submission/submission.csv")
else:
    print("\nNo valid improvements - using baseline")
    import shutil
    shutil.copy(baseline_path, '/home/submission/submission.csv')
    print("Copied baseline to /home/submission/submission.csv")
