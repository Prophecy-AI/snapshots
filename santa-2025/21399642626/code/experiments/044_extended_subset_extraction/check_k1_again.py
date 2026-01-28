"""
Check k=1 extraction again using exp_043 baseline.
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
    polygons = [get_tree_polygon(x, y, a) for x, y, a in trees]
    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > threshold:
                    return True
    return False

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

# Load baseline (exp_043)
print("Loading baseline (exp_043)...")
baseline_path = "/home/code/experiments/043_subset_extraction/ensemble_043.csv"
baseline = load_baseline(baseline_path)
baseline_scores = {n: compute_score(baseline[n], n) for n in range(1, 201)}
total_baseline = sum(baseline_scores.values())
print(f"Baseline total: {total_baseline:.6f}")

# k=1 extraction
print("\n" + "="*60)
print("k=1 EXTRACTION (remove 1 tree from N+1)")
print("="*60)

all_improvements = {}
MIN_IMPROVEMENT = 0.0001

for source_n in range(2, 201):
    source_trees = baseline[source_n]
    target_n = source_n - 1
    
    best_improvement = 0
    best_remove_idx = None
    best_subset = None
    
    for remove_idx in range(source_n):
        subset = [t for i, t in enumerate(source_trees) if i != remove_idx]
        subset_score = compute_score(subset, target_n)
        
        if subset_score < baseline_scores[target_n] - MIN_IMPROVEMENT:
            if check_overlap(subset):
                continue
            
            improvement = baseline_scores[target_n] - subset_score
            if improvement > best_improvement:
                best_improvement = improvement
                best_remove_idx = remove_idx
                best_subset = subset
    
    if best_subset is not None:
        all_improvements[target_n] = (best_subset, best_improvement, source_n, best_remove_idx)
        print(f"✅ N={target_n}: improvement={best_improvement:.6f} (from N={source_n}, remove tree {best_remove_idx})")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

if all_improvements:
    total_improvement = sum(imp[1] for imp in all_improvements.values())
    print(f"Total improvements found: {len(all_improvements)}")
    print(f"Total score improvement: {total_improvement:.6f}")
else:
    print("No additional improvements found from k=1 extraction")
