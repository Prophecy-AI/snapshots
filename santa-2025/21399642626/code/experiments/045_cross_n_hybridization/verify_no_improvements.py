"""
Verify that no more improvements exist from k=1 extraction.
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

# Load baseline (exp_044)
print("Loading baseline (exp_044)...")
baseline_path = "/home/code/experiments/044_extended_subset_extraction/ensemble_044.csv"
baseline = load_baseline(baseline_path)
baseline_scores = {n: compute_score(baseline[n], n) for n in range(1, 201)}
total_baseline = sum(baseline_scores.values())
print(f"Baseline total: {total_baseline:.6f}")

# Check k=1 extraction for ALL N values
print("\n" + "="*60)
print("VERIFYING NO MORE k=1 IMPROVEMENTS EXIST")
print("="*60)

MIN_IMPROVEMENT = 0.00001  # Very low threshold to catch any improvements

improvements_found = []

for source_n in range(2, 201):
    source_trees = baseline[source_n]
    target_n = source_n - 1
    
    for remove_idx in range(source_n):
        subset = [t for i, t in enumerate(source_trees) if i != remove_idx]
        subset_score = compute_score(subset, target_n)
        
        if subset_score < baseline_scores[target_n] - MIN_IMPROVEMENT:
            if not check_overlap(subset):
                improvement = baseline_scores[target_n] - subset_score
                improvements_found.append((target_n, improvement, source_n, remove_idx))

if improvements_found:
    print(f"Found {len(improvements_found)} potential improvements:")
    for n, imp, source_n, remove_idx in sorted(improvements_found, key=lambda x: -x[1]):
        print(f"  N={n}: improvement={imp:.8f} (from N={source_n}, remove tree {remove_idx})")
else:
    print("CONFIRMED: No more k=1 improvements exist")
    print("The baseline (exp_044) is at a local optimum for subset extraction")
