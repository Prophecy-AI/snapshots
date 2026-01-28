"""
Systematic extraction: For each N, try extracting from N+1, N+2, N+3, etc.
This extends the successful subset extraction approach.
"""

import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
from itertools import combinations
import time
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

def extract_best_subset(source_trees, target_n, baseline_score, min_improvement=0.0001):
    """
    Find the best subset of target_n trees from source_trees.
    """
    source_n = len(source_trees)
    k = source_n - target_n
    
    if k <= 0 or k > 5:  # Limit k to avoid combinatorial explosion
        return None, float('inf')
    
    best_score = baseline_score
    best_subset = None
    
    # Try all combinations of removing k trees
    for remove_indices in combinations(range(source_n), k):
        subset = [t for i, t in enumerate(source_trees) if i not in remove_indices]
        
        score = compute_score(subset, target_n)
        if score >= best_score - min_improvement:
            continue
        
        if check_overlap(subset):
            continue
        
        best_score = score
        best_subset = subset
    
    return best_subset, best_score

# Load baseline (exp_044)
print("Loading baseline (exp_044)...")
baseline_path = "/home/code/experiments/044_extended_subset_extraction/ensemble_044.csv"
baseline = load_baseline(baseline_path)
baseline_scores = {n: compute_score(baseline[n], n) for n in range(1, 201)}
total_baseline = sum(baseline_scores.values())
print(f"Baseline total: {total_baseline:.6f}")

# Systematic extraction: for each N, try extracting from N+1, N+2, N+3, N+4, N+5
print("\n" + "="*60)
print("SYSTEMATIC EXTRACTION FROM N+k (k=1,2,3,4,5)")
print("="*60)

all_improvements = {}
MIN_IMPROVEMENT = 0.0001

for target_n in range(1, 196):  # Up to 195 so we can try N+5
    best_improvement = 0
    best_subset = None
    best_source = None
    
    for k in range(1, 6):  # k=1,2,3,4,5
        source_n = target_n + k
        if source_n > 200:
            continue
        
        source_trees = baseline[source_n]
        subset, score = extract_best_subset(source_trees, target_n, baseline_scores[target_n], MIN_IMPROVEMENT)
        
        if subset is not None:
            improvement = baseline_scores[target_n] - score
            if improvement > best_improvement:
                best_improvement = improvement
                best_subset = subset
                best_source = (source_n, k)
    
    if best_subset is not None and best_improvement > MIN_IMPROVEMENT:
        all_improvements[target_n] = (best_subset, best_improvement, best_source)
        print(f"✅ N={target_n}: improvement={best_improvement:.6f} (from N={best_source[0]}, k={best_source[1]})")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

if all_improvements:
    total_improvement = sum(imp for _, imp, _ in all_improvements.values())
    print(f"Total improvements found: {len(all_improvements)}")
    print(f"Total score improvement: {total_improvement:.6f}")
    
    print("\nAll improvements:")
    for n, (_, imp, source) in sorted(all_improvements.items(), key=lambda x: -x[1][1]):
        print(f"  N={n}: improvement={imp:.6f} (from N={source[0]}, k={source[1]})")
    
    # Create ensemble
    ensemble = {n: baseline[n] for n in range(1, 201)}
    for n, (subset, _, _) in all_improvements.items():
        ensemble[n] = subset
    
    new_total = sum(compute_score(ensemble[n], n) for n in range(1, 201))
    print(f"\nBaseline total: {total_baseline:.6f}")
    print(f"New total: {new_total:.6f}")
    print(f"Actual improvement: {total_baseline - new_total:.6f}")
else:
    print("No improvements found from systematic extraction")
