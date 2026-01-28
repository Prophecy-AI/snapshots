"""
Greedy Cross-N Hybridization: Add trees one at a time, checking for overlaps.
"""

import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import random
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

def check_overlap_with_existing(new_tree, existing_trees, threshold=1e-15):
    """Check if new_tree overlaps with any existing trees."""
    new_poly = get_tree_polygon(*new_tree)
    for tree in existing_trees:
        existing_poly = get_tree_polygon(*tree)
        if new_poly.intersects(existing_poly) and not new_poly.touches(existing_poly):
            intersection = new_poly.intersection(existing_poly)
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

def greedy_hybrid(target_n, solutions, window=5, num_attempts=100):
    """
    Greedy construction: add trees one at a time from nearby N solutions.
    """
    # Collect candidate trees from nearby N values
    candidate_trees = []
    for n in range(max(1, target_n - window), min(201, target_n + window + 1)):
        if n in solutions:
            for tree in solutions[n]:
                candidate_trees.append(tree)
    
    if len(candidate_trees) < target_n:
        return None, float('inf')
    
    best_solution = None
    best_score = float('inf')
    
    for attempt in range(num_attempts):
        # Shuffle candidates
        random.shuffle(candidate_trees)
        
        # Greedy construction
        selected = []
        for tree in candidate_trees:
            if len(selected) >= target_n:
                break
            
            # Check if this tree overlaps with already selected trees
            if not check_overlap_with_existing(tree, selected):
                selected.append(tree)
        
        if len(selected) == target_n:
            score = compute_score(selected, target_n)
            if score < best_score:
                best_score = score
                best_solution = selected
    
    return best_solution, best_score

# Load baseline (exp_044)
print("Loading baseline (exp_044)...")
baseline_path = "/home/code/experiments/044_extended_subset_extraction/ensemble_044.csv"
baseline = load_baseline(baseline_path)
baseline_scores = {n: compute_score(baseline[n], n) for n in range(1, 201)}
total_baseline = sum(baseline_scores.values())
print(f"Baseline total: {total_baseline:.6f}")

# Test greedy hybrid on small N values
print("\n" + "="*60)
print("TESTING GREEDY CROSS-N HYBRIDIZATION")
print("="*60)

test_n_values = [10, 15, 20, 25, 30]
MIN_IMPROVEMENT = 0.0001

for target_n in test_n_values:
    print(f"\nN={target_n}: Testing greedy hybrid...")
    start_time = time.time()
    
    hybrid_solution, hybrid_score = greedy_hybrid(target_n, baseline, window=5, num_attempts=200)
    
    elapsed = time.time() - start_time
    
    if hybrid_solution is None:
        print(f"  No valid solution found [{elapsed:.1f}s]")
        continue
    
    improvement = baseline_scores[target_n] - hybrid_score
    
    if improvement > MIN_IMPROVEMENT:
        print(f"  ✅ IMPROVED: {baseline_scores[target_n]:.6f} -> {hybrid_score:.6f} (improvement: {improvement:.6f}) [{elapsed:.1f}s]")
    else:
        print(f"  ❌ No improvement: baseline {baseline_scores[target_n]:.6f}, hybrid {hybrid_score:.6f} (diff: {improvement:.6f}) [{elapsed:.1f}s]")

# Try on all N values
print("\n" + "="*60)
print("GREEDY HYBRID ON ALL N VALUES")
print("="*60)

all_improvements = {}

for target_n in range(2, 201):
    hybrid_solution, hybrid_score = greedy_hybrid(target_n, baseline, window=5, num_attempts=100)
    
    if hybrid_solution is None:
        continue
    
    improvement = baseline_scores[target_n] - hybrid_score
    
    if improvement > MIN_IMPROVEMENT:
        all_improvements[target_n] = (hybrid_solution, improvement)
        print(f"✅ N={target_n}: {baseline_scores[target_n]:.6f} -> {hybrid_score:.6f} (improvement: {improvement:.6f})")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

if all_improvements:
    total_improvement = sum(imp for _, imp in all_improvements.values())
    print(f"Total improvements found: {len(all_improvements)}")
    print(f"Total score improvement: {total_improvement:.6f}")
else:
    print("No improvements found from greedy cross-N hybridization")
