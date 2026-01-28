"""
Cross-N Hybridization: Use trees from MULTIPLE N values to construct new solutions.
The idea: trees from nearby N solutions might combine to form better solutions.
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

def check_overlap(trees, threshold=1e-15):
    """Fast overlap check."""
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

def cross_n_hybridization(target_n, solutions, num_candidates=500, window=10):
    """
    Create new N-tree solution by selecting trees from nearby N values.
    
    Args:
        target_n: Target number of trees
        solutions: Dict of n -> list of (x, y, angle) trees
        num_candidates: Number of random candidates to try
        window: Range of N values to consider (N-window to N+window)
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
    
    for _ in range(num_candidates):
        # Random selection of target_n trees
        selected = random.sample(candidate_trees, target_n)
        
        # Quick score check
        score = compute_score(selected, target_n)
        if score >= best_score:
            continue
        
        # Check overlaps only for promising candidates
        if check_overlap(selected):
            continue
        
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

# Test on small N values first
print("\n" + "="*60)
print("TESTING CROSS-N HYBRIDIZATION ON SMALL N VALUES")
print("="*60)

test_n_values = [10, 15, 20, 25, 30]
MIN_IMPROVEMENT = 0.0001

for target_n in test_n_values:
    print(f"\nN={target_n}: Testing cross-N hybridization...")
    start_time = time.time()
    
    hybrid_solution, hybrid_score = cross_n_hybridization(target_n, baseline, num_candidates=1000, window=10)
    
    elapsed = time.time() - start_time
    
    if hybrid_solution is None:
        print(f"  No valid solution found [{elapsed:.1f}s]")
        continue
    
    improvement = baseline_scores[target_n] - hybrid_score
    
    if improvement > MIN_IMPROVEMENT:
        print(f"  ✅ IMPROVED: {baseline_scores[target_n]:.6f} -> {hybrid_score:.6f} (improvement: {improvement:.6f}) [{elapsed:.1f}s]")
    else:
        print(f"  ❌ No improvement: baseline {baseline_scores[target_n]:.6f}, hybrid {hybrid_score:.6f} [{elapsed:.1f}s]")

# Now try on all N values
print("\n" + "="*60)
print("CROSS-N HYBRIDIZATION ON ALL N VALUES")
print("="*60)

all_improvements = {}

for target_n in range(2, 201):  # Skip N=1 (only 1 tree)
    hybrid_solution, hybrid_score = cross_n_hybridization(target_n, baseline, num_candidates=500, window=10)
    
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
    
    print("\nTop 10 improvements:")
    for n, (_, imp) in sorted(all_improvements.items(), key=lambda x: -x[1][1])[:10]:
        print(f"  N={n}: improvement={imp:.6f}")
else:
    print("No improvements found from cross-N hybridization")
    print("The baseline solutions are already optimal for their respective N values")
