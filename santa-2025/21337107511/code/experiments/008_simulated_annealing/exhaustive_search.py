"""
Exhaustive search around baseline positions.

For each tree, try ALL positions in a fine grid around its current position.
This will definitively tell us if any improvement is possible.
"""

import numpy as np
import pandas as pd
import math
from shapely import Polygon
from shapely.affinity import rotate, translate
from decimal import Decimal, getcontext
import time
import json
from itertools import product

getcontext().prec = 30

# Tree polygon vertices
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]
TREE_COORDS = list(zip(TX, TY))

def get_tree_polygon(x, y, angle):
    poly = Polygon(TREE_COORDS)
    poly = rotate(poly, angle, origin=(0, 0))
    poly = translate(poly, x, y)
    return poly

def calculate_bounding_box(trees):
    all_coords = []
    for x, y, angle in trees:
        poly = get_tree_polygon(x, y, angle)
        all_coords.extend(poly.exterior.coords)
    
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    
    return max(max(xs) - min(xs), max(ys) - min(ys))

def has_overlap_fast(trees, changed_idx):
    n = len(trees)
    if n <= 1:
        return False
    
    changed_poly = get_tree_polygon(*trees[changed_idx])
    
    for i in range(n):
        if i == changed_idx:
            continue
        other_poly = get_tree_polygon(*trees[i])
        if changed_poly.intersects(other_poly) and not changed_poly.touches(other_poly):
            intersection = changed_poly.intersection(other_poly)
            if intersection.area > 1e-15:
                return True
    return False

def exhaustive_search_single_tree(trees, tree_idx, dx_range=0.01, dy_range=0.01, da_range=2.0, steps=11):
    """
    Exhaustively search around a single tree's position.
    """
    n = len(trees)
    x, y, angle = trees[tree_idx]
    baseline_side = calculate_bounding_box(trees)
    
    best_side = baseline_side
    best_pos = (x, y, angle)
    improvements_found = 0
    
    # Generate search grid
    dx_values = np.linspace(-dx_range, dx_range, steps)
    dy_values = np.linspace(-dy_range, dy_range, steps)
    da_values = np.linspace(-da_range, da_range, steps)
    
    total_positions = len(dx_values) * len(dy_values) * len(da_values)
    checked = 0
    
    for dx in dx_values:
        for dy in dy_values:
            for da in da_values:
                new_trees = [list(t) for t in trees]
                new_trees[tree_idx] = [x + dx, y + dy, angle + da]
                
                if not has_overlap_fast(new_trees, tree_idx):
                    new_side = calculate_bounding_box(new_trees)
                    
                    if new_side < best_side:
                        best_side = new_side
                        best_pos = (x + dx, y + dy, angle + da)
                        improvements_found += 1
                
                checked += 1
    
    improvement = baseline_side - best_side
    return best_pos, best_side, improvement, improvements_found

def exhaustive_search_all_trees(trees, n, dx_range=0.005, dy_range=0.005, da_range=1.0, steps=7):
    """
    Exhaustively search around all trees' positions.
    """
    current_trees = [list(t) for t in trees]
    baseline_side = calculate_bounding_box(current_trees)
    total_improvement = 0
    
    print(f"  Baseline side: {baseline_side:.8f}")
    
    for tree_idx in range(n):
        best_pos, best_side, improvement, improvements_found = exhaustive_search_single_tree(
            current_trees, tree_idx, dx_range, dy_range, da_range, steps
        )
        
        if improvement > 1e-10:
            current_trees[tree_idx] = list(best_pos)
            total_improvement += improvement
            print(f"    Tree {tree_idx}: improvement = {improvement:.10f}")
    
    final_side = calculate_bounding_box(current_trees)
    print(f"  Final side: {final_side:.8f}")
    print(f"  Total improvement: {total_improvement:.10f}")
    
    return current_trees, final_side, total_improvement

def load_baseline(csv_path):
    df = pd.read_csv(csv_path)
    
    solutions = {}
    for n in range(1, 201):
        n_df = df[df['id'].str.startswith(f'{n:03d}_')]
        trees = []
        for _, row in n_df.iterrows():
            x = float(str(row['x']).replace('s', ''))
            y = float(str(row['y']).replace('s', ''))
            angle = float(str(row['deg']).replace('s', ''))
            trees.append([x, y, angle])
        solutions[n] = trees
    
    return solutions

def test_exhaustive(baseline_solutions, test_ns=[5, 10]):
    """Test exhaustive search on small N values."""
    print("=" * 60)
    print("EXHAUSTIVE SEARCH AROUND BASELINE")
    print("=" * 60)
    
    results = {}
    
    for n in test_ns:
        print(f"\nN={n}:")
        trees = baseline_solutions[n]
        baseline_side = calculate_bounding_box(trees)
        baseline_score = (baseline_side ** 2) / n
        
        start_time = time.time()
        improved_trees, improved_side, total_improvement = exhaustive_search_all_trees(
            trees, n, dx_range=0.005, dy_range=0.005, da_range=1.0, steps=7
        )
        elapsed = time.time() - start_time
        
        improved_score = (improved_side ** 2) / n
        score_improvement = baseline_score - improved_score
        
        results[n] = {
            'baseline_side': baseline_side,
            'baseline_score': baseline_score,
            'improved_side': improved_side,
            'improved_score': improved_score,
            'improvement': score_improvement,
            'time': elapsed
        }
        
        print(f"  Score improvement: {score_improvement:.10f}")
        print(f"  Time: {elapsed:.2f}s")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_improvement = sum(r['improvement'] for r in results.values())
    print(f"Total score improvement: {total_improvement:.10f}")
    
    return results

if __name__ == "__main__":
    baseline_path = "/home/nonroot/snapshots/santa-2025/21337353543/submission/submission.csv"
    print(f"Loading baseline from {baseline_path}")
    baseline_solutions = load_baseline(baseline_path)
    
    # Test exhaustive search
    test_results = test_exhaustive(baseline_solutions, test_ns=[5, 10])
    
    # Save results
    with open('/home/code/experiments/008_simulated_annealing/exhaustive_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\nResults saved to exhaustive_results.json")
