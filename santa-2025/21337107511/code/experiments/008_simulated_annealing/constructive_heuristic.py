"""
Constructive Heuristic for Tree Packing

Instead of perturbing the baseline, build solutions from scratch.
This can find different local optima that may be better.

Approach: Bottom-Left Fill with rotation optimization
1. Place trees one at a time
2. For each tree, find the position that minimizes bounding box
3. Try multiple rotation angles
"""

import numpy as np
import pandas as pd
import math
import random
from shapely import Polygon
from shapely.affinity import rotate, translate
from decimal import Decimal, getcontext
import time
import json

getcontext().prec = 30

# Tree polygon vertices
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]
TREE_COORDS = list(zip(TX, TY))

def get_tree_polygon(x, y, angle):
    """Create tree polygon at position (x, y) with rotation angle (degrees)."""
    poly = Polygon(TREE_COORDS)
    poly = rotate(poly, angle, origin=(0, 0))
    poly = translate(poly, x, y)
    return poly

def calculate_bounding_box_from_polys(polygons):
    """Calculate bounding box from list of polygons."""
    all_coords = []
    for poly in polygons:
        all_coords.extend(poly.exterior.coords)
    
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    
    return max(max(xs) - min(xs), max(ys) - min(ys))

def calculate_bounding_box(trees):
    """Calculate the bounding box side length for a set of trees."""
    polygons = [get_tree_polygon(x, y, angle) for x, y, angle in trees]
    return calculate_bounding_box_from_polys(polygons)

def has_overlap_with_placed(new_poly, placed_polys):
    """Check if new polygon overlaps with any placed polygon."""
    for poly in placed_polys:
        if new_poly.intersects(poly) and not new_poly.touches(poly):
            intersection = new_poly.intersection(poly)
            if intersection.area > 1e-15:
                return True
    return False

def find_best_position(placed_polys, angle, grid_step=0.1):
    """Find the best position for a new tree with given angle."""
    if not placed_polys:
        # First tree - place at origin
        return 0, 0, 0  # x, y, bbox_side
    
    # Get current bounding box
    all_coords = []
    for poly in placed_polys:
        all_coords.extend(poly.exterior.coords)
    
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Search area: around the current bounding box
    search_min_x = min_x - 1.5
    search_max_x = max_x + 1.5
    search_min_y = min_y - 1.5
    search_max_y = max_y + 1.5
    
    best_pos = None
    best_side = float('inf')
    
    # Grid search for position
    x = search_min_x
    while x <= search_max_x:
        y = search_min_y
        while y <= search_max_y:
            new_poly = get_tree_polygon(x, y, angle)
            
            if not has_overlap_with_placed(new_poly, placed_polys):
                # Calculate new bounding box
                test_polys = placed_polys + [new_poly]
                side = calculate_bounding_box_from_polys(test_polys)
                
                if side < best_side:
                    best_side = side
                    best_pos = (x, y)
            
            y += grid_step
        x += grid_step
    
    if best_pos is None:
        # No valid position found - expand search
        return find_best_position(placed_polys, angle, grid_step * 2)
    
    return best_pos[0], best_pos[1], best_side

def construct_solution(n, angles_to_try=[0, 45, 90, 135, 180, 225, 270, 315], grid_step=0.1):
    """Construct a solution by placing trees one at a time."""
    trees = []
    placed_polys = []
    
    for i in range(n):
        best_tree = None
        best_side = float('inf')
        
        # Try different angles
        for angle in angles_to_try:
            x, y, side = find_best_position(placed_polys, angle, grid_step)
            
            if side < best_side:
                best_side = side
                best_tree = (x, y, angle)
        
        if best_tree is None:
            raise ValueError(f"Could not place tree {i}")
        
        trees.append(list(best_tree))
        placed_polys.append(get_tree_polygon(*best_tree))
    
    return trees, best_side

def load_baseline(csv_path):
    """Load baseline solution from CSV."""
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

def test_constructive(baseline_solutions, test_ns=[2, 3, 4, 5]):
    """Test constructive heuristic on small N values."""
    print("=" * 60)
    print("TESTING CONSTRUCTIVE HEURISTIC")
    print("=" * 60)
    
    results = {}
    
    for n in test_ns:
        print(f"\nN={n}:")
        baseline_trees = baseline_solutions[n]
        baseline_side = calculate_bounding_box(baseline_trees)
        baseline_score = (baseline_side ** 2) / n
        
        start_time = time.time()
        
        # Try constructive heuristic
        constructed_trees, constructed_side = construct_solution(
            n, 
            angles_to_try=[0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345],
            grid_step=0.05
        )
        
        elapsed = time.time() - start_time
        
        constructed_score = (constructed_side ** 2) / n
        improvement = baseline_score - constructed_score
        
        results[n] = {
            'baseline_side': baseline_side,
            'baseline_score': baseline_score,
            'constructed_side': constructed_side,
            'constructed_score': constructed_score,
            'improvement': improvement,
            'time': elapsed
        }
        
        print(f"  Baseline: side={baseline_side:.6f}, score={baseline_score:.6f}")
        print(f"  Constructed: side={constructed_side:.6f}, score={constructed_score:.6f}")
        print(f"  Improvement: {improvement:.8f} ({improvement/baseline_score*100:.4f}%)")
        print(f"  Time: {elapsed:.2f}s")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_improvement = sum(r['improvement'] for r in results.values())
    print(f"Total improvement: {total_improvement:.8f}")
    
    return results

if __name__ == "__main__":
    # Load baseline
    baseline_path = "/home/nonroot/snapshots/santa-2025/21337353543/submission/submission.csv"
    print(f"Loading baseline from {baseline_path}")
    baseline_solutions = load_baseline(baseline_path)
    
    # Test constructive heuristic on very small N
    test_results = test_constructive(baseline_solutions, test_ns=[2, 3, 4, 5])
    
    # Save results
    with open('/home/code/experiments/008_simulated_annealing/constructive_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\nResults saved to constructive_results.json")
