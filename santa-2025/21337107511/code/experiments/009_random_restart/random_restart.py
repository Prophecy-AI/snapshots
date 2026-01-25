"""
Random Restart Strategy for Tree Packing

Key insight: The baseline is at ONE local optimum. There may be BETTER local optima
that can only be found by starting from DIFFERENT initial configurations.

Approach:
1. Generate random valid configurations (not starting from baseline)
2. Apply local search to each
3. Keep the best solution found

This is fundamentally different from SA which perturbs an existing solution.
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

# Tree dimensions (approximate)
TREE_WIDTH = 0.7  # max(TX) - min(TX)
TREE_HEIGHT = 1.0  # max(TY) - min(TY)

def get_tree_polygon(x, y, angle):
    """Create tree polygon at position (x, y) with rotation angle (degrees)."""
    poly = Polygon(TREE_COORDS)
    poly = rotate(poly, angle, origin=(0, 0))
    poly = translate(poly, x, y)
    return poly

def calculate_bounding_box(trees):
    """Calculate the bounding box side length for a set of trees."""
    all_coords = []
    for x, y, angle in trees:
        poly = get_tree_polygon(x, y, angle)
        all_coords.extend(poly.exterior.coords)
    
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    
    return max(max(xs) - min(xs), max(ys) - min(ys))

def has_overlap(trees):
    """Check if any trees overlap."""
    n = len(trees)
    if n <= 1:
        return False
    
    polygons = [get_tree_polygon(x, y, angle) for x, y, angle in trees]
    
    for i in range(n):
        for j in range(i + 1, n):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > 1e-15:
                    return True
    return False

def has_overlap_with_placed(new_poly, placed_polys):
    """Check if new polygon overlaps with any placed polygon."""
    for poly in placed_polys:
        if new_poly.intersects(poly) and not new_poly.touches(poly):
            intersection = new_poly.intersection(poly)
            if intersection.area > 1e-15:
                return True
    return False

def generate_random_valid_config(n, max_attempts=1000):
    """
    Generate a random valid configuration of n trees.
    
    Strategy: Place trees one at a time in random positions,
    ensuring no overlaps.
    """
    # Estimate required area based on n
    # Each tree needs roughly 0.7 * 1.0 = 0.7 square units
    # With some packing efficiency, estimate side length
    estimated_side = math.sqrt(n * 0.7 / 0.5)  # 50% packing efficiency
    
    trees = []
    placed_polys = []
    
    for i in range(n):
        placed = False
        attempts = 0
        
        while not placed and attempts < max_attempts:
            # Random position within estimated bounds
            x = random.uniform(-estimated_side/2, estimated_side/2)
            y = random.uniform(-estimated_side/2, estimated_side/2)
            # Random angle (multiples of 45 degrees work well)
            angle = random.choice([0, 45, 90, 135, 180, 225, 270, 315])
            
            new_poly = get_tree_polygon(x, y, angle)
            
            if not has_overlap_with_placed(new_poly, placed_polys):
                trees.append([x, y, angle])
                placed_polys.append(new_poly)
                placed = True
            
            attempts += 1
        
        if not placed:
            # Expand search area and try again
            estimated_side *= 1.5
            attempts = 0
            while not placed and attempts < max_attempts:
                x = random.uniform(-estimated_side/2, estimated_side/2)
                y = random.uniform(-estimated_side/2, estimated_side/2)
                angle = random.choice([0, 45, 90, 135, 180, 225, 270, 315])
                
                new_poly = get_tree_polygon(x, y, angle)
                
                if not has_overlap_with_placed(new_poly, placed_polys):
                    trees.append([x, y, angle])
                    placed_polys.append(new_poly)
                    placed = True
                
                attempts += 1
        
        if not placed:
            raise ValueError(f"Could not place tree {i} after {max_attempts*2} attempts")
    
    return trees

def local_search(trees, max_iter=1000):
    """
    Simple local search to improve a configuration.
    Try small moves for each tree and keep improvements.
    """
    n = len(trees)
    current = [list(t) for t in trees]
    current_side = calculate_bounding_box(current)
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        
        for idx in range(n):
            x, y, angle = current[idx]
            
            # Try small moves
            for dx in [-0.05, 0, 0.05]:
                for dy in [-0.05, 0, 0.05]:
                    for da in [-5, 0, 5]:
                        if dx == 0 and dy == 0 and da == 0:
                            continue
                        
                        new_trees = [list(t) for t in current]
                        new_trees[idx] = [x + dx, y + dy, angle + da]
                        
                        if not has_overlap(new_trees):
                            new_side = calculate_bounding_box(new_trees)
                            
                            if new_side < current_side - 1e-10:
                                current = new_trees
                                current_side = new_side
                                improved = True
    
    return current, current_side

def random_restart_optimization(n, num_restarts=50, verbose=False):
    """
    Random restart optimization.
    
    Generate multiple random configurations and optimize each.
    Return the best solution found.
    """
    best_solution = None
    best_side = float('inf')
    
    for restart in range(num_restarts):
        try:
            # Generate random valid configuration
            trees = generate_random_valid_config(n)
            initial_side = calculate_bounding_box(trees)
            
            # Apply local search
            optimized, optimized_side = local_search(trees, max_iter=100)
            
            if optimized_side < best_side:
                best_side = optimized_side
                best_solution = optimized
                if verbose:
                    print(f"  Restart {restart}: initial={initial_side:.4f}, optimized={optimized_side:.4f} (NEW BEST)")
            elif verbose and restart % 10 == 0:
                print(f"  Restart {restart}: initial={initial_side:.4f}, optimized={optimized_side:.4f}")
        
        except ValueError as e:
            if verbose:
                print(f"  Restart {restart}: Failed to generate config - {e}")
    
    return best_solution, best_side

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

def test_random_restart(baseline_solutions, test_ns=[5, 10, 15, 20], num_restarts=30):
    """Test random restart on small N values."""
    print("=" * 60)
    print("TESTING RANDOM RESTART STRATEGY")
    print("=" * 60)
    
    results = {}
    
    for n in test_ns:
        print(f"\nN={n}:")
        baseline_trees = baseline_solutions[n]
        baseline_side = calculate_bounding_box(baseline_trees)
        baseline_score = (baseline_side ** 2) / n
        
        start_time = time.time()
        best_trees, best_side = random_restart_optimization(n, num_restarts=num_restarts, verbose=True)
        elapsed = time.time() - start_time
        
        best_score = (best_side ** 2) / n
        improvement = baseline_score - best_score
        
        results[n] = {
            'baseline_side': baseline_side,
            'baseline_score': baseline_score,
            'best_side': best_side,
            'best_score': best_score,
            'improvement': improvement,
            'time': elapsed
        }
        
        print(f"\n  RESULTS for N={n}:")
        print(f"    Baseline: side={baseline_side:.6f}, score={baseline_score:.6f}")
        print(f"    Best found: side={best_side:.6f}, score={best_score:.6f}")
        print(f"    Improvement: {improvement:.8f} ({improvement/baseline_score*100:.4f}%)")
        print(f"    Time: {elapsed:.2f}s")
        
        if improvement > 0:
            print(f"    ✅ BEAT BASELINE!")
        else:
            print(f"    ❌ Did not beat baseline (gap: {-improvement:.6f})")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_improvement = sum(r['improvement'] for r in results.values())
    beat_baseline = sum(1 for r in results.values() if r['improvement'] > 0)
    
    print(f"Total improvement: {total_improvement:.8f}")
    print(f"N values that beat baseline: {beat_baseline}/{len(test_ns)}")
    
    if beat_baseline > 0:
        print("\n✅ Random restart CAN find better solutions! Scale up to all N.")
    else:
        print("\n⚠️ Random restart did not beat baseline on any N value.")
    
    return results

if __name__ == "__main__":
    # Load baseline
    baseline_path = "/home/nonroot/snapshots/santa-2025/21337353543/submission/submission.csv"
    print(f"Loading baseline from {baseline_path}")
    baseline_solutions = load_baseline(baseline_path)
    
    # Test random restart
    test_results = test_random_restart(baseline_solutions, test_ns=[5, 10, 15, 20], num_restarts=30)
    
    # Save results
    with open('/home/code/experiments/009_random_restart/test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\nResults saved to test_results.json")
