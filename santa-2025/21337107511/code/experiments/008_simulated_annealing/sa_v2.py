"""
Simulated Annealing v2 - More aggressive approach

Key changes:
1. Larger initial perturbations
2. Separate rotation-only and translation-only moves
3. More iterations
4. Different temperature schedule
5. Multi-restart
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

def calculate_bounding_box(trees):
    """Calculate the bounding box side length for a set of trees."""
    all_coords = []
    for x, y, angle in trees:
        poly = get_tree_polygon(x, y, angle)
        all_coords.extend(poly.exterior.coords)
    
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    
    side = max(max(xs) - min(xs), max(ys) - min(ys))
    return side

def has_overlap_fast(trees, changed_idx):
    """Fast overlap check - only check the changed tree against others."""
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

def simulated_annealing_v2(trees, n, max_iter=30000, verbose=False):
    """
    Simulated annealing v2 with more aggressive exploration.
    
    Key changes:
    - Higher starting temperature
    - Larger perturbations
    - Different move types (translation, rotation, combined)
    """
    current = [list(t) for t in trees]
    current_side = calculate_bounding_box(current)
    best = [list(t) for t in current]
    best_side = current_side
    
    # More aggressive temperature schedule
    T = 2.0  # Higher starting temperature
    T_min = 1e-8
    cooling = 0.9997  # Slower cooling
    
    accepted = 0
    improved = 0
    
    for iteration in range(max_iter):
        # Random tree selection
        idx = random.randint(0, n - 1)
        x, y, angle = current[idx]
        
        # Choose move type
        move_type = random.choice(['translate', 'rotate', 'combined', 'swap'])
        
        new_trees = [list(t) for t in current]
        
        if move_type == 'translate':
            # Translation only - larger moves at high T
            scale = max(0.5 * T, 0.001)  # Up to 0.5 units at T=1
            dx = random.uniform(-scale, scale)
            dy = random.uniform(-scale, scale)
            new_trees[idx] = [x + dx, y + dy, angle]
            
        elif move_type == 'rotate':
            # Rotation only - larger rotations at high T
            scale = max(45 * T, 0.1)  # Up to 45 degrees at T=1
            da = random.uniform(-scale, scale)
            new_trees[idx] = [x, y, (angle + da) % 360]
            
        elif move_type == 'combined':
            # Both translation and rotation
            t_scale = max(0.3 * T, 0.001)
            r_scale = max(30 * T, 0.1)
            dx = random.uniform(-t_scale, t_scale)
            dy = random.uniform(-t_scale, t_scale)
            da = random.uniform(-r_scale, r_scale)
            new_trees[idx] = [x + dx, y + dy, (angle + da) % 360]
            
        elif move_type == 'swap' and n > 1:
            # Swap positions of two trees
            idx2 = random.randint(0, n - 1)
            while idx2 == idx:
                idx2 = random.randint(0, n - 1)
            x2, y2, angle2 = current[idx2]
            new_trees[idx] = [x2, y2, angle]  # Keep original rotation
            new_trees[idx2] = [x, y, angle2]
        
        # Check for overlaps
        overlap = has_overlap_fast(new_trees, idx)
        if move_type == 'swap' and not overlap:
            overlap = has_overlap_fast(new_trees, idx2)
        
        if not overlap:
            new_side = calculate_bounding_box(new_trees)
            delta = new_side - current_side
            
            # Accept with Metropolis criterion
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current = new_trees
                current_side = new_side
                accepted += 1
                
                if new_side < best_side:
                    best_side = new_side
                    best = [list(t) for t in new_trees]
                    improved += 1
                    if verbose:
                        print(f"  Iter {iteration}: New best side = {best_side:.8f} (T={T:.6f})")
        
        # Cool down
        T = max(T * cooling, T_min)
        
        if verbose and iteration % 10000 == 0:
            print(f"  Iter {iteration}: T={T:.6f}, current={current_side:.6f}, best={best_side:.6f}")
    
    return best, best_side, accepted, improved

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

def calculate_score(solutions):
    """Calculate total score for all N values."""
    total = 0
    for n, trees in solutions.items():
        side = calculate_bounding_box(trees)
        score = (side ** 2) / n
        total += score
    return total

def test_small_n_v2(baseline_solutions, test_ns=[5, 10, 15, 20], max_iter=30000, num_restarts=3):
    """Test SA v2 on small N values with multiple restarts."""
    print("=" * 60)
    print("TESTING SIMULATED ANNEALING V2 ON SMALL N VALUES")
    print("=" * 60)
    
    results = {}
    
    for n in test_ns:
        print(f"\nN={n}:")
        trees = baseline_solutions[n]
        baseline_side = calculate_bounding_box(trees)
        baseline_score = (baseline_side ** 2) / n
        
        best_overall_side = baseline_side
        best_overall_trees = trees
        
        for restart in range(num_restarts):
            print(f"  Restart {restart + 1}/{num_restarts}:")
            start_time = time.time()
            
            # Start from baseline with small random perturbation
            perturbed_trees = [[x + random.uniform(-0.01, 0.01), 
                               y + random.uniform(-0.01, 0.01), 
                               angle] for x, y, angle in trees]
            
            improved_trees, improved_side, accepted, improved_count = simulated_annealing_v2(
                perturbed_trees, n, max_iter=max_iter, verbose=True
            )
            elapsed = time.time() - start_time
            
            if improved_side < best_overall_side:
                best_overall_side = improved_side
                best_overall_trees = improved_trees
                print(f"    NEW BEST: {improved_side:.8f}")
            
            print(f"    Time: {elapsed:.2f}s, Accepted: {accepted}/{max_iter}")
        
        improved_score = (best_overall_side ** 2) / n
        improvement = baseline_score - improved_score
        
        results[n] = {
            'baseline_side': baseline_side,
            'baseline_score': baseline_score,
            'improved_side': best_overall_side,
            'improved_score': improved_score,
            'improvement': improvement
        }
        
        print(f"\n  FINAL for N={n}:")
        print(f"    Baseline: side={baseline_side:.6f}, score={baseline_score:.6f}")
        print(f"    Improved: side={best_overall_side:.6f}, score={improved_score:.6f}")
        print(f"    Improvement: {improvement:.8f} ({improvement/baseline_score*100:.4f}%)")
    
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
    
    baseline_score = calculate_score(baseline_solutions)
    print(f"Baseline score: {baseline_score:.6f}")
    
    # Test on small N with v2
    test_results = test_small_n_v2(baseline_solutions, test_ns=[5, 10, 15], max_iter=20000, num_restarts=2)
    
    # Save results
    with open('/home/code/experiments/008_simulated_annealing/test_results_v2.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\nTest results saved to test_results_v2.json")
