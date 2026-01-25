"""
Simulated Annealing for Tree Packing Optimization

Key insight: The baseline is at a LOCAL OPTIMUM. Simple local search cannot escape it.
SA accepts worse solutions with probability exp(-delta/T), allowing escape from local optima.
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
import os

getcontext().prec = 30

# Tree polygon vertices (from competition)
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]
TREE_COORDS = list(zip(TX, TY))

# SA parameters
T_START = 1.0
T_END = 0.00001
MAX_ITER = 20000  # Per N value

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

def has_overlap(trees):
    """Check if any trees overlap (excluding touching)."""
    n = len(trees)
    if n <= 1:
        return False
    
    polygons = [get_tree_polygon(x, y, angle) for x, y, angle in trees]
    
    for i in range(n):
        for j in range(i + 1, n):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                # Check if it's a real overlap (not just touching)
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > 1e-15:  # Small tolerance
                    return True
    return False

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

def simulated_annealing(trees, n, max_iter=MAX_ITER, t_start=T_START, t_end=T_END, verbose=False):
    """
    Simulated annealing for tree packing.
    
    Key: Accept WORSE solutions with probability exp(-delta/T)
    This allows escaping local optima!
    """
    current = [list(t) for t in trees]  # Make mutable copies
    current_side = calculate_bounding_box(current)
    best = [list(t) for t in current]
    best_side = current_side
    
    T = t_start
    cooling = (t_end / t_start) ** (1.0 / max_iter)
    
    accepted = 0
    improved = 0
    
    for iteration in range(max_iter):
        # Random perturbation
        idx = random.randint(0, n - 1)
        x, y, angle = current[idx]
        
        # Perturbation magnitudes - scale with temperature
        # At high T: large moves (0.1 units, 10 degrees)
        # At low T: small moves (0.001 units, 0.1 degrees)
        scale = max(T, 0.01)  # Don't go too small
        dx = random.uniform(-0.1, 0.1) * scale
        dy = random.uniform(-0.1, 0.1) * scale
        da = random.uniform(-10, 10) * scale
        
        # Apply perturbation
        new_trees = [list(t) for t in current]
        new_trees[idx] = [x + dx, y + dy, (angle + da) % 360]
        
        # Check for overlaps (fast check - only changed tree)
        if not has_overlap_fast(new_trees, idx):
            new_side = calculate_bounding_box(new_trees)
            delta = new_side - current_side
            
            # KEY: Accept worse solutions with probability exp(-delta/T)
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current = new_trees
                current_side = new_side
                accepted += 1
                
                if new_side < best_side:
                    best_side = new_side
                    best = [list(t) for t in new_trees]
                    improved += 1
                    if verbose:
                        print(f"  Iter {iteration}: New best side = {best_side:.8f}")
        
        # Cool down
        T *= cooling
        
        # Progress report
        if verbose and iteration % 5000 == 0:
            print(f"  Iter {iteration}: T={T:.6f}, current={current_side:.6f}, best={best_side:.6f}, accepted={accepted}")
    
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

def test_small_n(baseline_solutions, test_ns=[10, 15, 20, 25, 30], max_iter=10000):
    """Test SA on small N values first."""
    print("=" * 60)
    print("TESTING SIMULATED ANNEALING ON SMALL N VALUES")
    print("=" * 60)
    
    results = {}
    
    for n in test_ns:
        print(f"\nN={n}:")
        trees = baseline_solutions[n]
        baseline_side = calculate_bounding_box(trees)
        baseline_score = (baseline_side ** 2) / n
        
        # Run SA
        start_time = time.time()
        improved_trees, improved_side, accepted, improved_count = simulated_annealing(
            trees, n, max_iter=max_iter, verbose=True
        )
        elapsed = time.time() - start_time
        
        improved_score = (improved_side ** 2) / n
        improvement = baseline_score - improved_score
        
        results[n] = {
            'baseline_side': baseline_side,
            'baseline_score': baseline_score,
            'improved_side': improved_side,
            'improved_score': improved_score,
            'improvement': improvement,
            'accepted': accepted,
            'improved_count': improved_count,
            'time': elapsed
        }
        
        print(f"  Baseline: side={baseline_side:.6f}, score={baseline_score:.6f}")
        print(f"  Improved: side={improved_side:.6f}, score={improved_score:.6f}")
        print(f"  Improvement: {improvement:.8f} ({improvement/baseline_score*100:.4f}%)")
        print(f"  Accepted: {accepted}/{max_iter}, Improved: {improved_count}")
        print(f"  Time: {elapsed:.2f}s")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_improvement = sum(r['improvement'] for r in results.values())
    avg_improvement = total_improvement / len(results)
    print(f"Total improvement: {total_improvement:.8f}")
    print(f"Average improvement per N: {avg_improvement:.8f}")
    
    if total_improvement > 0.001:
        print("\n✅ SA shows promise! Scale up to all N values.")
    else:
        print("\n⚠️ SA shows minimal improvement. May need parameter tuning.")
    
    return results

def run_full_optimization(baseline_solutions, max_iter=20000):
    """Run SA on all N values."""
    print("=" * 60)
    print("RUNNING FULL SIMULATED ANNEALING OPTIMIZATION")
    print("=" * 60)
    
    improved_solutions = {}
    total_baseline = 0
    total_improved = 0
    improvements_found = 0
    
    for n in range(1, 201):
        trees = baseline_solutions[n]
        baseline_side = calculate_bounding_box(trees)
        baseline_score = (baseline_side ** 2) / n
        total_baseline += baseline_score
        
        # Run SA
        improved_trees, improved_side, accepted, improved_count = simulated_annealing(
            trees, n, max_iter=max_iter, verbose=False
        )
        
        improved_score = (improved_side ** 2) / n
        improvement = baseline_score - improved_score
        
        # Keep best solution
        if improved_score < baseline_score:
            improved_solutions[n] = improved_trees
            total_improved += improved_score
            improvements_found += 1
            print(f"N={n}: IMPROVED by {improvement:.8f} ({improvement/baseline_score*100:.4f}%)")
        else:
            improved_solutions[n] = trees  # Keep baseline
            total_improved += baseline_score
        
        if n % 20 == 0:
            print(f"Progress: {n}/200 complete, {improvements_found} improvements found")
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Baseline score: {total_baseline:.6f}")
    print(f"Improved score: {total_improved:.6f}")
    print(f"Total improvement: {total_baseline - total_improved:.6f}")
    print(f"N values improved: {improvements_found}/200")
    
    return improved_solutions, total_improved

def save_submission(solutions, output_path):
    """Save solutions to submission CSV with proper formatting."""
    rows = []
    for n in range(1, 201):
        trees = solutions[n]
        for i, (x, y, angle) in enumerate(trees):
            # Format with 's' prefix and high precision
            x_str = f"s{Decimal(str(x)):.18f}"
            y_str = f"s{Decimal(str(y)):.18f}"
            deg_str = f"s{Decimal(str(angle)):.18f}"
            rows.append({
                'id': f'{n:03d}_{i}',
                'x': x_str,
                'y': y_str,
                'deg': deg_str
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")

if __name__ == "__main__":
    # Load baseline
    baseline_path = "/home/nonroot/snapshots/21337353543.csv"  # Valid baseline
    print(f"Loading baseline from {baseline_path}")
    baseline_solutions = load_baseline(baseline_path)
    
    baseline_score = calculate_score(baseline_solutions)
    print(f"Baseline score: {baseline_score:.6f}")
    
    # Test on small N first
    test_results = test_small_n(baseline_solutions, test_ns=[10, 15, 20, 25, 30], max_iter=10000)
    
    # Save test results
    with open('/home/code/experiments/008_simulated_annealing/test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\nTest results saved to test_results.json")
