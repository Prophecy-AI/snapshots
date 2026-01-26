"""
Backpacking Technique from crodoc kernel

Key insight: Start from N=200 and iterate backward.
When smaller N has worse score than expected, copy trees from larger N.
This propagates good packing patterns from large N to small N.

This is different from "rebuild from corners" because:
1. It uses the FIRST n trees (by index), not trees closest to a corner
2. It propagates improvements backward through the chain
3. It updates the "best layout" as it goes
"""

import numpy as np
from numba import njit
import pandas as pd
import time
import json
from decimal import Decimal, getcontext

getcontext().prec = 30

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125], dtype=np.float64)
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5], dtype=np.float64)

@njit
def rotate_vertices(tx, ty, angle_deg):
    angle_rad = angle_deg * np.pi / 180.0
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rx = tx * cos_a - ty * sin_a
    ry = tx * sin_a + ty * cos_a
    return rx, ry

@njit
def get_tree_vertices(x, y, angle):
    rx, ry = rotate_vertices(TX, TY, angle)
    return rx + x, ry + y

@njit
def compute_bbox(trees_x, trees_y, trees_angle, n):
    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf
    
    for i in range(n):
        vx, vy = get_tree_vertices(trees_x[i], trees_y[i], trees_angle[i])
        for j in range(15):
            if vx[j] < min_x: min_x = vx[j]
            if vx[j] > max_x: max_x = vx[j]
            if vy[j] < min_y: min_y = vy[j]
            if vy[j] > max_y: max_y = vy[j]
    
    return max(max_x - min_x, max_y - min_y)

@njit
def point_in_polygon(px, py, poly_x, poly_y, n_vertices):
    inside = False
    j = n_vertices - 1
    for i in range(n_vertices):
        if ((poly_y[i] > py) != (poly_y[j] > py)) and \
           (px < (poly_x[j] - poly_x[i]) * (py - poly_y[i]) / (poly_y[j] - poly_y[i]) + poly_x[i]):
            inside = not inside
        j = i
    return inside

@njit
def segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    d1 = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
    d2 = (x4 - x3) * (y2 - y3) - (y4 - y3) * (x2 - x3)
    d3 = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    d4 = (x2 - x1) * (y4 - y1) - (y2 - y1) * (x4 - x1)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

@njit
def polygons_overlap(vx1, vy1, vx2, vy2, n1, n2):
    for i in range(n1):
        if point_in_polygon(vx1[i], vy1[i], vx2, vy2, n2):
            return True
    for i in range(n2):
        if point_in_polygon(vx2[i], vy2[i], vx1, vy1, n1):
            return True
    for i in range(n1):
        i_next = (i + 1) % n1
        for j in range(n2):
            j_next = (j + 1) % n2
            if segments_intersect(vx1[i], vy1[i], vx1[i_next], vy1[i_next],
                                  vx2[j], vy2[j], vx2[j_next], vy2[j_next]):
                return True
    return False

@njit
def has_any_overlap(trees_x, trees_y, trees_angle, n):
    """Check if any trees overlap."""
    for i in range(n):
        vx1, vy1 = get_tree_vertices(trees_x[i], trees_y[i], trees_angle[i])
        for j in range(i + 1, n):
            vx2, vy2 = get_tree_vertices(trees_x[j], trees_y[j], trees_angle[j])
            
            if max(vx1) < min(vx2) or max(vx2) < min(vx1):
                continue
            if max(vy1) < min(vy2) or max(vy2) < min(vy1):
                continue
            
            if polygons_overlap(vx1, vy1, vx2, vy2, 15, 15):
                return True
    return False

def load_baseline(csv_path):
    """Load baseline solution from CSV."""
    df = pd.read_csv(csv_path)
    
    solutions = {}
    for n in range(1, 201):
        n_df = df[df['id'].str.startswith(f'{n:03d}_')]
        trees_x = np.zeros(n, dtype=np.float64)
        trees_y = np.zeros(n, dtype=np.float64)
        trees_angle = np.zeros(n, dtype=np.float64)
        
        for idx, (_, row) in enumerate(n_df.iterrows()):
            trees_x[idx] = float(str(row['x']).replace('s', ''))
            trees_y[idx] = float(str(row['y']).replace('s', ''))
            trees_angle[idx] = float(str(row['deg']).replace('s', ''))
        
        solutions[n] = (trees_x, trees_y, trees_angle)
    
    return solutions

def compute_score(trees_x, trees_y, trees_angle, n):
    """Compute score for n trees."""
    side = compute_bbox(trees_x, trees_y, trees_angle, n)
    return (side ** 2) / n

def backpacking(solutions, verbose=True):
    """
    Backpacking technique: Start from N=200 and iterate backward.
    
    Key insight: If N=200 layout is well-optimized, its first K trees
    might be better than the existing K-tree solution.
    """
    improvements = []
    
    # Start with N=200 as the best layout
    best_x = solutions[200][0].copy()
    best_y = solutions[200][1].copy()
    best_angle = solutions[200][2].copy()
    
    if verbose:
        print("Starting backpacking from N=200...")
    
    for n in range(199, 0, -1):
        current_x, current_y, current_angle = solutions[n]
        current_score = compute_score(current_x, current_y, current_angle, n)
        
        # Try using first n trees from best_layout
        subset_x = best_x[:n].copy()
        subset_y = best_y[:n].copy()
        subset_angle = best_angle[:n].copy()
        
        # Check for overlaps in subset
        if has_any_overlap(subset_x, subset_y, subset_angle, n):
            # Subset has overlaps, can't use it
            # Update best_layout with current solution
            best_x[:n] = current_x
            best_y[:n] = current_y
            best_angle[:n] = current_angle
            continue
        
        subset_score = compute_score(subset_x, subset_y, subset_angle, n)
        
        if subset_score < current_score - 1e-10:
            improvement = current_score - subset_score
            if verbose:
                print(f"  N={n}: IMPROVEMENT! {current_score:.10f} -> {subset_score:.10f} (gain: {improvement:.10f})")
            improvements.append((n, improvement))
            # Update solutions with the better subset
            solutions[n] = (subset_x, subset_y, subset_angle)
        else:
            # Current solution is better, update best_layout
            best_x[:n] = current_x
            best_y[:n] = current_y
            best_angle[:n] = current_angle
    
    return improvements

def backpacking_reverse(solutions, verbose=True):
    """
    Reverse backpacking: Start from N=1 and iterate forward.
    
    Key insight: If N=K layout is well-optimized, adding one more tree
    from N=K+1 might give a better K+1 solution.
    """
    improvements = []
    
    # Start with N=1 as the best layout
    best_x = np.zeros(200, dtype=np.float64)
    best_y = np.zeros(200, dtype=np.float64)
    best_angle = np.zeros(200, dtype=np.float64)
    
    best_x[0] = solutions[1][0][0]
    best_y[0] = solutions[1][1][0]
    best_angle[0] = solutions[1][2][0]
    
    if verbose:
        print("\nStarting reverse backpacking from N=1...")
    
    for n in range(2, 201):
        current_x, current_y, current_angle = solutions[n]
        current_score = compute_score(current_x, current_y, current_angle, n)
        
        # Try extending best_layout with one tree from current solution
        # Find the tree in current that's not in best_layout[:n-1]
        # This is complex, so let's try a simpler approach:
        # Use the current solution's last tree added to best_layout
        
        test_x = best_x[:n].copy()
        test_y = best_y[:n].copy()
        test_angle = best_angle[:n].copy()
        
        # Try adding each tree from current solution as the n-th tree
        best_addition_score = float('inf')
        best_addition = None
        
        for i in range(n):
            test_x[n-1] = current_x[i]
            test_y[n-1] = current_y[i]
            test_angle[n-1] = current_angle[i]
            
            if not has_any_overlap(test_x, test_y, test_angle, n):
                score = compute_score(test_x, test_y, test_angle, n)
                if score < best_addition_score:
                    best_addition_score = score
                    best_addition = (current_x[i], current_y[i], current_angle[i])
        
        if best_addition is not None and best_addition_score < current_score - 1e-10:
            improvement = current_score - best_addition_score
            if verbose:
                print(f"  N={n}: IMPROVEMENT! {current_score:.10f} -> {best_addition_score:.10f} (gain: {improvement:.10f})")
            improvements.append((n, improvement))
            best_x[n-1], best_y[n-1], best_angle[n-1] = best_addition
            solutions[n] = (best_x[:n].copy(), best_y[:n].copy(), best_angle[:n].copy())
        else:
            # Current solution is better
            best_x[:n] = current_x
            best_y[:n] = current_y
            best_angle[:n] = current_angle
    
    return improvements

def main():
    # Load baseline
    baseline_path = "/home/nonroot/snapshots/santa-2025/21337353543/submission/submission.csv"
    print(f"Loading baseline from {baseline_path}")
    solutions = load_baseline(baseline_path)
    
    # Compute baseline total score
    baseline_total = sum(compute_score(*solutions[n], n) for n in range(1, 201))
    print(f"Baseline total score: {baseline_total:.6f}")
    
    # Warm up Numba
    print("\nWarming up Numba JIT...")
    _ = compute_bbox(solutions[5][0], solutions[5][1], solutions[5][2], 5)
    _ = has_any_overlap(solutions[5][0], solutions[5][1], solutions[5][2], 5)
    print("JIT compilation complete.")
    
    # Run backpacking
    print("\n" + "=" * 60)
    print("BACKPACKING TECHNIQUE")
    print("=" * 60)
    
    start_time = time.time()
    improvements = backpacking(solutions, verbose=True)
    elapsed = time.time() - start_time
    
    print(f"\nBackpacking completed in {elapsed:.2f}s")
    print(f"Improvements found: {len(improvements)}")
    
    if improvements:
        total_improvement = sum(imp for _, imp in improvements)
        print(f"Total improvement: {total_improvement:.10f}")
        print(f"Improved N values: {[n for n, _ in improvements]}")
    else:
        print("No improvements found.")
    
    # Run reverse backpacking
    print("\n" + "=" * 60)
    print("REVERSE BACKPACKING TECHNIQUE")
    print("=" * 60)
    
    # Reload baseline for reverse backpacking
    solutions = load_baseline(baseline_path)
    
    start_time = time.time()
    improvements_reverse = backpacking_reverse(solutions, verbose=True)
    elapsed = time.time() - start_time
    
    print(f"\nReverse backpacking completed in {elapsed:.2f}s")
    print(f"Improvements found: {len(improvements_reverse)}")
    
    if improvements_reverse:
        total_improvement = sum(imp for _, imp in improvements_reverse)
        print(f"Total improvement: {total_improvement:.10f}")
        print(f"Improved N values: {[n for n, _ in improvements_reverse]}")
    else:
        print("No improvements found.")
    
    # Compute final score
    final_total = sum(compute_score(*solutions[n], n) for n in range(1, 201))
    total_gain = baseline_total - final_total
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline score: {baseline_total:.6f}")
    print(f"Final score: {final_total:.6f}")
    print(f"Total improvement: {total_gain:.10f}")
    
    # Save results
    results = {
        'baseline_score': baseline_total,
        'final_score': final_total,
        'total_improvement': total_gain,
        'backpacking_improvements': len(improvements),
        'reverse_backpacking_improvements': len(improvements_reverse)
    }
    
    with open('/home/code/experiments/014_backpacking/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results.json")
    
    return total_gain

if __name__ == "__main__":
    total_gain = main()
