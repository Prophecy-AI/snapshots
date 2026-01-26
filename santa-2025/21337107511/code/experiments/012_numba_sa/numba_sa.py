"""
Numba-Accelerated Simulated Annealing

Key improvements over previous attempts:
1. Numba JIT compilation for 100x speedup
2. Correct SA parameters (15000 iterations, proper temp schedule)
3. Proper Metropolis acceptance criterion
4. Small perturbation scale (0.002 instead of 0.1)

The goal is to escape the local optimum that all previous approaches were stuck at.
"""

import numpy as np
from numba import njit, prange
import pandas as pd
import time
import json
from decimal import Decimal, getcontext

getcontext().prec = 30

# Tree polygon vertices (15 vertices)
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125], dtype=np.float64)
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5], dtype=np.float64)

@njit
def rotate_vertices(tx, ty, angle_deg):
    """Rotate tree vertices by angle (degrees)."""
    angle_rad = angle_deg * np.pi / 180.0
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rx = tx * cos_a - ty * sin_a
    ry = tx * sin_a + ty * cos_a
    return rx, ry

@njit
def get_tree_vertices(x, y, angle):
    """Get all 15 vertices of a tree at position (x, y) with rotation angle."""
    rx, ry = rotate_vertices(TX, TY, angle)
    vx = rx + x
    vy = ry + y
    return vx, vy

@njit
def compute_bbox(trees_x, trees_y, trees_angle, n):
    """Compute bounding box side length for n trees."""
    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf
    
    for i in range(n):
        vx, vy = get_tree_vertices(trees_x[i], trees_y[i], trees_angle[i])
        for j in range(15):
            if vx[j] < min_x:
                min_x = vx[j]
            if vx[j] > max_x:
                max_x = vx[j]
            if vy[j] < min_y:
                min_y = vy[j]
            if vy[j] > max_y:
                max_y = vy[j]
    
    return max(max_x - min_x, max_y - min_y)

@njit
def point_in_polygon(px, py, poly_x, poly_y, n_vertices):
    """Check if point (px, py) is inside polygon using ray casting."""
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
    """Check if line segment (x1,y1)-(x2,y2) intersects with (x3,y3)-(x4,y4)."""
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
    """Check if two polygons overlap (excluding touching)."""
    # Check if any vertex of polygon 1 is inside polygon 2
    for i in range(n1):
        if point_in_polygon(vx1[i], vy1[i], vx2, vy2, n2):
            return True
    
    # Check if any vertex of polygon 2 is inside polygon 1
    for i in range(n2):
        if point_in_polygon(vx2[i], vy2[i], vx1, vy1, n1):
            return True
    
    # Check if any edges intersect
    for i in range(n1):
        i_next = (i + 1) % n1
        for j in range(n2):
            j_next = (j + 1) % n2
            if segments_intersect(vx1[i], vy1[i], vx1[i_next], vy1[i_next],
                                  vx2[j], vy2[j], vx2[j_next], vy2[j_next]):
                return True
    
    return False

@njit
def check_tree_overlap(trees_x, trees_y, trees_angle, n, tree_idx):
    """Check if tree at tree_idx overlaps with any other tree."""
    vx1, vy1 = get_tree_vertices(trees_x[tree_idx], trees_y[tree_idx], trees_angle[tree_idx])
    
    for i in range(n):
        if i == tree_idx:
            continue
        
        # Quick bounding box check first
        vx2, vy2 = get_tree_vertices(trees_x[i], trees_y[i], trees_angle[i])
        
        # Bounding box check
        if max(vx1) < min(vx2) or max(vx2) < min(vx1):
            continue
        if max(vy1) < min(vy2) or max(vy2) < min(vy1):
            continue
        
        # Detailed polygon overlap check
        if polygons_overlap(vx1, vy1, vx2, vy2, 15, 15):
            return True
    
    return False

@njit
def fast_sa(trees_x, trees_y, trees_angle, n, n_iterations=15000, seed=42):
    """
    Numba-accelerated simulated annealing.
    
    Key parameters:
    - n_iterations: 15000 (not 1000)
    - temp: 1.0 → 0.000005 (exponential decay)
    - perturbation scale: 0.002 (not 0.1)
    - Metropolis acceptance: accept if delta < 0 OR random() < exp(-delta/temp)
    """
    np.random.seed(seed)
    
    # Make copies
    current_x = trees_x.copy()
    current_y = trees_y.copy()
    current_angle = trees_angle.copy()
    
    best_x = trees_x.copy()
    best_y = trees_y.copy()
    best_angle = trees_angle.copy()
    
    current_side = compute_bbox(current_x, current_y, current_angle, n)
    best_side = current_side
    
    # Temperature schedule
    temp_start = 1.0
    temp_end = 0.000005
    
    accepted = 0
    improved = 0
    
    for iteration in range(n_iterations):
        # Exponential temperature decay
        temp = temp_start * (temp_end / temp_start) ** (iteration / n_iterations)
        
        # Random tree selection
        tree_idx = np.random.randint(0, n)
        
        # Save old values
        old_x = current_x[tree_idx]
        old_y = current_y[tree_idx]
        old_angle = current_angle[tree_idx]
        
        # Small perturbation (scale = 0.002)
        dx = (np.random.random() - 0.5) * 0.004  # ±0.002
        dy = (np.random.random() - 0.5) * 0.004
        dangle = (np.random.random() - 0.5) * 4.0  # ±2 degrees
        
        # Apply perturbation
        current_x[tree_idx] += dx
        current_y[tree_idx] += dy
        current_angle[tree_idx] += dangle
        
        # Check for overlaps
        if check_tree_overlap(current_x, current_y, current_angle, n, tree_idx):
            # Reject - restore
            current_x[tree_idx] = old_x
            current_y[tree_idx] = old_y
            current_angle[tree_idx] = old_angle
            continue
        
        # Compute new bounding box
        new_side = compute_bbox(current_x, current_y, current_angle, n)
        delta = new_side - current_side
        
        # Metropolis acceptance criterion
        if delta < 0 or np.random.random() < np.exp(-delta / temp):
            current_side = new_side
            accepted += 1
            
            if new_side < best_side:
                best_side = new_side
                best_x[:] = current_x
                best_y[:] = current_y
                best_angle[:] = current_angle
                improved += 1
        else:
            # Reject - restore
            current_x[tree_idx] = old_x
            current_y[tree_idx] = old_y
            current_angle[tree_idx] = old_angle
    
    return best_side, best_x, best_y, best_angle, accepted, improved

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

def test_numba_sa(baseline_solutions, test_ns=[10, 20, 30], n_iterations=15000, n_rounds=5):
    """Test Numba SA on small N values."""
    print("=" * 60)
    print("TESTING NUMBA-ACCELERATED SA")
    print(f"Parameters: iterations={n_iterations}, rounds={n_rounds}")
    print("=" * 60)
    
    results = {}
    
    for n in test_ns:
        print(f"\nN={n}:")
        trees_x, trees_y, trees_angle = baseline_solutions[n]
        baseline_side = compute_bbox(trees_x, trees_y, trees_angle, n)
        baseline_score = (baseline_side ** 2) / n
        
        best_side = baseline_side
        best_x = trees_x.copy()
        best_y = trees_y.copy()
        best_angle = trees_angle.copy()
        
        total_time = 0
        total_accepted = 0
        total_improved = 0
        
        for round_idx in range(n_rounds):
            start_time = time.time()
            
            # Run SA from current best
            new_side, new_x, new_y, new_angle, accepted, improved = fast_sa(
                best_x.copy(), best_y.copy(), best_angle.copy(), n,
                n_iterations=n_iterations, seed=round_idx * 1000
            )
            
            elapsed = time.time() - start_time
            total_time += elapsed
            total_accepted += accepted
            total_improved += improved
            
            if new_side < best_side:
                improvement = best_side - new_side
                best_side = new_side
                best_x = new_x
                best_y = new_y
                best_angle = new_angle
                print(f"  Round {round_idx + 1}: side={new_side:.8f}, improvement={improvement:.10f}, time={elapsed:.2f}s")
            else:
                print(f"  Round {round_idx + 1}: no improvement, time={elapsed:.2f}s")
        
        best_score = (best_side ** 2) / n
        improvement = baseline_score - best_score
        
        results[n] = {
            'baseline_side': float(baseline_side),
            'baseline_score': float(baseline_score),
            'best_side': float(best_side),
            'best_score': float(best_score),
            'improvement': float(improvement),
            'total_time': total_time,
            'total_accepted': total_accepted,
            'total_improved': total_improved
        }
        
        print(f"\n  RESULTS for N={n}:")
        print(f"    Baseline: side={baseline_side:.8f}, score={baseline_score:.8f}")
        print(f"    Best: side={best_side:.8f}, score={best_score:.8f}")
        print(f"    Improvement: {improvement:.10f} ({improvement/baseline_score*100:.6f}%)")
        print(f"    Total time: {total_time:.2f}s")
        
        if improvement > 1e-10:
            print(f"    ✅ IMPROVED!")
        else:
            print(f"    ❌ No improvement")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_improvement = sum(r['improvement'] for r in results.values())
    improved_count = sum(1 for r in results.values() if r['improvement'] > 1e-10)
    
    print(f"Total improvement: {total_improvement:.10f}")
    print(f"N values improved: {improved_count}/{len(test_ns)}")
    
    return results

if __name__ == "__main__":
    # Load baseline
    baseline_path = "/home/nonroot/snapshots/santa-2025/21337353543/submission/submission.csv"
    print(f"Loading baseline from {baseline_path}")
    baseline_solutions = load_baseline(baseline_path)
    
    # Warm up Numba JIT
    print("\nWarming up Numba JIT...")
    trees_x, trees_y, trees_angle = baseline_solutions[5]
    _ = fast_sa(trees_x.copy(), trees_y.copy(), trees_angle.copy(), 5, n_iterations=100, seed=0)
    print("JIT compilation complete.")
    
    # Test on small N values
    test_results = test_numba_sa(baseline_solutions, test_ns=[10, 20, 30], n_iterations=15000, n_rounds=5)
    
    # Save results
    with open('/home/code/experiments/012_numba_sa/test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\nResults saved to test_results.json")
