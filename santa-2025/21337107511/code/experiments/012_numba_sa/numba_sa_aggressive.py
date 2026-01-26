"""
Aggressive Numba SA with larger perturbations and more iterations.

The previous attempt with small perturbations (0.002) found nothing.
Let's try larger perturbations to escape the local optimum.
"""

import numpy as np
from numba import njit
import pandas as pd
import time
import json

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
def check_tree_overlap(trees_x, trees_y, trees_angle, n, tree_idx):
    vx1, vy1 = get_tree_vertices(trees_x[tree_idx], trees_y[tree_idx], trees_angle[tree_idx])
    
    for i in range(n):
        if i == tree_idx:
            continue
        vx2, vy2 = get_tree_vertices(trees_x[i], trees_y[i], trees_angle[i])
        
        if max(vx1) < min(vx2) or max(vx2) < min(vx1):
            continue
        if max(vy1) < min(vy2) or max(vy2) < min(vy1):
            continue
        
        if polygons_overlap(vx1, vy1, vx2, vy2, 15, 15):
            return True
    return False

@njit
def aggressive_sa(trees_x, trees_y, trees_angle, n, n_iterations=50000, seed=42):
    """
    Aggressive SA with larger perturbations and higher temperature.
    """
    np.random.seed(seed)
    
    current_x = trees_x.copy()
    current_y = trees_y.copy()
    current_angle = trees_angle.copy()
    
    best_x = trees_x.copy()
    best_y = trees_y.copy()
    best_angle = trees_angle.copy()
    
    current_side = compute_bbox(current_x, current_y, current_angle, n)
    best_side = current_side
    
    # Higher starting temperature
    temp_start = 5.0
    temp_end = 0.00001
    
    accepted = 0
    improved = 0
    
    for iteration in range(n_iterations):
        temp = temp_start * (temp_end / temp_start) ** (iteration / n_iterations)
        
        tree_idx = np.random.randint(0, n)
        
        old_x = current_x[tree_idx]
        old_y = current_y[tree_idx]
        old_angle = current_angle[tree_idx]
        
        # Larger perturbations that scale with temperature
        scale = max(temp * 0.1, 0.001)  # 0.5 at start, 0.001 at end
        dx = (np.random.random() - 0.5) * scale * 2
        dy = (np.random.random() - 0.5) * scale * 2
        dangle = (np.random.random() - 0.5) * scale * 100  # Up to 50 degrees at start
        
        current_x[tree_idx] += dx
        current_y[tree_idx] += dy
        current_angle[tree_idx] += dangle
        
        if check_tree_overlap(current_x, current_y, current_angle, n, tree_idx):
            current_x[tree_idx] = old_x
            current_y[tree_idx] = old_y
            current_angle[tree_idx] = old_angle
            continue
        
        new_side = compute_bbox(current_x, current_y, current_angle, n)
        delta = new_side - current_side
        
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
            current_x[tree_idx] = old_x
            current_y[tree_idx] = old_y
            current_angle[tree_idx] = old_angle
    
    return best_side, best_x, best_y, best_angle, accepted, improved

def load_baseline(csv_path):
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

def test_aggressive_sa(baseline_solutions, test_ns=[10, 20, 30], n_iterations=50000, n_rounds=3):
    print("=" * 60)
    print("TESTING AGGRESSIVE NUMBA SA")
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
        
        for round_idx in range(n_rounds):
            start_time = time.time()
            
            new_side, new_x, new_y, new_angle, accepted, improved = aggressive_sa(
                best_x.copy(), best_y.copy(), best_angle.copy(), n,
                n_iterations=n_iterations, seed=round_idx * 1000
            )
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            if new_side < best_side:
                improvement = best_side - new_side
                best_side = new_side
                best_x = new_x
                best_y = new_y
                best_angle = new_angle
                print(f"  Round {round_idx + 1}: side={new_side:.8f}, improvement={improvement:.10f}, accepted={accepted}, time={elapsed:.2f}s")
            else:
                print(f"  Round {round_idx + 1}: no improvement, accepted={accepted}, time={elapsed:.2f}s")
        
        best_score = (best_side ** 2) / n
        improvement = baseline_score - best_score
        
        results[n] = {
            'baseline_side': float(baseline_side),
            'baseline_score': float(baseline_score),
            'best_side': float(best_side),
            'best_score': float(best_score),
            'improvement': float(improvement),
            'total_time': total_time
        }
        
        print(f"\n  RESULTS for N={n}:")
        print(f"    Baseline: side={baseline_side:.8f}, score={baseline_score:.8f}")
        print(f"    Best: side={best_side:.8f}, score={best_score:.8f}")
        print(f"    Improvement: {improvement:.10f}")
        
        if improvement > 1e-10:
            print(f"    ✅ IMPROVED!")
        else:
            print(f"    ❌ No improvement")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_improvement = sum(r['improvement'] for r in results.values())
    improved_count = sum(1 for r in results.values() if r['improvement'] > 1e-10)
    
    print(f"Total improvement: {total_improvement:.10f}")
    print(f"N values improved: {improved_count}/{len(test_ns)}")
    
    return results

if __name__ == "__main__":
    baseline_path = "/home/nonroot/snapshots/santa-2025/21337353543/submission/submission.csv"
    print(f"Loading baseline from {baseline_path}")
    baseline_solutions = load_baseline(baseline_path)
    
    # Warm up
    print("\nWarming up Numba JIT...")
    trees_x, trees_y, trees_angle = baseline_solutions[5]
    _ = aggressive_sa(trees_x.copy(), trees_y.copy(), trees_angle.copy(), 5, n_iterations=100, seed=0)
    print("JIT compilation complete.")
    
    # Test aggressive SA
    test_results = test_aggressive_sa(baseline_solutions, test_ns=[10, 20, 30], n_iterations=50000, n_rounds=3)
    
    with open('/home/code/experiments/012_numba_sa/aggressive_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\nResults saved to aggressive_results.json")
