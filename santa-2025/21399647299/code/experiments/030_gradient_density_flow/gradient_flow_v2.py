"""
Gradient-Based Density Flow Optimization V2

Enhanced version with:
1. Angle optimization (not just position)
2. Larger perturbations to escape local optima
3. Multi-start from different initial configurations
4. Simulated annealing-like temperature schedule
"""

import numpy as np
import pandas as pd
from shapely import Polygon
from shapely.affinity import rotate, translate
import math
import time
import json
from collections import defaultdict

# Tree polygon vertices (centered at origin)
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def get_tree_vertices(x, y, angle_deg):
    """Get the vertices of a tree at position (x, y) with rotation angle_deg."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    vertices = []
    for tx, ty in zip(TX, TY):
        rx = tx * cos_a - ty * sin_a
        ry = tx * sin_a + ty * cos_a
        vertices.append((rx + x, ry + y))
    
    return vertices

def get_tree_polygon(x, y, angle_deg):
    """Get Shapely polygon for a tree."""
    vertices = get_tree_vertices(x, y, angle_deg)
    return Polygon(vertices)

def compute_bbox(xs, ys, angles):
    """Compute bounding box size for all trees."""
    all_x = []
    all_y = []
    
    for x, y, angle in zip(xs, ys, angles):
        vertices = get_tree_vertices(x, y, angle)
        for vx, vy in vertices:
            all_x.append(vx)
            all_y.append(vy)
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    return max(max_x - min_x, max_y - min_y)

def compute_score(xs, ys, angles, n):
    """Compute score for N trees."""
    bbox = compute_bbox(xs, ys, angles)
    return bbox ** 2 / n

def check_overlap_fast(xs, ys, angles):
    """Fast overlap check using Shapely."""
    n = len(xs)
    polygons = [get_tree_polygon(x, y, a) for x, y, a in zip(xs, ys, angles)]
    
    for i in range(n):
        for j in range(i + 1, n):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > 1e-10:
                    return True
    return False

def compute_overlap_penalty(xs, ys, angles):
    """Compute total overlap area as penalty."""
    n = len(xs)
    polygons = [get_tree_polygon(x, y, a) for x, y, a in zip(xs, ys, angles)]
    
    total_overlap = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if polygons[i].intersects(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                total_overlap += intersection.area
    
    return total_overlap

def compute_numerical_gradient(xs, ys, angles, n, eps=1e-4):
    """
    Compute numerical gradient of score with respect to positions and angles.
    """
    base_score = compute_score(xs, ys, angles, n)
    
    grad_x = []
    grad_y = []
    grad_a = []
    
    for i in range(len(xs)):
        # Gradient w.r.t. x
        xs_plus = xs.copy()
        xs_plus[i] += eps
        score_plus = compute_score(xs_plus, ys, angles, n)
        grad_x.append((score_plus - base_score) / eps)
        
        # Gradient w.r.t. y
        ys_plus = ys.copy()
        ys_plus[i] += eps
        score_plus = compute_score(xs, ys_plus, angles, n)
        grad_y.append((score_plus - base_score) / eps)
        
        # Gradient w.r.t. angle
        angles_plus = angles.copy()
        angles_plus[i] += eps * 10  # Larger step for angles
        score_plus = compute_score(xs, ys, angles_plus, n)
        grad_a.append((score_plus - base_score) / (eps * 10))
    
    return grad_x, grad_y, grad_a

def gradient_descent_with_angles(xs, ys, angles, n, max_iterations=2000, 
                                   initial_step=0.005, verbose=False):
    """
    Gradient descent optimizing positions AND angles.
    """
    xs = list(xs)
    ys = list(ys)
    angles = list(angles)
    
    best_score = compute_score(xs, ys, angles, n)
    best_xs, best_ys, best_angles = xs.copy(), ys.copy(), angles.copy()
    
    step_size = initial_step
    no_improve_count = 0
    
    for iteration in range(max_iterations):
        # Compute numerical gradient
        grad_x, grad_y, grad_a = compute_numerical_gradient(xs, ys, angles, n)
        
        # Take gradient step (negative direction to minimize)
        new_xs = [xs[i] - step_size * grad_x[i] for i in range(len(xs))]
        new_ys = [ys[i] - step_size * grad_y[i] for i in range(len(ys))]
        new_angles = [angles[i] - step_size * 10 * grad_a[i] for i in range(len(angles))]
        
        # Check for overlaps
        if check_overlap_fast(new_xs, new_ys, new_angles):
            step_size *= 0.5
            no_improve_count += 1
            if step_size < 1e-8:
                break
            continue
        
        new_score = compute_score(new_xs, new_ys, new_angles, n)
        
        if new_score < best_score - 1e-9:
            best_score = new_score
            best_xs = new_xs.copy()
            best_ys = new_ys.copy()
            best_angles = new_angles.copy()
            xs, ys, angles = new_xs, new_ys, new_angles
            no_improve_count = 0
            step_size = min(step_size * 1.2, initial_step)
        else:
            no_improve_count += 1
            step_size *= 0.9
        
        if no_improve_count > 50:
            break
        
        if verbose and iteration % 200 == 0:
            print(f"    Iter {iteration}: score={best_score:.6f}")
    
    return best_xs, best_ys, best_angles, best_score

def multi_start_optimization(xs, ys, angles, n, num_restarts=10, verbose=True):
    """
    Multi-start optimization with random perturbations.
    """
    best_score = compute_score(xs, ys, angles, n)
    best_xs, best_ys, best_angles = xs.copy(), ys.copy(), angles.copy()
    
    for restart in range(num_restarts):
        if restart == 0:
            # First restart: use original
            curr_xs, curr_ys, curr_angles = xs.copy(), ys.copy(), angles.copy()
        else:
            # Random perturbation
            perturbation_scale = 0.05 * (1 + restart / num_restarts)
            curr_xs = [x + np.random.uniform(-perturbation_scale, perturbation_scale) for x in best_xs]
            curr_ys = [y + np.random.uniform(-perturbation_scale, perturbation_scale) for y in best_ys]
            curr_angles = [a + np.random.uniform(-10, 10) for a in best_angles]
            
            # Skip if overlapping
            if check_overlap_fast(curr_xs, curr_ys, curr_angles):
                continue
        
        # Run gradient descent
        opt_xs, opt_ys, opt_angles, opt_score = gradient_descent_with_angles(
            curr_xs, curr_ys, curr_angles, n,
            max_iterations=1000,
            initial_step=0.01,
            verbose=False
        )
        
        if opt_score < best_score - 1e-9:
            best_score = opt_score
            best_xs = opt_xs
            best_ys = opt_ys
            best_angles = opt_angles
            if verbose:
                print(f"    Restart {restart}: NEW BEST {best_score:.6f}")
    
    return best_xs, best_ys, best_angles, best_score

def coordinate_descent(xs, ys, angles, n, max_iterations=500, verbose=False):
    """
    Coordinate descent: optimize one tree at a time.
    """
    xs = list(xs)
    ys = list(ys)
    angles = list(angles)
    
    best_score = compute_score(xs, ys, angles, n)
    
    for iteration in range(max_iterations):
        improved = False
        
        for i in range(len(xs)):
            # Try small perturbations for tree i
            for dx in [-0.01, 0, 0.01]:
                for dy in [-0.01, 0, 0.01]:
                    for da in [-5, 0, 5]:
                        if dx == 0 and dy == 0 and da == 0:
                            continue
                        
                        new_xs = xs.copy()
                        new_ys = ys.copy()
                        new_angles = angles.copy()
                        
                        new_xs[i] += dx
                        new_ys[i] += dy
                        new_angles[i] += da
                        
                        if check_overlap_fast(new_xs, new_ys, new_angles):
                            continue
                        
                        new_score = compute_score(new_xs, new_ys, new_angles, n)
                        
                        if new_score < best_score - 1e-9:
                            best_score = new_score
                            xs, ys, angles = new_xs, new_ys, new_angles
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
        
        if verbose and iteration % 50 == 0:
            print(f"    Coord descent iter {iteration}: {best_score:.6f}")
    
    return xs, ys, angles, best_score

def load_baseline_solution(csv_path):
    """Load baseline solution from CSV."""
    df = pd.read_csv(csv_path)
    
    solutions = {}
    for _, row in df.iterrows():
        id_parts = row['id'].split('_')
        n = int(id_parts[0])
        i = int(id_parts[1])
        
        x = float(str(row['x']).replace('s', ''))
        y = float(str(row['y']).replace('s', ''))
        deg = float(str(row['deg']).replace('s', ''))
        
        if n not in solutions:
            solutions[n] = []
        solutions[n].append((x, y, deg))
    
    return solutions

def main():
    print("=" * 60)
    print("GRADIENT-BASED DENSITY FLOW V2 - WITH ANGLE OPTIMIZATION")
    print("=" * 60)
    
    # Load baseline
    baseline_path = "/home/submission/submission.csv"
    print(f"\nLoading baseline from {baseline_path}...")
    solutions = load_baseline_solution(baseline_path)
    
    # Test on small N first
    test_ns = [5, 10, 15, 20, 25, 30]
    
    results = {}
    improvements = 0
    total_improvement = 0.0
    
    print("\n" + "=" * 60)
    print("TESTING GRADIENT DESCENT + COORDINATE DESCENT")
    print("=" * 60)
    
    for n in test_ns:
        if n not in solutions:
            continue
            
        trees = solutions[n]
        xs = [t[0] for t in trees]
        ys = [t[1] for t in trees]
        angles = [t[2] for t in trees]
        
        baseline_score = compute_score(xs, ys, angles, n)
        print(f"\nN={n}: Baseline score = {baseline_score:.6f}")
        
        start_time = time.time()
        
        # Method 1: Multi-start gradient descent
        print("  Running multi-start gradient descent...")
        opt_xs, opt_ys, opt_angles, opt_score = multi_start_optimization(
            xs, ys, angles, n, num_restarts=5, verbose=True
        )
        
        # Method 2: Coordinate descent refinement
        print("  Running coordinate descent refinement...")
        opt_xs, opt_ys, opt_angles, opt_score = coordinate_descent(
            opt_xs, opt_ys, opt_angles, n, max_iterations=200, verbose=True
        )
        
        elapsed = time.time() - start_time
        
        improvement = baseline_score - opt_score
        results[n] = {
            'baseline': baseline_score,
            'optimized': opt_score,
            'improvement': improvement,
            'time': elapsed
        }
        
        if improvement > 1e-6:
            improvements += 1
            total_improvement += improvement
            print(f"  ✅ IMPROVED: {baseline_score:.6f} -> {opt_score:.6f} ({improvement:+.6f})")
        else:
            print(f"  ❌ No improvement: {baseline_score:.6f} -> {opt_score:.6f}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"N values tested: {len(test_ns)}")
    print(f"Improvements found: {improvements}")
    print(f"Total improvement: {total_improvement:.6f}")
    
    # Save results
    with open('/home/code/experiments/030_gradient_density_flow/v2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results, improvements, total_improvement

if __name__ == "__main__":
    results, improvements, total_improvement = main()
