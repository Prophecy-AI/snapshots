"""
Asymmetric Tessellation Optimization

Key insight from analysis:
- Large N solutions use bimodal angle distribution (~67° and ~247°)
- Trees at these angles interlock better than uniform angles
- This creates higher packing density

Approach:
1. Create a checkerboard pattern with alternating angles
2. Optimize the angle pair
3. Optimize the spacing
4. Compare to baseline
"""

import numpy as np
import pandas as pd
from shapely import Polygon
import math
import time
import json

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def get_tree_vertices(x, y, angle_deg):
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
    vertices = get_tree_vertices(x, y, angle_deg)
    return Polygon(vertices)

def compute_bbox(xs, ys, angles):
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
    bbox = compute_bbox(xs, ys, angles)
    return bbox ** 2 / n

def check_overlap(xs, ys, angles):
    n = len(xs)
    polygons = [get_tree_polygon(x, y, a) for x, y, a in zip(xs, ys, angles)]
    
    for i in range(n):
        for j in range(i + 1, n):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > 1e-10:
                    return True
    return False

def create_checkerboard_tessellation(n, angle1, angle2, spacing_x, spacing_y, offset_x=0, offset_y=0):
    """
    Create a checkerboard tessellation with alternating angles.
    """
    side = int(math.ceil(math.sqrt(n)))
    
    xs, ys, angles = [], [], []
    count = 0
    
    for row in range(side + 2):  # Extra rows to ensure we have enough
        for col in range(side + 2):
            if count >= n:
                break
            
            x = col * spacing_x
            y = row * spacing_y
            
            # Offset every other row
            if row % 2 == 1:
                x += offset_x
            
            # Alternate angles in checkerboard pattern
            if (row + col) % 2 == 0:
                angle = angle1
            else:
                angle = angle2
            
            xs.append(x)
            ys.append(y)
            angles.append(angle)
            count += 1
        
        if count >= n:
            break
    
    # Center the configuration
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    xs = [x - cx for x in xs]
    ys = [y - cy for y in ys]
    
    return xs[:n], ys[:n], angles[:n]

def compact_tessellation(xs, ys, angles, n, max_iterations=100):
    """
    Compact the tessellation by moving trees toward the center.
    """
    xs = list(xs)
    ys = list(ys)
    
    best_score = compute_score(xs, ys, angles, n)
    best_xs, best_ys = xs.copy(), ys.copy()
    
    for iteration in range(max_iterations):
        improved = False
        
        # Find centroid
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        
        for i in range(len(xs)):
            # Try to move toward center
            dx = cx - xs[i]
            dy = cy - ys[i]
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < 0.01:
                continue
            
            for step in [0.05, 0.02, 0.01]:
                new_x = xs[i] + step * dx / dist
                new_y = ys[i] + step * dy / dist
                
                old_x, old_y = xs[i], ys[i]
                xs[i], ys[i] = new_x, new_y
                
                if check_overlap(xs, ys, angles):
                    xs[i], ys[i] = old_x, old_y
                else:
                    new_score = compute_score(xs, ys, angles, n)
                    if new_score < best_score:
                        best_score = new_score
                        best_xs = xs.copy()
                        best_ys = ys.copy()
                        improved = True
                    break
        
        if not improved:
            break
    
    return best_xs, best_ys, angles, best_score

def optimize_tessellation_params(n, angle_range=(60, 80), spacing_range=(0.45, 0.65), verbose=True):
    """
    Optimize the tessellation parameters for a given N.
    """
    best_score = float('inf')
    best_params = None
    best_config = None
    
    # Grid search over parameters
    for angle1 in range(angle_range[0], angle_range[1] + 1, 2):
        angle2 = angle1 + 180
        
        for spacing in np.arange(spacing_range[0], spacing_range[1] + 0.01, 0.02):
            for offset in [0, spacing/4, spacing/2]:
                xs, ys, angles = create_checkerboard_tessellation(
                    n, angle1, angle2, spacing, spacing, offset, 0
                )
                
                # Check for overlaps
                if check_overlap(xs, ys, angles):
                    continue
                
                score = compute_score(xs, ys, angles, n)
                
                if score < best_score:
                    best_score = score
                    best_params = (angle1, spacing, offset)
                    best_config = (xs, ys, angles)
    
    if best_config is None:
        if verbose:
            print(f"  No valid configuration found for N={n}")
        return None, None, None, float('inf')
    
    # Compact the best configuration
    xs, ys, angles = best_config
    xs, ys, angles, score = compact_tessellation(xs, ys, angles, n, max_iterations=50)
    
    if verbose:
        print(f"  Best params: angle1={best_params[0]}°, spacing={best_params[1]:.2f}, offset={best_params[2]:.2f}")
        print(f"  Score: {score:.6f}")
    
    return xs, ys, angles, score

def load_baseline_solution(csv_path):
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
    print("ASYMMETRIC TESSELLATION OPTIMIZATION")
    print("=" * 60)
    
    baseline_path = "/home/submission/submission.csv"
    solutions = load_baseline_solution(baseline_path)
    
    # Test on large N values where the bimodal pattern is strongest
    test_ns = [50, 100, 150, 200]
    
    results = {}
    improvements = 0
    total_improvement = 0.0
    
    print("\n" + "=" * 60)
    print("OPTIMIZING ASYMMETRIC TESSELLATION")
    print("=" * 60)
    
    for n in test_ns:
        if n not in solutions:
            continue
            
        trees = solutions[n]
        baseline_xs = [t[0] for t in trees]
        baseline_ys = [t[1] for t in trees]
        baseline_angles = [t[2] for t in trees]
        
        baseline_score = compute_score(baseline_xs, baseline_ys, baseline_angles, n)
        print(f"\nN={n}: Baseline score = {baseline_score:.6f}")
        
        # Optimize tessellation
        start_time = time.time()
        xs, ys, angles, score = optimize_tessellation_params(n, verbose=True)
        elapsed = time.time() - start_time
        
        if xs is None:
            print(f"  ❌ No valid tessellation found")
            continue
        
        improvement = baseline_score - score
        results[n] = {
            'baseline': baseline_score,
            'tessellation': score,
            'improvement': improvement,
            'time': elapsed
        }
        
        if improvement > 1e-6:
            improvements += 1
            total_improvement += improvement
            print(f"  ✅ IMPROVED: {baseline_score:.6f} -> {score:.6f} ({improvement:+.6f})")
        else:
            print(f"  ❌ No improvement: {baseline_score:.6f} -> {score:.6f}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"N values tested: {len(test_ns)}")
    print(f"Improvements found: {improvements}")
    print(f"Total improvement: {total_improvement:.6f}")
    
    with open('/home/code/experiments/031_asymmetric_tessellation/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results, improvements, total_improvement

if __name__ == "__main__":
    results, improvements, total_improvement = main()
