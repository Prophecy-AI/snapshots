"""
Asymmetric Tessellation V2 - Start from baseline and try to improve

Instead of creating from scratch, analyze the baseline pattern and try to:
1. Find trees that could be moved to reduce bbox
2. Try rotating trees to better angles
3. Use the bimodal pattern insight
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
    
    return max(max_x - min_x, max_y - min_y), min_x, max_x, min_y, max_y

def compute_score(xs, ys, angles, n):
    bbox, _, _, _, _ = compute_bbox(xs, ys, angles)
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

def find_boundary_trees(xs, ys, angles):
    """Find trees that are on the bounding box boundary."""
    bbox, min_x, max_x, min_y, max_y = compute_bbox(xs, ys, angles)
    
    boundary_trees = []
    eps = 0.01
    
    for i in range(len(xs)):
        vertices = get_tree_vertices(xs[i], ys[i], angles[i])
        
        on_boundary = False
        for vx, vy in vertices:
            if abs(vx - min_x) < eps or abs(vx - max_x) < eps:
                on_boundary = True
                break
            if abs(vy - min_y) < eps or abs(vy - max_y) < eps:
                on_boundary = True
                break
        
        if on_boundary:
            boundary_trees.append(i)
    
    return boundary_trees

def try_rotate_tree(xs, ys, angles, tree_idx, new_angle):
    """Try rotating a single tree to a new angle."""
    new_angles = angles.copy()
    new_angles[tree_idx] = new_angle
    
    if check_overlap(xs, ys, new_angles):
        return None, float('inf')
    
    score = compute_score(xs, ys, new_angles, len(xs))
    return new_angles, score

def try_move_tree(xs, ys, angles, tree_idx, dx, dy):
    """Try moving a single tree."""
    new_xs = xs.copy()
    new_ys = ys.copy()
    new_xs[tree_idx] += dx
    new_ys[tree_idx] += dy
    
    if check_overlap(new_xs, new_ys, angles):
        return None, None, float('inf')
    
    score = compute_score(new_xs, new_ys, angles, len(xs))
    return new_xs, new_ys, score

def optimize_boundary_trees(xs, ys, angles, n, max_iterations=50, verbose=False):
    """
    Optimize by focusing on boundary trees - they determine the bbox.
    """
    xs = list(xs)
    ys = list(ys)
    angles = list(angles)
    
    best_score = compute_score(xs, ys, angles, n)
    best_xs, best_ys, best_angles = xs.copy(), ys.copy(), angles.copy()
    
    for iteration in range(max_iterations):
        improved = False
        
        # Find boundary trees
        boundary_trees = find_boundary_trees(xs, ys, angles)
        
        if verbose and iteration % 10 == 0:
            print(f"    Iter {iteration}: score={best_score:.6f}, boundary trees={len(boundary_trees)}")
        
        for tree_idx in boundary_trees:
            # Try different angles for this tree
            current_angle = angles[tree_idx]
            
            for delta_angle in [-10, -5, -2, -1, 1, 2, 5, 10]:
                new_angle = current_angle + delta_angle
                new_angles, score = try_rotate_tree(xs, ys, angles, tree_idx, new_angle)
                
                if new_angles is not None and score < best_score - 1e-9:
                    best_score = score
                    best_angles = new_angles.copy()
                    angles = new_angles
                    improved = True
                    break
            
            # Try moving this tree inward
            bbox, min_x, max_x, min_y, max_y = compute_bbox(xs, ys, angles)
            
            for dx in [-0.01, 0, 0.01]:
                for dy in [-0.01, 0, 0.01]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    new_xs, new_ys, score = try_move_tree(xs, ys, angles, tree_idx, dx, dy)
                    
                    if new_xs is not None and score < best_score - 1e-9:
                        best_score = score
                        best_xs = new_xs.copy()
                        best_ys = new_ys.copy()
                        xs = new_xs
                        ys = new_ys
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_xs, best_ys, best_angles, best_score

def enforce_bimodal_angles(xs, ys, angles, n, angle1=67, angle2=247, verbose=False):
    """
    Try to enforce the bimodal angle pattern.
    """
    xs = list(xs)
    ys = list(ys)
    angles = list(angles)
    
    best_score = compute_score(xs, ys, angles, n)
    best_angles = angles.copy()
    
    # For each tree, try to rotate to the nearest bimodal angle
    for i in range(len(angles)):
        current_angle = angles[i] % 360
        
        # Find nearest bimodal angle
        dist1 = min(abs(current_angle - angle1), abs(current_angle - angle1 - 360), abs(current_angle - angle1 + 360))
        dist2 = min(abs(current_angle - angle2), abs(current_angle - angle2 - 360), abs(current_angle - angle2 + 360))
        
        target_angle = angle1 if dist1 < dist2 else angle2
        
        # Try to rotate to target
        new_angles, score = try_rotate_tree(xs, ys, angles, i, target_angle)
        
        if new_angles is not None and score < best_score - 1e-9:
            best_score = score
            best_angles = new_angles.copy()
            angles = new_angles
            if verbose:
                print(f"    Tree {i}: {current_angle:.1f}° -> {target_angle}° (score: {score:.6f})")
    
    return xs, ys, best_angles, best_score

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
    print("ASYMMETRIC TESSELLATION V2 - IMPROVE BASELINE")
    print("=" * 60)
    
    baseline_path = "/home/submission/submission.csv"
    solutions = load_baseline_solution(baseline_path)
    
    # Test on various N values
    test_ns = [20, 30, 50, 100, 150, 200]
    
    results = {}
    improvements = 0
    total_improvement = 0.0
    
    print("\n" + "=" * 60)
    print("OPTIMIZING BOUNDARY TREES AND ANGLES")
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
        
        # Method 1: Optimize boundary trees
        print("  Optimizing boundary trees...")
        opt_xs, opt_ys, opt_angles, score1 = optimize_boundary_trees(
            xs, ys, angles, n, max_iterations=30, verbose=True
        )
        print(f"    After boundary optimization: {score1:.6f}")
        
        # Method 2: Try to enforce bimodal angles
        print("  Enforcing bimodal angles...")
        opt_xs, opt_ys, opt_angles, score2 = enforce_bimodal_angles(
            opt_xs, opt_ys, opt_angles, n, verbose=False
        )
        print(f"    After bimodal enforcement: {score2:.6f}")
        
        elapsed = time.time() - start_time
        
        final_score = min(score1, score2)
        improvement = baseline_score - final_score
        
        results[n] = {
            'baseline': baseline_score,
            'optimized': final_score,
            'improvement': improvement,
            'time': elapsed
        }
        
        if improvement > 1e-6:
            improvements += 1
            total_improvement += improvement
            print(f"  ✅ IMPROVED: {baseline_score:.6f} -> {final_score:.6f} ({improvement:+.6f})")
        else:
            print(f"  ❌ No improvement: {baseline_score:.6f} -> {final_score:.6f}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"N values tested: {len(test_ns)}")
    print(f"Improvements found: {improvements}")
    print(f"Total improvement: {total_improvement:.6f}")
    
    with open('/home/code/experiments/031_asymmetric_tessellation/v2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results, improvements, total_improvement

if __name__ == "__main__":
    results, improvements, total_improvement = main()
