"""
Jostle/Compaction Algorithm Implementation

This is a fundamentally different approach from SA:
1. Jostle: randomly perturb a tree
2. Resolve overlaps: push overlapping trees apart
3. Compact: move all trees toward center
4. Accept if score improves

The key insight is that we ALLOW temporary overlaps and then resolve them,
which can help escape local optima that SA cannot.
"""

import numpy as np
import pandas as pd
from shapely import Polygon
from shapely.ops import unary_union
import math
import random
import time
import json

# Tree polygon vertices
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

def compute_bbox(trees):
    """Compute bounding box size for all trees."""
    all_x = []
    all_y = []
    
    for x, y, angle in trees:
        vertices = get_tree_vertices(x, y, angle)
        for vx, vy in vertices:
            all_x.append(vx)
            all_y.append(vy)
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    return max(max_x - min_x, max_y - min_y)

def compute_score(trees, n):
    """Compute score for N trees."""
    bbox = compute_bbox(trees)
    return bbox ** 2 / n

def validate_no_overlap(trees, tolerance=1e-9):
    """Check if any trees overlap."""
    n = len(trees)
    polygons = [get_tree_polygon(x, y, a) for x, y, a in trees]
    
    for i in range(n):
        for j in range(i + 1, n):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > tolerance:
                    return False
    return True

def find_overlapping_pairs(trees, tolerance=1e-9):
    """Find all pairs of overlapping trees."""
    n = len(trees)
    polygons = [get_tree_polygon(x, y, a) for x, y, a in trees]
    
    overlaps = []
    for i in range(n):
        for j in range(i + 1, n):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > tolerance:
                    overlaps.append((i, j, intersection.area))
    
    return overlaps

def resolve_overlaps(trees, max_iterations=100, step_size=0.01):
    """
    Resolve overlaps by pushing overlapping trees apart.
    """
    trees = [list(t) for t in trees]  # Make mutable
    
    for iteration in range(max_iterations):
        overlaps = find_overlapping_pairs(trees)
        
        if not overlaps:
            break
        
        # For each overlap, push trees apart
        for i, j, area in overlaps:
            # Direction from j to i
            dx = trees[i][0] - trees[j][0]
            dy = trees[i][1] - trees[j][1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < 1e-6:
                # Random direction if coincident
                angle = random.uniform(0, 2*math.pi)
                dx = math.cos(angle)
                dy = math.sin(angle)
                dist = 1.0
            
            # Push apart proportional to overlap
            push = step_size * (1 + area * 10)
            
            trees[i][0] += push * dx / dist
            trees[i][1] += push * dy / dist
            trees[j][0] -= push * dx / dist
            trees[j][1] -= push * dy / dist
    
    return [tuple(t) for t in trees]

def compact_toward_center(trees, step_size=0.01, max_iterations=50):
    """
    Compact configuration by moving trees toward center.
    """
    trees = [list(t) for t in trees]
    n = len(trees)
    
    best_score = compute_score([tuple(t) for t in trees], n)
    best_trees = [tuple(t) for t in trees]
    
    for iteration in range(max_iterations):
        # Compute centroid
        cx = sum(t[0] for t in trees) / n
        cy = sum(t[1] for t in trees) / n
        
        improved = False
        
        for i in range(n):
            # Direction toward center
            dx = cx - trees[i][0]
            dy = cy - trees[i][1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < 0.01:
                continue
            
            # Try to move toward center
            old_x, old_y = trees[i][0], trees[i][1]
            trees[i][0] += step_size * dx / dist
            trees[i][1] += step_size * dy / dist
            
            # Check for overlaps
            if not validate_no_overlap([tuple(t) for t in trees]):
                trees[i][0] = old_x
                trees[i][1] = old_y
            else:
                new_score = compute_score([tuple(t) for t in trees], n)
                if new_score < best_score:
                    best_score = new_score
                    best_trees = [tuple(t) for t in trees]
                    improved = True
        
        if not improved:
            break
    
    return best_trees, best_score

def jostle_optimize(trees, n_iterations=5000, jostle_scale=0.02, verbose=False):
    """
    Main jostle optimization algorithm.
    """
    n = len(trees)
    trees = list(trees)
    
    best_score = compute_score(trees, n)
    best_trees = trees.copy()
    
    no_improve_count = 0
    
    for iteration in range(n_iterations):
        # Select random tree to jostle
        i = random.randint(0, n - 1)
        
        # Jostle: small random displacement
        dx = random.uniform(-jostle_scale, jostle_scale)
        dy = random.uniform(-jostle_scale, jostle_scale)
        da = random.uniform(-5, 5)  # angle change in degrees
        
        # Apply jostle
        new_trees = trees.copy()
        x, y, a = new_trees[i]
        new_trees[i] = (x + dx, y + dy, (a + da) % 360)
        
        # Resolve any overlaps
        new_trees = resolve_overlaps(new_trees, max_iterations=20, step_size=0.005)
        
        # Check if valid
        if not validate_no_overlap(new_trees):
            no_improve_count += 1
            continue
        
        # Compact toward center
        new_trees, new_score = compact_toward_center(new_trees, step_size=0.005, max_iterations=10)
        
        # Check if improved
        if new_score < best_score - 1e-9:
            best_score = new_score
            best_trees = new_trees.copy()
            trees = new_trees
            no_improve_count = 0
            
            if verbose:
                print(f"  Iter {iteration}: NEW BEST {best_score:.6f}")
        else:
            no_improve_count += 1
        
        # Adaptive jostle scale
        if no_improve_count > 100:
            jostle_scale = min(jostle_scale * 1.1, 0.1)
            no_improve_count = 0
        
        if verbose and iteration % 500 == 0:
            print(f"  Iter {iteration}: score={best_score:.6f}, jostle_scale={jostle_scale:.4f}")
    
    return best_trees, best_score

def load_baseline_solution(csv_path):
    """Load baseline solution from CSV."""
    df = pd.read_csv(csv_path)
    
    solutions = {}
    for _, row in df.iterrows():
        id_parts = str(row['id']).split('_')
        n = int(id_parts[0])
        i = int(id_parts[1])
        
        x = float(str(row['x']).replace('s', ''))
        y = float(str(row['y']).replace('s', ''))
        deg = float(str(row['deg']).replace('s', ''))
        
        # Normalize angle
        deg = deg % 360.0
        if deg < 0:
            deg += 360.0
        
        if n not in solutions:
            solutions[n] = []
        solutions[n].append((x, y, deg))
    
    return solutions

def main():
    print("=" * 60)
    print("JOSTLE/COMPACTION ALGORITHM")
    print("=" * 60)
    
    # Load baseline
    baseline_path = "/home/submission/submission.csv"
    print(f"\nLoading baseline from {baseline_path}...")
    solutions = load_baseline_solution(baseline_path)
    
    # Test on small N values first
    test_ns = [10, 15, 20, 25, 30, 40, 50]
    
    results = {}
    improvements = 0
    total_improvement = 0.0
    
    print("\n" + "=" * 60)
    print("TESTING JOSTLE OPTIMIZATION")
    print("=" * 60)
    
    for n in test_ns:
        if n not in solutions:
            continue
        
        trees = solutions[n]
        baseline_score = compute_score(trees, n)
        print(f"\nN={n}: Baseline score = {baseline_score:.6f}")
        
        start_time = time.time()
        
        # Run jostle optimization
        opt_trees, opt_score = jostle_optimize(
            trees, 
            n_iterations=2000,
            jostle_scale=0.02,
            verbose=True
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
    with open('/home/code/experiments/034_jostle_compaction/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results, improvements, total_improvement

if __name__ == "__main__":
    results, improvements, total_improvement = main()
