"""
Gradient-Based Density Flow Optimization

This is a fundamentally different approach from SA:
- SA uses random perturbations
- Gradient descent uses DIRECTED movement based on the objective function

Algorithm:
1. Compute the gradient of bounding box with respect to each tree position
2. Move trees in the direction that reduces the bounding box
3. Add "density flow" - trees are attracted toward the centroid
4. Add "boundary tension" - trees on the boundary are pushed inward
5. Use adaptive step sizes and momentum to escape local optima
"""

import numpy as np
import pandas as pd
from shapely import Polygon
from shapely.affinity import rotate, translate
import math
import time
import json

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
        # Rotate
        rx = tx * cos_a - ty * sin_a
        ry = tx * sin_a + ty * cos_a
        # Translate
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

def check_overlap(xs, ys, angles):
    """Check if any trees overlap."""
    n = len(xs)
    polygons = [get_tree_polygon(x, y, a) for x, y, a in zip(xs, ys, angles)]
    
    for i in range(n):
        for j in range(i + 1, n):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                # Check if it's a real overlap (not just touching)
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > 1e-10:
                    return True
    return False

def compute_bbox_gradient(xs, ys, angles):
    """
    Compute the gradient of the bounding box with respect to tree positions.
    
    Returns: list of (dx, dy) for each tree - the direction to move to reduce bbox
    """
    n = len(xs)
    
    # Get all vertices
    all_vertices = []
    for i in range(n):
        verts = get_tree_vertices(xs[i], ys[i], angles[i])
        all_vertices.append(verts)
    
    # Find min/max x/y
    all_x = [v[0] for verts in all_vertices for v in verts]
    all_y = [v[1] for verts in all_vertices for v in verts]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Gradient: trees on boundary should move inward
    # Weight by how much they contribute to the limiting dimension
    gradients = []
    eps = 1e-6
    
    for i in range(n):
        dx, dy = 0.0, 0.0
        
        for vx, vy in all_vertices[i]:
            # If this vertex is on the boundary, push it inward
            if abs(vx - min_x) < eps:
                dx += 1.0  # Push right
            if abs(vx - max_x) < eps:
                dx -= 1.0  # Push left
            if abs(vy - min_y) < eps:
                dy += 1.0  # Push up
            if abs(vy - max_y) < eps:
                dy -= 1.0  # Push down
        
        # Normalize
        mag = math.sqrt(dx*dx + dy*dy)
        if mag > eps:
            dx /= mag
            dy /= mag
        
        gradients.append((dx, dy))
    
    return gradients

def density_flow(xs, ys):
    """
    Compute density flow - trees are attracted toward the centroid.
    This helps pack trees more tightly.
    """
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    
    flows = []
    for x, y in zip(xs, ys):
        dx = (cx - x)
        dy = (cy - y)
        # Normalize
        mag = math.sqrt(dx*dx + dy*dy)
        if mag > 1e-6:
            dx /= mag
            dy /= mag
        flows.append((dx, dy))
    
    return flows

def repulsion_force(xs, ys, angles, min_dist=0.1):
    """
    Compute repulsion forces between trees that are too close.
    This prevents overlaps.
    """
    n = len(xs)
    forces = [(0.0, 0.0) for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < min_dist and dist > 1e-6:
                # Repulsion force inversely proportional to distance
                force = (min_dist - dist) / dist
                fx = dx * force
                fy = dy * force
                
                forces[i] = (forces[i][0] + fx, forces[i][1] + fy)
                forces[j] = (forces[j][0] - fx, forces[j][1] - fy)
    
    return forces

def gradient_descent_optimize(xs, ys, angles, max_iterations=5000, 
                               initial_step=0.01, density_weight=0.3,
                               momentum=0.9, verbose=True):
    """
    Optimize tree positions using gradient descent with momentum.
    """
    n = len(xs)
    xs = list(xs)
    ys = list(ys)
    angles = list(angles)
    
    best_score = compute_score(xs, ys, angles, n)
    best_xs, best_ys, best_angles = xs.copy(), ys.copy(), angles.copy()
    
    # Momentum terms
    vx = [0.0] * n
    vy = [0.0] * n
    
    step_size = initial_step
    no_improve_count = 0
    
    for iteration in range(max_iterations):
        # Compute gradients
        bbox_grad = compute_bbox_gradient(xs, ys, angles)
        density = density_flow(xs, ys)
        repulsion = repulsion_force(xs, ys, angles)
        
        # Combine gradients
        new_xs = []
        new_ys = []
        
        for i in range(n):
            # Combined gradient
            dx = bbox_grad[i][0] + density_weight * density[i][0] + repulsion[i][0]
            dy = bbox_grad[i][1] + density_weight * density[i][1] + repulsion[i][1]
            
            # Apply momentum
            vx[i] = momentum * vx[i] + (1 - momentum) * dx
            vy[i] = momentum * vy[i] + (1 - momentum) * dy
            
            new_xs.append(xs[i] + vx[i] * step_size)
            new_ys.append(ys[i] + vy[i] * step_size)
        
        # Check for overlaps
        if check_overlap(new_xs, new_ys, angles):
            # Reduce step size and try again
            step_size *= 0.5
            no_improve_count += 1
            if step_size < 1e-6:
                break
            continue
        
        # Compute new score
        new_score = compute_score(new_xs, new_ys, angles, n)
        
        if new_score < best_score:
            best_score = new_score
            best_xs = new_xs.copy()
            best_ys = new_ys.copy()
            xs = new_xs
            ys = new_ys
            no_improve_count = 0
            # Increase step size slightly
            step_size = min(step_size * 1.1, initial_step)
        else:
            no_improve_count += 1
            # Reduce step size
            step_size *= 0.95
        
        if no_improve_count > 100:
            # Try a random restart with small perturbation
            for i in range(n):
                xs[i] = best_xs[i] + np.random.uniform(-0.01, 0.01)
                ys[i] = best_ys[i] + np.random.uniform(-0.01, 0.01)
            step_size = initial_step
            no_improve_count = 0
        
        if verbose and iteration % 500 == 0:
            print(f"  Iter {iteration}: score={best_score:.6f}, step={step_size:.6f}")
    
    return best_xs, best_ys, best_angles, best_score

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
    print("GRADIENT-BASED DENSITY FLOW OPTIMIZATION")
    print("=" * 60)
    
    # Load baseline
    baseline_path = "/home/submission/submission.csv"
    print(f"\nLoading baseline from {baseline_path}...")
    solutions = load_baseline_solution(baseline_path)
    
    # Test on small N first
    test_ns = [10, 15, 20, 25, 30, 40, 50]
    
    results = {}
    improvements = 0
    total_improvement = 0.0
    
    print("\n" + "=" * 60)
    print("TESTING ON SMALL N VALUES")
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
        
        # Run gradient descent
        start_time = time.time()
        new_xs, new_ys, new_angles, new_score = gradient_descent_optimize(
            xs, ys, angles, 
            max_iterations=3000,
            initial_step=0.02,
            density_weight=0.3,
            momentum=0.9,
            verbose=True
        )
        elapsed = time.time() - start_time
        
        improvement = baseline_score - new_score
        results[n] = {
            'baseline': baseline_score,
            'optimized': new_score,
            'improvement': improvement,
            'time': elapsed
        }
        
        if improvement > 1e-6:
            improvements += 1
            total_improvement += improvement
            print(f"  ✅ IMPROVED: {baseline_score:.6f} -> {new_score:.6f} ({improvement:+.6f})")
        else:
            print(f"  ❌ No improvement: {baseline_score:.6f} -> {new_score:.6f}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"N values tested: {len(test_ns)}")
    print(f"Improvements found: {improvements}")
    print(f"Total improvement: {total_improvement:.6f}")
    
    # Save results
    with open('/home/code/experiments/030_gradient_density_flow/small_n_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results, improvements, total_improvement

if __name__ == "__main__":
    results, improvements, total_improvement = main()
