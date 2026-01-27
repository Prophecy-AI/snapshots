"""
Physics-Based Packing Simulation

A completely different approach: treat trees as physical objects with:
1. Attractive forces toward the center (gravity)
2. Repulsive forces between overlapping trees (collision)
3. Boundary forces pushing trees inward
4. Damping to reach equilibrium

This is fundamentally different from SA because:
- SA makes random moves and accepts/rejects
- Physics simulation uses continuous force-based movement
- May find different equilibrium states
"""

import numpy as np
import pandas as pd
from shapely import Polygon
from shapely.ops import unary_union
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

def compute_pairwise_forces(xs, ys, angles, repulsion_strength=1.0):
    """
    Compute repulsion forces between all pairs of trees.
    Force is proportional to overlap area.
    """
    n = len(xs)
    polygons = [get_tree_polygon(x, y, a) for x, y, a in zip(xs, ys, angles)]
    
    forces_x = [0.0] * n
    forces_y = [0.0] * n
    
    for i in range(n):
        for j in range(i + 1, n):
            # Direction from j to i
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < 1e-6:
                # Random direction if coincident
                angle = np.random.uniform(0, 2*math.pi)
                dx = math.cos(angle)
                dy = math.sin(angle)
                dist = 1.0
            
            # Check overlap
            if polygons[i].intersects(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                overlap_area = intersection.area
                
                if overlap_area > 1e-10:
                    # Repulsion force proportional to overlap
                    force = repulsion_strength * overlap_area / dist
                    
                    forces_x[i] += force * dx / dist
                    forces_y[i] += force * dy / dist
                    forces_x[j] -= force * dx / dist
                    forces_y[j] -= force * dy / dist
            else:
                # Small attraction to keep trees close (but not overlapping)
                min_dist = 0.5  # Minimum desired distance
                if dist > min_dist:
                    attraction = 0.001 * (dist - min_dist)
                    forces_x[i] -= attraction * dx / dist
                    forces_y[i] -= attraction * dy / dist
                    forces_x[j] += attraction * dx / dist
                    forces_y[j] += attraction * dy / dist
    
    return forces_x, forces_y

def compute_boundary_forces(xs, ys, angles, boundary_strength=0.5):
    """
    Compute forces pushing trees away from the bounding box boundary.
    """
    n = len(xs)
    bbox, min_x, max_x, min_y, max_y = compute_bbox(xs, ys, angles)
    
    forces_x = [0.0] * n
    forces_y = [0.0] * n
    
    for i in range(n):
        vertices = get_tree_vertices(xs[i], ys[i], angles[i])
        
        for vx, vy in vertices:
            # Push away from boundaries
            if abs(vx - min_x) < 0.01:
                forces_x[i] += boundary_strength
            if abs(vx - max_x) < 0.01:
                forces_x[i] -= boundary_strength
            if abs(vy - min_y) < 0.01:
                forces_y[i] += boundary_strength
            if abs(vy - max_y) < 0.01:
                forces_y[i] -= boundary_strength
    
    return forces_x, forces_y

def compute_centroid_forces(xs, ys, centroid_strength=0.01):
    """
    Compute forces attracting trees toward the centroid.
    """
    n = len(xs)
    cx = sum(xs) / n
    cy = sum(ys) / n
    
    forces_x = []
    forces_y = []
    
    for x, y in zip(xs, ys):
        dx = cx - x
        dy = cy - y
        forces_x.append(centroid_strength * dx)
        forces_y.append(centroid_strength * dy)
    
    return forces_x, forces_y

def physics_simulation(xs, ys, angles, n, max_steps=1000, dt=0.01, damping=0.9, verbose=False):
    """
    Run physics simulation to pack trees.
    """
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    angles = np.array(angles, dtype=float)
    
    # Velocities
    vx = np.zeros(len(xs))
    vy = np.zeros(len(xs))
    
    best_score = compute_score(xs, ys, angles, n)
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_angles = angles.copy()
    
    for step in range(max_steps):
        # Compute forces
        rep_fx, rep_fy = compute_pairwise_forces(xs, ys, angles, repulsion_strength=2.0)
        bnd_fx, bnd_fy = compute_boundary_forces(xs, ys, angles, boundary_strength=0.5)
        cen_fx, cen_fy = compute_centroid_forces(xs, ys, centroid_strength=0.02)
        
        # Total force
        fx = np.array(rep_fx) + np.array(bnd_fx) + np.array(cen_fx)
        fy = np.array(rep_fy) + np.array(bnd_fy) + np.array(cen_fy)
        
        # Update velocities with damping
        vx = damping * vx + dt * fx
        vy = damping * vy + dt * fy
        
        # Update positions
        new_xs = xs + dt * vx
        new_ys = ys + dt * vy
        
        # Check if valid (no overlaps)
        if not check_overlap(new_xs, new_ys, angles):
            xs = new_xs
            ys = new_ys
            
            score = compute_score(xs, ys, angles, n)
            if score < best_score:
                best_score = score
                best_xs = xs.copy()
                best_ys = ys.copy()
                best_angles = angles.copy()
        else:
            # Reduce velocities if overlap
            vx *= 0.5
            vy *= 0.5
        
        if verbose and step % 100 == 0:
            print(f"    Step {step}: score={best_score:.6f}")
    
    return best_xs.tolist(), best_ys.tolist(), best_angles.tolist(), best_score

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
    print("PHYSICS-BASED PACKING SIMULATION")
    print("=" * 60)
    
    baseline_path = "/home/submission/submission.csv"
    print(f"\nLoading baseline from {baseline_path}...")
    solutions = load_baseline_solution(baseline_path)
    
    test_ns = [5, 10, 15, 20, 25, 30, 40, 50]
    
    results = {}
    improvements = 0
    total_improvement = 0.0
    
    print("\n" + "=" * 60)
    print("TESTING PHYSICS SIMULATION")
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
        
        # Run physics simulation
        opt_xs, opt_ys, opt_angles, opt_score = physics_simulation(
            xs, ys, angles, n, 
            max_steps=500,
            dt=0.005,
            damping=0.8,
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
    
    with open('/home/code/experiments/030_gradient_density_flow/physics_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results, improvements, total_improvement

if __name__ == "__main__":
    results, improvements, total_improvement = main()
