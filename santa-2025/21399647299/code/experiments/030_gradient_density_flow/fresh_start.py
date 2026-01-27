"""
Fresh Start Optimization

Instead of trying to improve the existing solution, generate completely new
configurations from scratch and see if we can find better packings.

Approaches:
1. Hexagonal lattice with optimal angles
2. Spiral placement
3. Random placement with physics settling
4. Grid with rotation optimization
"""

import numpy as np
import pandas as pd
from shapely import Polygon
import math
import time
import json
from itertools import product

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

def hexagonal_lattice(n, spacing=0.8, angle=45):
    """Generate hexagonal lattice positions."""
    xs, ys, angles = [], [], []
    
    # Estimate grid size needed
    side = int(math.ceil(math.sqrt(n * 2)))
    
    count = 0
    for row in range(side * 2):
        for col in range(side * 2):
            if count >= n:
                break
            
            x = col * spacing
            y = row * spacing * math.sqrt(3) / 2
            
            # Offset every other row
            if row % 2 == 1:
                x += spacing / 2
            
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
    
    return xs, ys, angles

def spiral_placement(n, initial_radius=0.5, angle_step=137.5, radius_growth=0.1, tree_angle=45):
    """Generate spiral positions (golden angle)."""
    xs, ys, angles = [], [], []
    
    for i in range(n):
        angle = math.radians(i * angle_step)
        radius = initial_radius + i * radius_growth
        
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        
        xs.append(x)
        ys.append(y)
        angles.append(tree_angle)
    
    return xs, ys, angles

def grid_placement(n, spacing=0.8, tree_angle=45):
    """Generate grid positions."""
    side = int(math.ceil(math.sqrt(n)))
    
    xs, ys, angles = [], [], []
    count = 0
    
    for row in range(side):
        for col in range(side):
            if count >= n:
                break
            
            xs.append(col * spacing)
            ys.append(row * spacing)
            angles.append(tree_angle)
            count += 1
        
        if count >= n:
            break
    
    # Center
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    xs = [x - cx for x in xs]
    ys = [y - cy for y in ys]
    
    return xs, ys, angles

def compact_configuration(xs, ys, angles, n, max_iterations=500):
    """
    Compact a configuration by moving trees toward center while avoiding overlaps.
    """
    xs = list(xs)
    ys = list(ys)
    angles = list(angles)
    
    best_score = compute_score(xs, ys, angles, n)
    best_xs, best_ys = xs.copy(), ys.copy()
    
    for iteration in range(max_iterations):
        # Find centroid
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        
        improved = False
        
        # Try to move each tree toward center
        for i in range(len(xs)):
            dx = cx - xs[i]
            dy = cy - ys[i]
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < 0.01:
                continue
            
            # Try different step sizes
            for step in [0.05, 0.02, 0.01, 0.005]:
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

def optimize_angles(xs, ys, angles, n, angle_options=[0, 30, 45, 60, 90, 120, 135, 150, 180]):
    """
    Try different angles for each tree to minimize bounding box.
    """
    xs = list(xs)
    ys = list(ys)
    angles = list(angles)
    
    best_score = compute_score(xs, ys, angles, n)
    best_angles = angles.copy()
    
    for i in range(len(angles)):
        for new_angle in angle_options:
            old_angle = angles[i]
            angles[i] = new_angle
            
            if check_overlap(xs, ys, angles):
                angles[i] = old_angle
                continue
            
            new_score = compute_score(xs, ys, angles, n)
            if new_score < best_score:
                best_score = new_score
                best_angles = angles.copy()
            else:
                angles[i] = old_angle
    
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
    print("FRESH START OPTIMIZATION")
    print("=" * 60)
    
    baseline_path = "/home/submission/submission.csv"
    print(f"\nLoading baseline from {baseline_path}...")
    solutions = load_baseline_solution(baseline_path)
    
    test_ns = [5, 10, 15, 20, 25, 30]
    
    results = {}
    improvements = 0
    total_improvement = 0.0
    
    print("\n" + "=" * 60)
    print("TESTING FRESH START CONFIGURATIONS")
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
        
        best_score = baseline_score
        best_method = "baseline"
        
        # Try different initial configurations
        configs = [
            ("hexagonal_0.6", hexagonal_lattice(n, spacing=0.6, angle=45)),
            ("hexagonal_0.7", hexagonal_lattice(n, spacing=0.7, angle=45)),
            ("hexagonal_0.8", hexagonal_lattice(n, spacing=0.8, angle=45)),
            ("spiral", spiral_placement(n, initial_radius=0.3, angle_step=137.5, radius_growth=0.08)),
            ("grid_0.7", grid_placement(n, spacing=0.7, tree_angle=45)),
            ("grid_0.8", grid_placement(n, spacing=0.8, tree_angle=45)),
        ]
        
        for name, (xs, ys, angles) in configs:
            # Check if valid
            if check_overlap(xs, ys, angles):
                print(f"  {name}: Initial config has overlaps, skipping")
                continue
            
            initial_score = compute_score(xs, ys, angles, n)
            
            # Compact the configuration
            xs, ys, angles, score = compact_configuration(xs, ys, angles, n, max_iterations=200)
            
            # Optimize angles
            xs, ys, angles, score = optimize_angles(xs, ys, angles, n)
            
            print(f"  {name}: {initial_score:.6f} -> {score:.6f}")
            
            if score < best_score:
                best_score = score
                best_method = name
        
        improvement = baseline_score - best_score
        results[n] = {
            'baseline': baseline_score,
            'best': best_score,
            'best_method': best_method,
            'improvement': improvement
        }
        
        if improvement > 1e-6:
            improvements += 1
            total_improvement += improvement
            print(f"  ✅ IMPROVED by {best_method}: {baseline_score:.6f} -> {best_score:.6f} ({improvement:+.6f})")
        else:
            print(f"  ❌ No improvement over baseline")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"N values tested: {len(test_ns)}")
    print(f"Improvements found: {improvements}")
    print(f"Total improvement: {total_improvement:.6f}")
    
    with open('/home/code/experiments/030_gradient_density_flow/fresh_start_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results, improvements, total_improvement

if __name__ == "__main__":
    results, improvements, total_improvement = main()
