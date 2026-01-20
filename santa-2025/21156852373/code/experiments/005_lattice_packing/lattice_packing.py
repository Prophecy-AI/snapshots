"""
Grid-Based Lattice Packing for Santa 2025
Based on jiweiliu's super-fast-simulated-annealing-with-translations kernel

Key idea: Generate 2-tree alternating lattice configurations and optimize
lattice parameters (spacing, angles) with SA.
"""

import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import math
import time
from multiprocessing import Pool, cpu_count

# Tree geometry
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def get_tree_vertices(cx, cy, angle_deg):
    """Get 15 vertices of tree polygon at given position and angle."""
    angle_rad = angle_deg * math.pi / 180.0
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    vertices = []
    for i in range(15):
        rx = TX[i] * cos_a - TY[i] * sin_a + cx
        ry = TX[i] * sin_a + TY[i] * cos_a + cy
        vertices.append((rx, ry))
    
    return vertices

def build_polygon(cx, cy, angle_deg):
    """Build Shapely polygon for a tree."""
    return Polygon(get_tree_vertices(cx, cy, angle_deg))

def has_overlap(trees):
    """Check if any trees overlap."""
    polygons = [build_polygon(x, y, a) for x, y, a in trees]
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            if polygons[i].intersects(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > 1e-12:
                    return True
    return False

def get_bounding_box_side(trees):
    """Get bounding box side length."""
    all_points = []
    for x, y, a in trees:
        vertices = get_tree_vertices(x, y, a)
        all_points.extend(vertices)
    
    all_points = np.array(all_points)
    side = max(all_points.max(axis=0) - all_points.min(axis=0))
    return side

def get_score(trees, n):
    """Get score for N trees."""
    side = get_bounding_box_side(trees)
    return (side ** 2) / n

def generate_2tree_lattice(ncols, nrows, a, b, angle1, angle2, append_x=False, append_y=False):
    """
    Generate a 2-tree alternating lattice.
    
    The lattice has two trees per unit cell:
    - Tree 1 at (0, 0) with angle1
    - Tree 2 at (a, b) with angle2
    
    The unit cell is repeated ncols x nrows times.
    """
    trees = []
    
    # Base lattice spacing
    dx = 2 * a  # Horizontal spacing between unit cells
    dy = 2 * b  # Vertical spacing between unit cells
    
    for row in range(nrows):
        for col in range(ncols):
            # Position of unit cell
            cx = col * dx
            cy = row * dy
            
            # Tree 1 (up orientation)
            trees.append((cx, cy, angle1))
            
            # Tree 2 (down orientation, offset by (a, b))
            trees.append((cx + a, cy + b, angle2))
    
    # Optionally add extra trees at edges
    if append_x:
        for row in range(nrows):
            cx = ncols * dx
            cy = row * dy
            trees.append((cx, cy, angle1))
    
    if append_y:
        for col in range(ncols):
            cx = col * dx
            cy = nrows * dy
            trees.append((cx, cy, angle1))
    
    return trees

def optimize_lattice_sa(ncols, nrows, append_x, append_y, n_target, 
                        n_iter=1000, seed=42):
    """
    Optimize lattice parameters using simulated annealing.
    
    Parameters to optimize:
    - a, b: offset of second tree in unit cell
    - angle1, angle2: angles of the two trees
    """
    np.random.seed(seed)
    
    # Initial parameters
    a = 0.5
    b = 0.5
    angle1 = 0.0
    angle2 = 180.0
    
    # Generate initial lattice
    trees = generate_2tree_lattice(ncols, nrows, a, b, angle1, angle2, append_x, append_y)
    
    # Trim to n_target trees
    if len(trees) > n_target:
        trees = trees[:n_target]
    elif len(trees) < n_target:
        return None, float('inf')  # Can't reach target
    
    # Check initial validity
    if has_overlap(trees):
        # Try to find valid initial parameters
        for _ in range(100):
            a = np.random.uniform(0.3, 1.0)
            b = np.random.uniform(0.3, 1.0)
            angle1 = np.random.uniform(0, 360)
            angle2 = np.random.uniform(0, 360)
            trees = generate_2tree_lattice(ncols, nrows, a, b, angle1, angle2, append_x, append_y)
            if len(trees) > n_target:
                trees = trees[:n_target]
            if not has_overlap(trees):
                break
        else:
            return None, float('inf')
    
    best_trees = trees
    best_score = get_score(trees, n_target)
    
    # SA parameters
    T = 0.01
    T_min = 0.00001
    cooling = 0.95
    
    current_a, current_b = a, b
    current_angle1, current_angle2 = angle1, angle2
    current_score = best_score
    
    for iteration in range(n_iter):
        # Propose new parameters
        new_a = current_a + np.random.uniform(-0.05, 0.05)
        new_b = current_b + np.random.uniform(-0.05, 0.05)
        new_angle1 = current_angle1 + np.random.uniform(-5, 5)
        new_angle2 = current_angle2 + np.random.uniform(-5, 5)
        
        # Clamp parameters
        new_a = max(0.2, min(1.5, new_a))
        new_b = max(0.2, min(1.5, new_b))
        
        # Generate new lattice
        new_trees = generate_2tree_lattice(ncols, nrows, new_a, new_b, new_angle1, new_angle2, append_x, append_y)
        if len(new_trees) > n_target:
            new_trees = new_trees[:n_target]
        
        # Check validity
        if has_overlap(new_trees):
            continue
        
        new_score = get_score(new_trees, n_target)
        
        # Accept or reject
        delta = new_score - current_score
        if delta < 0 or np.random.random() < math.exp(-delta / T):
            current_a, current_b = new_a, new_b
            current_angle1, current_angle2 = new_angle1, new_angle2
            current_score = new_score
            
            if new_score < best_score:
                best_score = new_score
                best_trees = new_trees
        
        # Cool down
        T = max(T_min, T * cooling)
    
    return best_trees, best_score

def generate_grid_configs(max_n=200):
    """Generate all viable grid configurations."""
    configs = []
    
    for ncols in range(1, 15):
        for nrows in range(1, 15):
            for append_x in [False, True]:
                for append_y in [False, True]:
                    n_base = 2 * ncols * nrows
                    n_append_x = nrows if append_x else 0
                    n_append_y = ncols if append_y else 0
                    n_trees = n_base + n_append_x + n_append_y
                    
                    if n_trees <= max_n:
                        configs.append((ncols, nrows, append_x, append_y, n_trees))
    
    # Sort by n_trees
    configs.sort(key=lambda x: x[4])
    return configs

def parse_value(val):
    if isinstance(val, str) and val.startswith('s'):
        return val[1:]
    return str(val)

def load_baseline(path):
    """Load baseline submission."""
    df = pd.read_csv(path)
    
    baseline = {}
    for n in range(1, 201):
        prefix = f"{n:03d}_"
        rows = df[df['id'].str.startswith(prefix)]
        
        trees = []
        for _, row in rows.iterrows():
            x = float(parse_value(row['x']))
            y = float(parse_value(row['y']))
            deg = float(parse_value(row['deg']))
            trees.append((x, y, deg))
        
        baseline[n] = trees
    
    return baseline

def main():
    print("Loading baseline...")
    baseline = load_baseline('/home/code/submission_candidates/candidate_000.csv')
    
    # Calculate baseline scores
    baseline_scores = {}
    baseline_total = 0
    for n in range(1, 201):
        score = get_score(baseline[n], n)
        baseline_scores[n] = score
        baseline_total += score
    
    print(f"Baseline total: {baseline_total:.6f}")
    print("="*60)
    
    # Generate grid configurations
    configs = generate_grid_configs(200)
    print(f"Generated {len(configs)} grid configurations")
    
    # Try to improve each N with lattice packing
    improved = {}
    improved_count = 0
    
    start_time = time.time()
    
    for ncols, nrows, append_x, append_y, n_trees in configs:
        if n_trees < 4:  # Skip very small N
            continue
        
        # Try to optimize this configuration
        trees, score = optimize_lattice_sa(ncols, nrows, append_x, append_y, n_trees, n_iter=500)
        
        if trees is not None and score < baseline_scores[n_trees] - 1e-9:
            improved[n_trees] = trees
            improvement = baseline_scores[n_trees] - score
            improved_count += 1
            print(f"N={n_trees}: {baseline_scores[n_trees]:.6f} -> {score:.6f} (improvement: {improvement:.6f})")
    
    elapsed = time.time() - start_time
    print(f"\nOptimization completed in {elapsed:.1f}s")
    print(f"Improved {improved_count} configurations")
    
    # Calculate new total
    new_total = 0
    for n in range(1, 201):
        if n in improved:
            new_total += get_score(improved[n], n)
        else:
            new_total += baseline_scores[n]
    
    print(f"\nNew total: {new_total:.6f}")
    print(f"Total improvement: {baseline_total - new_total:.6f}")
    
    # Save if improved
    if new_total < baseline_total:
        output_rows = []
        for n in range(1, 201):
            trees = improved.get(n, baseline[n])
            for i, (x, y, a) in enumerate(trees):
                output_rows.append({
                    'id': f"{n:03d}_{i}",
                    'x': f"s{x}",
                    'y': f"s{y}",
                    'deg': f"s{a}"
                })
        
        output_df = pd.DataFrame(output_rows)
        output_path = '/home/code/experiments/005_lattice_packing/lattice_improved.csv'
        output_df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
        
        return new_total
    else:
        print("No improvement - keeping baseline")
        return baseline_total

if __name__ == "__main__":
    main()
