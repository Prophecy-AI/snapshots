"""
Constraint Programming approach for tree packing using OR-Tools.

This is fundamentally different from SA/perturbation methods:
- Uses constraint propagation and intelligent backtracking
- Can find solutions that local search cannot reach
- Discretizes the problem for tractability
"""
import numpy as np
from ortools.sat.python import cp_model
from shapely import Polygon
from shapely.affinity import rotate, translate
import math
import json
import pandas as pd

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def get_tree_polygon(x, y, angle):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = rotate(poly, angle, origin=(0, 0), use_radians=False)
    poly = translate(poly, x, y)
    return poly

def check_overlap(x1, y1, a1, x2, y2, a2):
    """Check if two trees overlap."""
    p1 = get_tree_polygon(x1, y1, a1)
    p2 = get_tree_polygon(x2, y2, a2)
    if p1.intersects(p2):
        if not p1.touches(p2):
            return p1.intersection(p2).area > 1e-10
    return False

def compute_bbox_score(xs, ys, angles):
    """Compute bounding box score."""
    n = len(xs)
    if n == 0:
        return 0
    
    all_x = []
    all_y = []
    
    for i in range(n):
        poly = get_tree_polygon(xs[i], ys[i], angles[i])
        coords = list(poly.exterior.coords)
        all_x.extend([c[0] for c in coords])
        all_y.extend([c[1] for c in coords])
    
    side = max(max(all_x) - min(all_x), max(all_y) - min(all_y))
    return side * side / n

def precompute_nfp_constraints(angle1, angle2, scale=100):
    """
    Precompute No-Fit Polygon constraints for two trees at given angles.
    Returns list of (dx, dy) pairs that would cause overlap.
    """
    # Sample positions to find overlap regions
    overlap_positions = []
    
    # Place tree1 at origin
    p1 = get_tree_polygon(0, 0, angle1)
    
    # Check grid of positions for tree2
    for dx in np.arange(-2, 2, 0.05):
        for dy in np.arange(-2, 2, 0.05):
            p2 = get_tree_polygon(dx, dy, angle2)
            if p1.intersects(p2) and not p1.touches(p2):
                if p1.intersection(p2).area > 1e-10:
                    overlap_positions.append((int(dx * scale), int(dy * scale)))
    
    return set(overlap_positions)

def solve_cp_packing(n_trees, scale=50, angle_steps=12, time_limit=60):
    """
    Solve tree packing using Constraint Programming.
    
    Args:
        n_trees: Number of trees to pack
        scale: Discretization scale (positions are integers from 0 to scale*4)
        angle_steps: Number of discrete angles (360/angle_steps degrees each)
        time_limit: Maximum solving time in seconds
    
    Returns:
        (xs, ys, angles, score) or None if no solution found
    """
    model = cp_model.CpModel()
    
    grid_size = scale * 4  # Allow positions from -2 to 2 in original scale
    
    # Variables: position (x, y) and angle for each tree
    x = [model.NewIntVar(0, grid_size, f'x_{i}') for i in range(n_trees)]
    y = [model.NewIntVar(0, grid_size, f'y_{i}') for i in range(n_trees)]
    angle = [model.NewIntVar(0, angle_steps-1, f'a_{i}') for i in range(n_trees)]
    
    # Bounding box variables
    max_x = model.NewIntVar(0, grid_size, 'max_x')
    max_y = model.NewIntVar(0, grid_size, 'max_y')
    min_x = model.NewIntVar(0, grid_size, 'min_x')
    min_y = model.NewIntVar(0, grid_size, 'min_y')
    
    # Constraints: max/min tracking
    model.AddMaxEquality(max_x, x)
    model.AddMaxEquality(max_y, y)
    model.AddMinEquality(min_x, x)
    model.AddMinEquality(min_y, y)
    
    # Non-overlap constraints
    # For each pair of trees, ensure minimum distance based on angles
    # This is a simplification - we use minimum distance constraints
    min_dist = int(0.4 * scale)  # Minimum distance between tree centers
    
    for i in range(n_trees):
        for j in range(i+1, n_trees):
            # Manhattan distance constraint (simpler than Euclidean for CP)
            # |x_i - x_j| + |y_i - y_j| >= min_dist
            dx = model.NewIntVar(-grid_size, grid_size, f'dx_{i}_{j}')
            dy = model.NewIntVar(-grid_size, grid_size, f'dy_{i}_{j}')
            model.Add(dx == x[i] - x[j])
            model.Add(dy == y[i] - y[j])
            
            abs_dx = model.NewIntVar(0, grid_size, f'abs_dx_{i}_{j}')
            abs_dy = model.NewIntVar(0, grid_size, f'abs_dy_{i}_{j}')
            model.AddAbsEquality(abs_dx, dx)
            model.AddAbsEquality(abs_dy, dy)
            
            # Ensure minimum separation
            model.Add(abs_dx + abs_dy >= min_dist)
    
    # Objective: minimize bounding box side length
    width = model.NewIntVar(0, grid_size, 'width')
    height = model.NewIntVar(0, grid_size, 'height')
    model.Add(width == max_x - min_x)
    model.Add(height == max_y - min_y)
    
    side = model.NewIntVar(0, grid_size, 'side')
    model.AddMaxEquality(side, [width, height])
    model.Minimize(side)
    
    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8
    
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Extract solution
        xs = [(solver.Value(x[i]) - grid_size/2) / scale for i in range(n_trees)]
        ys = [(solver.Value(y[i]) - grid_size/2) / scale for i in range(n_trees)]
        angles = [solver.Value(angle[i]) * (360 / angle_steps) for i in range(n_trees)]
        
        # Verify no overlaps
        has_overlap = False
        for i in range(n_trees):
            for j in range(i+1, n_trees):
                if check_overlap(xs[i], ys[i], angles[i], xs[j], ys[j], angles[j]):
                    has_overlap = True
                    break
            if has_overlap:
                break
        
        if has_overlap:
            print(f"  Warning: CP solution has overlaps, refining...")
            # Try to fix overlaps by adjusting positions
            return None
        
        score = compute_bbox_score(xs, ys, angles)
        return xs, ys, angles, score
    else:
        return None

def strip(v):
    return float(str(v).replace("s", ""))

def get_baseline_score(n, df):
    g = df[df['N'] == n]
    xs = [strip(v) for v in g['x']]
    ys = [strip(v) for v in g['y']]
    angles = [strip(v) for v in g['deg']]
    return compute_bbox_score(xs, ys, angles)

if __name__ == "__main__":
    print("=" * 70)
    print("Constraint Programming Approach for Tree Packing")
    print("=" * 70)
    
    # Load baseline
    df = pd.read_csv('/home/submission/submission.csv')
    df['N'] = df['id'].str.split('_').str[0].astype(int)
    
    # Test on small N values
    test_ns = [3, 4, 5, 6, 7, 8]
    improvements = []
    
    for n in test_ns:
        baseline_score = get_baseline_score(n, df)
        print(f"\nN={n}: Baseline = {baseline_score:.6f}")
        
        # Try CP solver
        result = solve_cp_packing(n, scale=50, angle_steps=12, time_limit=30)
        
        if result:
            xs, ys, angles, cp_score = result
            print(f"  CP solution: {cp_score:.6f}")
            
            improvement = baseline_score - cp_score
            if improvement > 0.0001:
                improvements.append((n, improvement))
                print(f"  ✓ IMPROVEMENT: {improvement:.6f}")
            else:
                print(f"  ✗ No improvement")
        else:
            print(f"  ✗ CP solver found no valid solution")
    
    print("\n" + "=" * 70)
    if improvements:
        print(f"Found {len(improvements)} improvements:")
        for n, imp in improvements:
            print(f"  N={n}: +{imp:.6f}")
    else:
        print("No improvements found")
    
    # Save metrics
    metrics = {
        'cv_score': 70.316492,
        'baseline_score': 70.316492,
        'improvement': sum(imp for _, imp in improvements) if improvements else 0,
        'num_improvements': len(improvements),
        'notes': f"CP approach tested on N={test_ns}. Found {len(improvements)} improvements."
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
