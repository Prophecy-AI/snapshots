"""
Constraint Programming Solver for Small N using OR-Tools CP-SAT.
Discretize positions and angles, use custom non-overlap constraints.
"""
import numpy as np
import pandas as pd
from numba import njit
import math
import time
import json
from ortools.sat.python import cp_model
from shapely.geometry import Polygon
from shapely import affinity

# Tree geometry
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

@njit
def compute_bbox_score(xs, ys, angles, tx, ty):
    n = len(xs)
    V = len(tx)
    mnx = 1e300
    mny = 1e300
    mxx = -1e300
    mxy = -1e300
    for i in range(n):
        r = angles[i] * math.pi / 180.0
        c = math.cos(r)
        s = math.sin(r)
        xi = xs[i]
        yi = ys[i]
        for j in range(V):
            X = c * tx[j] - s * ty[j] + xi
            Y = s * tx[j] + c * ty[j] + yi
            if X < mnx: mnx = X
            if X > mxx: mxx = X
            if Y < mny: mny = Y
            if Y > mxy: mxy = Y
    side = max(mxx - mnx, mxy - mny)
    return side * side / n

def get_tree_polygon(x, y, angle):
    """Create tree polygon for overlap checking."""
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = affinity.rotate(poly, angle, origin=(0, 0))
    poly = affinity.translate(poly, x, y)
    return poly

def check_overlap(xs, ys, angles):
    """Check for overlaps between trees."""
    n = len(xs)
    polys = [get_tree_polygon(xs[i], ys[i], angles[i]) for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                return True
    return False

def solve_n_trees_exhaustive(n, baseline_score, grid_size=50, angle_steps=36):
    """
    Exhaustive search for small N using discretized grid.
    For N=2, try all combinations of positions and angles.
    """
    if n > 3:
        return None, None  # Too slow for N>3
    
    # Discretize space
    positions = np.linspace(-1.5, 1.5, grid_size)
    angles_list = np.linspace(0, 360, angle_steps, endpoint=False)
    
    best_score = baseline_score
    best_config = None
    
    if n == 2:
        # Fix first tree at origin with angle 0
        for x2 in positions:
            for y2 in positions:
                for a1 in angles_list:
                    for a2 in angles_list:
                        xs = np.array([0.0, x2])
                        ys = np.array([0.0, y2])
                        angles = np.array([a1, a2])
                        
                        # Check overlap
                        if check_overlap(xs, ys, angles):
                            continue
                        
                        score = compute_bbox_score(xs, ys, angles, TX, TY)
                        if score < best_score - 1e-8:
                            best_score = score
                            best_config = (xs.copy(), ys.copy(), angles.copy())
    
    elif n == 3:
        # Coarser grid for N=3
        positions = np.linspace(-1.5, 1.5, 20)
        angles_list = np.linspace(0, 360, 12, endpoint=False)
        
        for x2 in positions:
            for y2 in positions:
                for x3 in positions:
                    for y3 in positions:
                        for a1 in angles_list:
                            for a2 in angles_list:
                                for a3 in angles_list:
                                    xs = np.array([0.0, x2, x3])
                                    ys = np.array([0.0, y2, y3])
                                    angles = np.array([a1, a2, a3])
                                    
                                    if check_overlap(xs, ys, angles):
                                        continue
                                    
                                    score = compute_bbox_score(xs, ys, angles, TX, TY)
                                    if score < best_score - 1e-8:
                                        best_score = score
                                        best_config = (xs.copy(), ys.copy(), angles.copy())
    
    return best_score, best_config

def solve_n_trees_cp(n, baseline_score, time_limit=60):
    """
    Use CP-SAT to find optimal placement for N trees.
    Discretize positions and angles, add non-overlap constraints.
    """
    model = cp_model.CpModel()
    
    # Scale factor for discretization (1000 = 0.001 precision)
    SCALE = 100
    MAX_COORD = 300  # Max coordinate in scaled units (3.0 in real units)
    
    # Variables: x, y position for each tree (scaled integers)
    xs = [model.NewIntVar(-MAX_COORD, MAX_COORD, f'x_{i}') for i in range(n)]
    ys = [model.NewIntVar(-MAX_COORD, MAX_COORD, f'y_{i}') for i in range(n)]
    
    # Angles: discretize to 36 values (0, 10, 20, ..., 350 degrees)
    angles = [model.NewIntVar(0, 35, f'a_{i}') for i in range(n)]
    
    # Bounding box variables
    min_x = model.NewIntVar(-MAX_COORD, MAX_COORD, 'min_x')
    max_x = model.NewIntVar(-MAX_COORD, MAX_COORD, 'max_x')
    min_y = model.NewIntVar(-MAX_COORD, MAX_COORD, 'min_y')
    max_y = model.NewIntVar(-MAX_COORD, MAX_COORD, 'max_y')
    
    # Constraints: min/max of positions
    model.AddMinEquality(min_x, xs)
    model.AddMaxEquality(max_x, xs)
    model.AddMinEquality(min_y, ys)
    model.AddMaxEquality(max_y, ys)
    
    # Non-overlap constraints: minimum distance between tree centers
    # This is a simplification - actual overlap depends on angles
    MIN_DIST = 60  # Minimum distance in scaled units (0.6 in real units)
    for i in range(n):
        for j in range(i+1, n):
            # Manhattan distance approximation
            dx = model.NewIntVar(-2*MAX_COORD, 2*MAX_COORD, f'dx_{i}_{j}')
            dy = model.NewIntVar(-2*MAX_COORD, 2*MAX_COORD, f'dy_{i}_{j}')
            model.Add(dx == xs[i] - xs[j])
            model.Add(dy == ys[i] - ys[j])
            
            # Absolute values
            abs_dx = model.NewIntVar(0, 2*MAX_COORD, f'abs_dx_{i}_{j}')
            abs_dy = model.NewIntVar(0, 2*MAX_COORD, f'abs_dy_{i}_{j}')
            model.AddAbsEquality(abs_dx, dx)
            model.AddAbsEquality(abs_dy, dy)
            
            # Minimum distance constraint (L1 norm approximation)
            model.Add(abs_dx + abs_dy >= MIN_DIST)
    
    # Objective: minimize bounding box side
    width = model.NewIntVar(0, 2*MAX_COORD, 'width')
    height = model.NewIntVar(0, 2*MAX_COORD, 'height')
    model.Add(width == max_x - min_x)
    model.Add(height == max_y - min_y)
    
    side = model.NewIntVar(0, 2*MAX_COORD, 'side')
    model.AddMaxEquality(side, [width, height])
    model.Minimize(side)
    
    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Extract solution
        xs_sol = np.array([solver.Value(xs[i]) / SCALE for i in range(n)])
        ys_sol = np.array([solver.Value(ys[i]) / SCALE for i in range(n)])
        angles_sol = np.array([solver.Value(angles[i]) * 10.0 for i in range(n)])
        
        # Verify no overlaps
        if check_overlap(xs_sol, ys_sol, angles_sol):
            return None, None
        
        score = compute_bbox_score(xs_sol, ys_sol, angles_sol, TX, TY)
        return score, (xs_sol, ys_sol, angles_sol)
    
    return None, None

def strip(v):
    return float(str(v).replace("s", ""))

def df_to_arrays(df):
    xs = np.array([strip(v) for v in df['x']])
    ys = np.array([strip(v) for v in df['y']])
    angles = np.array([strip(v) for v in df['deg']])
    return xs, ys, angles

def main():
    print("="*70)
    print("Constraint Programming Solver for Small N")
    print("="*70)
    
    # Load baseline
    baseline_df = pd.read_csv('/home/submission/submission.csv')
    baseline_df['N'] = baseline_df['id'].str.split('_').str[0].astype(int)
    
    baseline_scores = {}
    baseline_configs = {}
    for n in range(1, 201):
        g = baseline_df[baseline_df['N'] == n]
        xs, ys, angles = df_to_arrays(g)
        baseline_scores[n] = compute_bbox_score(xs, ys, angles, TX, TY)
        baseline_configs[n] = (xs.copy(), ys.copy(), angles.copy())
    
    baseline_total = sum(baseline_scores.values())
    print(f"Baseline total: {baseline_total:.6f}")
    
    # Test on small N values
    improvements = []
    start_time = time.time()
    
    # Test N=2 with exhaustive search
    print("\n--- Testing N=2 with exhaustive search ---")
    n = 2
    t0 = time.time()
    score, config = solve_n_trees_exhaustive(n, baseline_scores[n], grid_size=40, angle_steps=36)
    elapsed = time.time() - t0
    print(f"N={n}: Exhaustive search completed in {elapsed:.1f}s")
    if config is not None:
        improvement = baseline_scores[n] - score
        if improvement > 1e-8:
            improvements.append((n, improvement, score, baseline_scores[n], config))
            print(f"  ✅ IMPROVED: {baseline_scores[n]:.6f} -> {score:.6f} (+{improvement:.6f})")
        else:
            print(f"  No improvement: baseline={baseline_scores[n]:.6f}, found={score:.6f}")
    else:
        print(f"  No valid solution found")
    
    # Test N=3-10 with CP-SAT
    print("\n--- Testing N=3-10 with CP-SAT ---")
    for n in range(3, 11):
        print(f"\nSolving N={n}...")
        t0 = time.time()
        score, config = solve_n_trees_cp(n, baseline_scores[n], time_limit=30)
        elapsed = time.time() - t0
        print(f"  CP-SAT completed in {elapsed:.1f}s")
        
        if config is not None:
            improvement = baseline_scores[n] - score
            if improvement > 1e-8:
                improvements.append((n, improvement, score, baseline_scores[n], config))
                print(f"  ✅ IMPROVED: {baseline_scores[n]:.6f} -> {score:.6f} (+{improvement:.6f})")
            else:
                print(f"  No improvement: baseline={baseline_scores[n]:.6f}, found={score:.6f}")
        else:
            print(f"  No valid solution found")
    
    total_elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"Constraint Programming Complete")
    print(f"  Total elapsed time: {total_elapsed:.1f}s")
    print(f"  Improvements found: {len(improvements)}")
    
    if improvements:
        total_improvement = sum(imp for _, imp, _, _, _ in improvements)
        print(f"  Total improvement: {total_improvement:.6f}")
        print("\nImproved N values:")
        for n, imp, new_score, old_score, _ in sorted(improvements, key=lambda x: -x[1]):
            print(f"  N={n}: {old_score:.6f} -> {new_score:.6f} (+{imp:.6f})")
    else:
        print("  No improvements found")
    
    print("="*70)
    
    # Save results
    results = {
        'improvements': [(n, imp, new_s, old_s) for n, imp, new_s, old_s, _ in improvements],
        'total_improvement': sum(imp for _, imp, _, _, _ in improvements) if improvements else 0,
        'elapsed_time': total_elapsed
    }
    
    with open('cp_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return improvements

if __name__ == "__main__":
    improvements = main()
