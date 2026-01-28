"""
Long Parallel Simulated Annealing - Run SA for 500,000 iterations per N value.
Uses all 26 CPU cores for parallel execution.
CRITICAL: Includes proper overlap checking to ensure valid solutions.
"""
import numpy as np
import pandas as pd
from numba import njit
import math
import time
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from shapely.geometry import Polygon
from shapely import affinity

# Configuration
NUM_WORKERS = 26
ITERATIONS_PER_N = 100_000  # Start with 100k, can increase if needed
COOLING_RATE = 0.99999

# Tree geometry
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

@njit
def compute_bbox_score(xs, ys, angles, tx, ty):
    """Fast bounding box score computation."""
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

def check_overlaps(xs, ys, angles):
    """Check for overlaps between trees using Shapely."""
    n = len(xs)
    polys = [get_tree_polygon(xs[i], ys[i], angles[i]) for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                return True
    return False

def check_single_overlap(xs, ys, angles, idx):
    """Check if tree at idx overlaps with any other tree."""
    n = len(xs)
    poly_i = get_tree_polygon(xs[idx], ys[idx], angles[idx])
    for j in range(n):
        if j != idx:
            poly_j = get_tree_polygon(xs[j], ys[j], angles[j])
            if poly_i.intersects(poly_j) and not poly_i.touches(poly_j):
                return True
    return False

def optimize_single_n(args):
    """Run SA on a single N value with overlap checking."""
    n, baseline_xs, baseline_ys, baseline_angles, iterations, baseline_score = args
    
    # Start from baseline
    xs = baseline_xs.copy()
    ys = baseline_ys.copy()
    angles = baseline_angles.copy()
    
    current_score = baseline_score
    best_score = baseline_score
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_angles = angles.copy()
    
    T = 0.1  # Initial temperature
    step_size = 0.005  # Small perturbation
    angle_step = 2.0
    
    improvements_found = 0
    
    for iteration in range(iterations):
        # Pick random tree
        i = np.random.randint(n)
        
        # Save old values
        old_x = xs[i]
        old_y = ys[i]
        old_angle = angles[i]
        
        # Random perturbation
        move_type = np.random.randint(3)
        if move_type == 0:  # Position only
            xs[i] += np.random.uniform(-step_size, step_size)
            ys[i] += np.random.uniform(-step_size, step_size)
        elif move_type == 1:  # Angle only
            angles[i] += np.random.uniform(-angle_step, angle_step)
            angles[i] = angles[i] % 360
        else:  # Both
            xs[i] += np.random.uniform(-step_size/2, step_size/2)
            ys[i] += np.random.uniform(-step_size/2, step_size/2)
            angles[i] += np.random.uniform(-angle_step/2, angle_step/2)
            angles[i] = angles[i] % 360
        
        # Check for overlap with this tree
        if check_single_overlap(xs, ys, angles, i):
            # Reject - restore
            xs[i] = old_x
            ys[i] = old_y
            angles[i] = old_angle
            continue
        
        # Compute new score
        new_score = compute_bbox_score(xs, ys, angles, TX, TY)
        
        # Accept or reject
        delta = new_score - current_score
        if delta < 0 or np.random.random() < math.exp(-delta / T):
            current_score = new_score
            if new_score < best_score - 1e-10:
                best_score = new_score
                best_xs = xs.copy()
                best_ys = ys.copy()
                best_angles = angles.copy()
                improvements_found += 1
        else:
            # Reject - restore
            xs[i] = old_x
            ys[i] = old_y
            angles[i] = old_angle
        
        # Cool down
        T *= COOLING_RATE
        
        # Adaptive step size
        if iteration % 10000 == 0 and iteration > 0:
            step_size *= 0.95
            angle_step *= 0.95
    
    improvement = baseline_score - best_score
    return n, best_score, best_xs, best_ys, best_angles, improvement, improvements_found

def strip(v):
    return float(str(v).replace("s", ""))

def df_to_arrays(df):
    xs = np.array([strip(v) for v in df['x']])
    ys = np.array([strip(v) for v in df['y']])
    angles = np.array([strip(v) for v in df['deg']])
    return xs, ys, angles

def main():
    print("="*70)
    print("Long Parallel Simulated Annealing")
    print(f"  Workers: {NUM_WORKERS}")
    print(f"  Iterations per N: {ITERATIONS_PER_N:,}")
    print(f"  Cooling rate: {COOLING_RATE}")
    print("="*70)
    
    # Load baseline
    baseline_df = pd.read_csv('/home/submission/submission.csv')
    baseline_df['N'] = baseline_df['id'].str.split('_').str[0].astype(int)
    
    # Calculate baseline scores
    baseline_scores = {}
    baseline_configs = {}
    for n in range(1, 201):
        g = baseline_df[baseline_df['N'] == n]
        xs, ys, angles = df_to_arrays(g)
        baseline_scores[n] = compute_bbox_score(xs, ys, angles, TX, TY)
        baseline_configs[n] = (xs.copy(), ys.copy(), angles.copy())
    
    baseline_total = sum(baseline_scores.values())
    print(f"\nBaseline total: {baseline_total:.6f}")
    
    # Prepare arguments for parallel execution
    args_list = []
    for n in range(1, 201):
        xs, ys, angles = baseline_configs[n]
        args_list.append((n, xs, ys, angles, ITERATIONS_PER_N, baseline_scores[n]))
    
    # Run in parallel
    print(f"\nStarting parallel optimization...")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(optimize_single_n, args_list))
    
    elapsed = time.time() - start_time
    print(f"\nOptimization completed in {elapsed:.1f}s")
    
    # Collect improvements
    improvements = []
    new_configs = {}
    
    for n, best_score, best_xs, best_ys, best_angles, improvement, num_improvements in results:
        new_configs[n] = (best_xs, best_ys, best_angles)
        
        if improvement > 1e-8:
            improvements.append((n, improvement, best_score, baseline_scores[n], num_improvements))
            print(f"N={n}: {baseline_scores[n]:.8f} -> {best_score:.8f} (+{improvement:.8f}) [{num_improvements} improvements found]")
    
    # Calculate new total
    new_total = sum(compute_bbox_score(new_configs[n][0], new_configs[n][1], new_configs[n][2], TX, TY) 
                   for n in range(1, 201))
    
    print("\n" + "="*70)
    print(f"Long Parallel SA Complete")
    print(f"  Elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"  Improvements found: {len(improvements)}")
    
    if improvements:
        total_improvement = sum(imp for _, imp, _, _, _ in improvements)
        print(f"  Total improvement: {total_improvement:.8f}")
        print(f"  New total score: {new_total:.6f}")
        print(f"  Baseline total: {baseline_total:.6f}")
    else:
        print("  No improvements found - baseline is at local optimum")
    
    print("="*70)
    
    # Save results
    results_dict = {
        'improvements': [(n, imp, new_s, old_s, num_imp) for n, imp, new_s, old_s, num_imp in improvements],
        'total_improvement': sum(imp for _, imp, _, _, _ in improvements) if improvements else 0,
        'elapsed_time': elapsed,
        'iterations_per_n': ITERATIONS_PER_N,
        'num_workers': NUM_WORKERS,
        'baseline_total': baseline_total,
        'new_total': new_total
    }
    
    with open('long_sa_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    return improvements, new_configs

if __name__ == "__main__":
    improvements, new_configs = main()
