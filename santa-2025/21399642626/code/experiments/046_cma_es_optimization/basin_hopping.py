"""
Basin Hopping optimization - combines local search with random jumps.
"""

import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
from scipy.optimize import basinhopping, minimize
import time
import warnings
warnings.filterwarnings('ignore')

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def parse_coord(val):
    if isinstance(val, str):
        if val.startswith('s'):
            return float(val[1:])
        return float(val)
    return float(val)

def get_tree_vertices(x, y, angle_deg):
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rx = TX * cos_a - TY * sin_a
    ry = TX * sin_a + TY * cos_a
    return rx + x, ry + y

def get_tree_polygon(x, y, angle_deg):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = affinity.rotate(poly, angle_deg, origin=(0, 0))
    poly = affinity.translate(poly, x, y)
    return poly

def compute_bbox_size(trees):
    all_x, all_y = [], []
    for x, y, angle in trees:
        vx, vy = get_tree_vertices(x, y, angle)
        all_x.extend(vx)
        all_y.extend(vy)
    if not all_x:
        return float('inf')
    return max(max(all_x) - min(all_x), max(all_y) - min(all_y))

def compute_score(trees, n):
    if not trees or len(trees) != n:
        return float('inf')
    size = compute_bbox_size(trees)
    return (size ** 2) / n

def compute_overlap_penalty(trees):
    polygons = [get_tree_polygon(x, y, a) for x, y, a in trees]
    total_overlap = 0
    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            if polygons[i].intersects(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                total_overlap += intersection.area
    return total_overlap

def check_overlap(trees, threshold=1e-15):
    polygons = [get_tree_polygon(x, y, a) for x, y, a in trees]
    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > threshold:
                    return True
    return False

def load_baseline(path):
    df = pd.read_csv(path)
    df['n'] = df['id'].apply(lambda x: int(x.split('_')[0]))
    df['x'] = df['x'].apply(parse_coord)
    df['y'] = df['y'].apply(parse_coord)
    df['deg'] = df['deg'].apply(parse_coord)
    
    result = {}
    for n in range(1, 201):
        n_df = df[df['n'] == n]
        if len(n_df) == n:
            result[n] = [(row['x'], row['y'], row['deg']) for _, row in n_df.iterrows()]
    return result

def flatten_solution(trees):
    flat = []
    for x, y, angle in trees:
        flat.extend([x, y, angle])
    return np.array(flat)

def unflatten_solution(flat, n):
    trees = []
    for i in range(n):
        x = flat[i * 3]
        y = flat[i * 3 + 1]
        angle = flat[i * 3 + 2] % 360
        trees.append((x, y, angle))
    return trees

def basin_hopping_optimize(n, initial_trees, baseline_score, niter=50, stepsize=0.05):
    """
    Optimize tree positions using Basin Hopping.
    """
    x0 = flatten_solution(initial_trees)
    
    # Objective function with overlap penalty
    def objective(x):
        trees = unflatten_solution(x, n)
        score = compute_score(trees, n)
        overlap = compute_overlap_penalty(trees)
        return score + 1000 * overlap
    
    # Custom step function for small perturbations
    class SmallStep:
        def __init__(self, stepsize=0.05):
            self.stepsize = stepsize
        def __call__(self, x):
            x_new = x + np.random.uniform(-self.stepsize, self.stepsize, len(x))
            return x_new
    
    try:
        result = basinhopping(
            objective, x0,
            niter=niter,
            T=0.1,
            stepsize=stepsize,
            take_step=SmallStep(stepsize),
            minimizer_kwargs={'method': 'L-BFGS-B'},
            disp=False
        )
        
        best_trees = unflatten_solution(result.x, n)
        
        if not check_overlap(best_trees):
            best_score = compute_score(best_trees, n)
            return best_trees, best_score
        else:
            return None, float('inf')
    except Exception as e:
        print(f"  Basin hopping error: {e}")
        return None, float('inf')

# Load baseline (exp_044)
print("Loading baseline (exp_044)...")
baseline_path = "/home/code/experiments/044_extended_subset_extraction/ensemble_044.csv"
baseline = load_baseline(baseline_path)
baseline_scores = {n: compute_score(baseline[n], n) for n in range(1, 201)}
total_baseline = sum(baseline_scores.values())
print(f"Baseline total: {total_baseline:.6f}")

# Test Basin Hopping on small N values
print("\n" + "="*60)
print("TESTING BASIN HOPPING")
print("="*60)

test_n_values = [5, 10, 15]
MIN_IMPROVEMENT = 0.0001

for n in test_n_values:
    print(f"\nN={n}: Running Basin Hopping...")
    start_time = time.time()
    
    optimized_trees, optimized_score = basin_hopping_optimize(
        n, baseline[n], baseline_scores[n], 
        niter=30, stepsize=0.03
    )
    
    elapsed = time.time() - start_time
    
    if optimized_trees is None:
        print(f"  No valid solution found [{elapsed:.1f}s]")
        continue
    
    improvement = baseline_scores[n] - optimized_score
    
    if improvement > MIN_IMPROVEMENT:
        print(f"  ✅ IMPROVED: {baseline_scores[n]:.6f} -> {optimized_score:.6f} (improvement: {improvement:.6f}) [{elapsed:.1f}s]")
    else:
        print(f"  ❌ No improvement: baseline {baseline_scores[n]:.6f}, BH {optimized_score:.6f} (diff: {improvement:.6f}) [{elapsed:.1f}s]")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("Basin Hopping tested on small N values")
