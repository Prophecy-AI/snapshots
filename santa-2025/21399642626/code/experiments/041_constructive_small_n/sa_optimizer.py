"""
Simulated Annealing optimizer for small N values.
Implements SA from scratch with aggressive parameters.
"""

import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import pandas as pd
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

def create_tree_polygon(x, y, angle_deg):
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

def compute_score_for_n(trees, n):
    size = compute_bbox_size(trees)
    return (size ** 2) / n

def check_overlap(trees, threshold=1e-10):
    polygons = [create_tree_polygon(x, y, a) for x, y, a in trees]
    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > threshold:
                    return True
    return False

def load_submission(path):
    df = pd.read_csv(path)
    if 'id' in df.columns:
        df['n'] = df['id'].apply(lambda x: int(x.split('_')[0]))
        df['i'] = df['id'].apply(lambda x: int(x.split('_')[1]))
    df['x'] = df['x'].apply(parse_coord)
    df['y'] = df['y'].apply(parse_coord)
    if 'deg' in df.columns:
        df['deg'] = df['deg'].apply(parse_coord)
    else:
        df['deg'] = 0
    
    result = {}
    for n in range(1, 201):
        n_df = df[df['n'] == n]
        if len(n_df) == n:
            trees = [(row['x'], row['y'], row['deg']) for _, row in n_df.iterrows()]
            result[n] = trees
    return result

def simulated_annealing(initial_trees, n, max_iterations=50000, initial_temp=1.0, cooling_rate=0.9999):
    """
    Simulated annealing optimizer.
    """
    current = list(initial_trees)
    current_score = compute_score_for_n(current, n)
    
    best = current.copy()
    best_score = current_score
    
    temp = initial_temp
    
    for iteration in range(max_iterations):
        # Pick a random tree to perturb
        idx = np.random.randint(n)
        x, y, angle = current[idx]
        
        # Perturbation size depends on temperature
        scale = temp * 0.5
        dx = np.random.uniform(-scale, scale)
        dy = np.random.uniform(-scale, scale)
        da = np.random.uniform(-30 * temp, 30 * temp)
        
        new_tree = (x + dx, y + dy, (angle + da) % 360)
        new_config = current.copy()
        new_config[idx] = new_tree
        
        # Check overlap
        if check_overlap(new_config):
            temp *= cooling_rate
            continue
        
        new_score = compute_score_for_n(new_config, n)
        
        # Accept or reject
        delta = new_score - current_score
        if delta < 0 or np.random.random() < np.exp(-delta / temp):
            current = new_config
            current_score = new_score
            
            if current_score < best_score:
                best = current.copy()
                best_score = current_score
        
        temp *= cooling_rate
    
    return best, best_score

def random_restart_sa(initial_trees, n, num_restarts=10, iterations_per_restart=10000):
    """
    Run SA multiple times with random restarts.
    """
    best = list(initial_trees)
    best_score = compute_score_for_n(best, n)
    
    for restart in range(num_restarts):
        # Start from initial or slightly perturbed
        if restart == 0:
            start = list(initial_trees)
        else:
            # Random perturbation of initial
            start = []
            for x, y, angle in initial_trees:
                dx = np.random.uniform(-0.1, 0.1)
                dy = np.random.uniform(-0.1, 0.1)
                da = np.random.uniform(-10, 10)
                start.append((x + dx, y + dy, (angle + da) % 360))
            
            # Fix overlaps
            if check_overlap(start):
                start = list(initial_trees)
        
        result, score = simulated_annealing(start, n, max_iterations=iterations_per_restart)
        
        if score < best_score:
            best = result
            best_score = score
    
    return best, best_score

# Load baseline
print("Loading baseline (exp_039)...")
baseline_path = "/home/code/experiments/039_per_n_analysis/safe_ensemble.csv"
baseline = load_submission(baseline_path)

# Compute baseline scores
baseline_scores = {n: compute_score_for_n(baseline[n], n) for n in range(1, 201)}
total_baseline = sum(baseline_scores.values())
print(f"Baseline total: {total_baseline:.6f}")

# Try to optimize small N values with SA
print("\n" + "="*60)
print("SIMULATED ANNEALING ON SMALL N VALUES")
print("="*60)

improvements = {}
MIN_IMPROVEMENT = 0.001  # Safety threshold

for n in range(2, 21):
    print(f"\nN={n}: Running SA with 10 restarts, 10K iterations each...")
    start_time = time.time()
    
    improved_trees, improved_score = random_restart_sa(baseline[n], n, num_restarts=10, iterations_per_restart=10000)
    
    elapsed = time.time() - start_time
    
    if improved_score < baseline_scores[n] - MIN_IMPROVEMENT:
        improvement = baseline_scores[n] - improved_score
        print(f"  IMPROVED: {baseline_scores[n]:.6f} -> {improved_score:.6f} (improvement: {improvement:.6f}) [{elapsed:.1f}s]")
        improvements[n] = (improved_trees, improved_score)
    else:
        print(f"  No improvement: baseline {baseline_scores[n]:.6f}, best found {improved_score:.6f} [{elapsed:.1f}s]")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

if improvements:
    total_improvement = sum(baseline_scores[n] - score for n, (_, score) in improvements.items())
    print(f"Total improvements found: {len(improvements)}")
    print(f"Total improvement: {total_improvement:.6f}")
    
    for n, (trees, score) in sorted(improvements.items()):
        print(f"  N={n}: {baseline_scores[n]:.6f} -> {score:.6f} (improvement: {baseline_scores[n] - score:.6f})")
else:
    print("No improvements found with MIN_IMPROVEMENT=0.001 threshold")
    print("Baseline is already at strong local optimum for small N values")
