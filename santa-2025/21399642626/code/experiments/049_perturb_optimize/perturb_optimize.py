"""
Perturb-and-Optimize: Perturb baseline solutions and then optimize.
This explores different basins of attraction that the original optimization might have missed.
"""

import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import random
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

def perturb_trees(trees, strength=0.01):
    """Perturb tree positions and angles by small random amounts."""
    perturbed = []
    for x, y, angle in trees:
        new_x = x + random.uniform(-strength, strength)
        new_y = y + random.uniform(-strength, strength)
        new_angle = (angle + random.uniform(-strength * 10, strength * 10)) % 360
        perturbed.append((new_x, new_y, new_angle))
    return perturbed

def local_search(trees, n, max_iterations=1000, step_size=0.001):
    """Simple local search to optimize tree positions."""
    current = list(trees)
    current_score = compute_score(current, n)
    
    for _ in range(max_iterations):
        # Pick a random tree to perturb
        idx = random.randint(0, n - 1)
        x, y, angle = current[idx]
        
        # Try small perturbations
        best_move = None
        best_score = current_score
        
        for dx in [-step_size, 0, step_size]:
            for dy in [-step_size, 0, step_size]:
                for da in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and da == 0:
                        continue
                    
                    new_tree = (x + dx, y + dy, (angle + da) % 360)
                    new_config = current.copy()
                    new_config[idx] = new_tree
                    
                    if check_overlap(new_config):
                        continue
                    
                    new_score = compute_score(new_config, n)
                    if new_score < best_score:
                        best_score = new_score
                        best_move = new_config
        
        if best_move is not None:
            current = best_move
            current_score = best_score
    
    return current, current_score

def perturb_and_optimize(n, baseline_trees, baseline_score, num_restarts=10, perturbation_strength=0.01):
    """
    Perturb baseline and then optimize.
    Try multiple random restarts to explore different basins.
    """
    best_trees = baseline_trees
    best_score = baseline_score
    
    for restart in range(num_restarts):
        # Perturb the baseline
        perturbed = perturb_trees(baseline_trees, strength=perturbation_strength)
        
        # Check if perturbed solution is valid
        if check_overlap(perturbed):
            continue
        
        # Run local search on perturbed solution
        optimized, score = local_search(perturbed, n, max_iterations=500, step_size=0.001)
        
        if score < best_score:
            best_score = score
            best_trees = optimized
    
    return best_trees, best_score

# Load baseline
print("Loading baseline (exp_044)...")
baseline_path = "/home/code/experiments/044_extended_subset_extraction/ensemble_044.csv"
baseline = load_baseline(baseline_path)
baseline_scores = {n: compute_score(baseline[n], n) for n in range(1, 201)}
total_baseline = sum(baseline_scores.values())
print(f"Baseline total: {total_baseline:.6f}")

# Test perturb-and-optimize on small N values
print("\n" + "="*60)
print("TESTING PERTURB-AND-OPTIMIZE ON SMALL N VALUES")
print("="*60)

test_n_values = [5, 10, 15, 20]
MIN_IMPROVEMENT = 0.0001

for n in test_n_values:
    print(f"\nN={n}: Running perturb-and-optimize...")
    start_time = time.time()
    
    optimized_trees, optimized_score = perturb_and_optimize(
        n, baseline[n], baseline_scores[n],
        num_restarts=20, perturbation_strength=0.01
    )
    
    elapsed = time.time() - start_time
    improvement = baseline_scores[n] - optimized_score
    
    if improvement > MIN_IMPROVEMENT:
        print(f"  ✅ IMPROVED: {baseline_scores[n]:.6f} -> {optimized_score:.6f} (improvement: {improvement:.6f}) [{elapsed:.1f}s]")
    else:
        print(f"  ❌ No improvement: baseline {baseline_scores[n]:.6f}, optimized {optimized_score:.6f} [{elapsed:.1f}s]")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("Perturb-and-optimize tested on small N values.")
print("If no improvements found, the baseline is at a very strong local optimum.")
