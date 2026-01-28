"""
Constructive algorithm for small N values (N=2-20).
Implements exhaustive search with rotation optimization.
"""

import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import pandas as pd
from itertools import product
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
    """Get tree polygon vertices after rotation and translation"""
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rx = TX * cos_a - TY * sin_a
    ry = TX * sin_a + TY * cos_a
    return rx + x, ry + y

def create_tree_polygon(x, y, angle_deg):
    """Create Shapely polygon for a tree"""
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = affinity.rotate(poly, angle_deg, origin=(0, 0))
    poly = affinity.translate(poly, x, y)
    return poly

def compute_bbox_size(trees):
    """Compute bounding box size for a set of trees"""
    all_x, all_y = [], []
    for x, y, angle in trees:
        vx, vy = get_tree_vertices(x, y, angle)
        all_x.extend(vx)
        all_y.extend(vy)
    if not all_x:
        return float('inf')
    return max(max(all_x) - min(all_x), max(all_y) - min(all_y))

def compute_score_for_n(trees, n):
    """Compute score contribution for N trees"""
    size = compute_bbox_size(trees)
    return (size ** 2) / n

def check_overlap(trees, threshold=1e-10):
    """Check if any trees overlap"""
    polygons = [create_tree_polygon(x, y, a) for x, y, a in trees]
    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > threshold:
                    return True
    return False

def load_submission(path):
    """Load submission and return dict of n -> list of (x, y, angle)"""
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

def optimize_n1():
    """Optimize N=1: Find best rotation angle"""
    best_score = float('inf')
    best_angle = 0
    
    # Try many angles
    for angle in np.linspace(0, 360, 3601):  # 0.1 degree increments
        trees = [(0, 0, angle)]
        score = compute_score_for_n(trees, 1)
        if score < best_score:
            best_score = score
            best_angle = angle
    
    return [(0, 0, best_angle)], best_score

def optimize_n2_exhaustive():
    """Optimize N=2: Try many configurations"""
    best_score = float('inf')
    best_config = None
    
    # Try different relative positions and angles
    angles = np.linspace(0, 360, 37)  # 10 degree increments
    
    for a1 in angles:
        for a2 in angles:
            # Try different relative positions
            for dx in np.linspace(-1.5, 1.5, 31):
                for dy in np.linspace(-1.5, 1.5, 31):
                    trees = [(0, 0, a1), (dx, dy, a2)]
                    
                    # Check overlap
                    if check_overlap(trees):
                        continue
                    
                    # Compute score
                    score = compute_score_for_n(trees, 2)
                    if score < best_score:
                        best_score = score
                        best_config = trees
    
    return best_config, best_score

def local_search_n(baseline_trees, n, iterations=1000):
    """Local search to improve a configuration"""
    current = list(baseline_trees)
    current_score = compute_score_for_n(current, n)
    
    for _ in range(iterations):
        # Pick a random tree to perturb
        idx = np.random.randint(n)
        x, y, angle = current[idx]
        
        # Try small perturbations
        for dx in [-0.01, 0, 0.01]:
            for dy in [-0.01, 0, 0.01]:
                for da in [-1, 0, 1]:
                    new_tree = (x + dx, y + dy, (angle + da) % 360)
                    new_config = current.copy()
                    new_config[idx] = new_tree
                    
                    if check_overlap(new_config):
                        continue
                    
                    new_score = compute_score_for_n(new_config, n)
                    if new_score < current_score:
                        current = new_config
                        current_score = new_score
    
    return current, current_score

# Load baseline
print("Loading baseline (exp_039)...")
baseline_path = "/home/code/experiments/039_per_n_analysis/safe_ensemble.csv"
baseline = load_submission(baseline_path)

# Compute baseline scores
baseline_scores = {n: compute_score_for_n(baseline[n], n) for n in range(1, 201)}
total_baseline = sum(baseline_scores.values())
print(f"Baseline total: {total_baseline:.6f}")

# Try to optimize small N values
print("\n" + "="*60)
print("OPTIMIZING SMALL N VALUES")
print("="*60)

improvements = {}

# N=1: Exhaustive rotation search
print("\nN=1: Exhaustive rotation search...")
n1_trees, n1_score = optimize_n1()
if n1_score < baseline_scores[1] - 0.001:
    print(f"  IMPROVED: {baseline_scores[1]:.6f} -> {n1_score:.6f} (improvement: {baseline_scores[1] - n1_score:.6f})")
    improvements[1] = (n1_trees, n1_score)
else:
    print(f"  No improvement: baseline {baseline_scores[1]:.6f}, best found {n1_score:.6f}")

# N=2-10: Local search from baseline
for n in range(2, 11):
    print(f"\nN={n}: Local search from baseline...")
    improved_trees, improved_score = local_search_n(baseline[n], n, iterations=5000)
    
    if improved_score < baseline_scores[n] - 0.001:
        print(f"  IMPROVED: {baseline_scores[n]:.6f} -> {improved_score:.6f} (improvement: {baseline_scores[n] - improved_score:.6f})")
        improvements[n] = (improved_trees, improved_score)
    else:
        print(f"  No improvement: baseline {baseline_scores[n]:.6f}, best found {improved_score:.6f}")

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
    print("Baseline is already at local optimum for small N values")
