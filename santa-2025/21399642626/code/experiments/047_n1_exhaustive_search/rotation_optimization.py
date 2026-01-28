"""
Rotation optimization for all N values.
Try rotating the entire solution to minimize bounding box.
"""

import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
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

def rotate_solution(trees, rotation_angle):
    """Rotate entire solution around centroid by given angle."""
    # Compute centroid
    all_x, all_y = [], []
    for x, y, angle in trees:
        all_x.append(x)
        all_y.append(y)
    cx, cy = np.mean(all_x), np.mean(all_y)
    
    # Rotate each tree position and angle
    cos_r = np.cos(np.radians(rotation_angle))
    sin_r = np.sin(np.radians(rotation_angle))
    
    rotated = []
    for x, y, angle in trees:
        # Rotate position around centroid
        dx, dy = x - cx, y - cy
        new_x = cx + dx * cos_r - dy * sin_r
        new_y = cy + dx * sin_r + dy * cos_r
        # Add rotation to tree angle
        new_angle = (angle + rotation_angle) % 360
        rotated.append((new_x, new_y, new_angle))
    
    return rotated

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

# Load baseline
print("Loading baseline (exp_044)...")
baseline_path = "/home/code/experiments/044_extended_subset_extraction/ensemble_044.csv"
baseline = load_baseline(baseline_path)
baseline_scores = {n: compute_score(baseline[n], n) for n in range(1, 201)}
total_baseline = sum(baseline_scores.values())
print(f"Baseline total: {total_baseline:.6f}")

# Try rotating each N solution to minimize bounding box
print("\n" + "="*60)
print("ROTATION OPTIMIZATION FOR ALL N VALUES")
print("="*60)

all_improvements = {}
MIN_IMPROVEMENT = 0.0001

for n in range(2, 201):  # Skip N=1 (already optimal)
    trees = baseline[n]
    best_score = baseline_scores[n]
    best_rotation = 0
    best_trees = trees
    
    # Try rotations from -45 to 45 degrees with 0.1 degree resolution
    for rotation_int in range(-450, 451):
        rotation = rotation_int / 10.0
        rotated = rotate_solution(trees, rotation)
        score = compute_score(rotated, n)
        
        if score < best_score - MIN_IMPROVEMENT:
            best_score = score
            best_rotation = rotation
            best_trees = rotated
    
    if best_rotation != 0:
        improvement = baseline_scores[n] - best_score
        all_improvements[n] = (best_trees, improvement, best_rotation)
        print(f"✅ N={n}: rotation={best_rotation:.1f}°, improvement={improvement:.6f}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

if all_improvements:
    total_improvement = sum(imp for _, imp, _ in all_improvements.values())
    print(f"Total improvements found: {len(all_improvements)}")
    print(f"Total score improvement: {total_improvement:.6f}")
else:
    print("No improvements found from rotation optimization")
    print("All N solutions are already rotation-optimized")
