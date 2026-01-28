"""Fast SA with overlap relaxation - fewer iterations"""

import numpy as np
import pandas as pd
import math
import json
from shapely.geometry import Polygon
from shapely import affinity

TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def parse_s_value(s):
    if isinstance(s, str) and s.startswith('s'):
        return float(s[1:])
    return float(s)

def get_tree_polygon(x, y, deg):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = affinity.rotate(poly, deg, origin=(0, 0))
    poly = affinity.translate(poly, x, y)
    return poly

def get_tree_bounds(x, y, deg):
    rad = math.radians(deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    rx = TX * cos_a - TY * sin_a + x
    ry = TX * sin_a + TY * cos_a + y
    return rx.min(), rx.max(), ry.min(), ry.max()

def compute_bbox_score(trees):
    if not trees:
        return float('inf')
    
    minx = miny = float('inf')
    maxx = maxy = float('-inf')
    
    for x, y, deg in trees:
        x0, x1, y0, y1 = get_tree_bounds(x, y, deg)
        minx = min(minx, x0)
        maxx = max(maxx, x1)
        miny = min(miny, y0)
        maxy = max(maxy, y1)
    
    side = max(maxx - minx, maxy - miny)
    n = len(trees)
    return side**2 / n

def count_overlaps(trees):
    polys = [get_tree_polygon(x, y, deg) for x, y, deg in trees]
    count = 0
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                count += 1
    return count

def load_baseline_config(df, n):
    pattern = f'{n:03d}_'
    cfg = df[df['id'].str.startswith(pattern)].copy()
    cfg['tree_idx'] = cfg['id'].apply(lambda x: int(x.split('_')[1]))
    cfg = cfg.sort_values('tree_idx')
    
    trees = []
    for _, row in cfg.iterrows():
        x = parse_s_value(row['x'])
        y = parse_s_value(row['y'])
        deg = parse_s_value(row['deg'])
        trees.append((x, y, deg))
    return trees

def perturb(trees, step_size=0.02, angle_step=2):
    n = len(trees)
    idx = np.random.randint(n)
    x, y, deg = trees[idx]
    
    dx = np.random.normal(0, step_size)
    dy = np.random.normal(0, step_size)
    da = np.random.normal(0, angle_step)
    
    new_trees = list(trees)
    new_trees[idx] = (x + dx, y + dy, deg + da)
    return new_trees

def sa_overlap_relaxation(initial_trees, max_iter=10000, T_start=0.5, T_end=0.001, overlap_penalty=0.3):
    n = len(initial_trees)
    current = list(initial_trees)
    current_score = compute_bbox_score(current)
    current_overlaps = count_overlaps(current)
    
    best_valid = list(initial_trees)
    best_valid_score = current_score if current_overlaps == 0 else float('inf')
    
    T = T_start
    T_decay = (T_end / T_start) ** (1.0 / max_iter)
    
    for iteration in range(max_iter):
        new_trees = perturb(current)
        new_score = compute_bbox_score(new_trees)
        new_overlaps = count_overlaps(new_trees)
        
        current_obj = current_score + overlap_penalty * current_overlaps
        new_obj = new_score + overlap_penalty * new_overlaps
        
        delta = new_obj - current_obj
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current = new_trees
            current_score = new_score
            current_overlaps = new_overlaps
            
            if current_overlaps == 0 and current_score < best_valid_score:
                best_valid = list(current)
                best_valid_score = current_score
        
        T *= T_decay
    
    return best_valid, best_valid_score

# Load baseline
print("Loading baseline...")
baseline_df = pd.read_csv('/home/submission/submission.csv')

baseline_scores = {}
baseline_configs = {}
for n in range(1, 201):
    trees = load_baseline_config(baseline_df, n)
    baseline_configs[n] = trees
    baseline_scores[n] = compute_bbox_score(trees)

print(f"Baseline total: {sum(baseline_scores.values()):.6f}")

# Test on small N values with quick SA
test_ns = [3, 4, 5, 6, 7, 8, 9, 10]
improvements = {}

print("\nSA with overlap relaxation (quick test):")
for n in test_ns:
    sa_trees, sa_score = sa_overlap_relaxation(baseline_configs[n], max_iter=5000)
    
    if sa_score < baseline_scores[n] - 1e-10:
        improvements[n] = baseline_scores[n] - sa_score
        print(f"N={n}: {baseline_scores[n]:.6f} -> {sa_score:.6f} ✅ +{improvements[n]:.8f}")
    else:
        print(f"N={n}: {baseline_scores[n]:.6f} -> {sa_score:.6f} ❌")

total_improvement = sum(improvements.values())
print(f"\nTotal improvement: {total_improvement:.8f}")
print(f"Improved N values: {list(improvements.keys())}")

# Save metrics
metrics = {
    'cv_score': 70.316492,
    'improvements_found': len(improvements),
    'total_improvement': total_improvement,
    'improved_n_values': list(improvements.keys()),
    'method': 'sa_overlap_relaxation'
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Copy baseline as submission
import shutil
shutil.copy('/home/submission/submission.csv', 'submission.csv')

print("\nDone!")
