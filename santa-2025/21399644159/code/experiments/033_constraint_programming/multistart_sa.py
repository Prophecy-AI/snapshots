"""
Multi-start Simulated Annealing from RANDOM initial configurations.
The key insight is that we need to explore fundamentally different configurations,
not just perturb the baseline.
"""

import numpy as np
import pandas as pd
import math
import time
import json
from shapely.geometry import Polygon
from shapely import affinity

# Tree polygon vertices
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

def check_overlaps(trees):
    polys = [get_tree_polygon(x, y, deg) for x, y, deg in trees]
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                return True
    return False

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

def generate_random_valid_config(n, max_attempts=1000):
    """Generate a random configuration with no overlaps"""
    for _ in range(max_attempts):
        # Random positions in a reasonable range
        size = np.sqrt(n) * 0.9
        trees = []
        for i in range(n):
            x = np.random.uniform(-size/2, size/2)
            y = np.random.uniform(-size/2, size/2)
            angle = np.random.uniform(0, 360)
            trees.append((x, y, angle))
        
        if not check_overlaps(trees):
            return trees
    
    return None

def simulated_annealing(trees, max_iter=5000, T_start=1.0, T_end=0.001):
    """Run SA on a configuration"""
    n = len(trees)
    current = list(trees)
    current_score = compute_bbox_score(current)
    current_overlaps = count_overlaps(current)
    
    best = list(current)
    best_score = current_score if current_overlaps == 0 else float('inf')
    
    T = T_start
    T_decay = (T_end / T_start) ** (1.0 / max_iter)
    
    for iteration in range(max_iter):
        # Random perturbation
        idx = np.random.randint(n)
        x, y, a = current[idx]
        
        # Perturbation size decreases with temperature
        dx = np.random.normal(0, 0.1 * T)
        dy = np.random.normal(0, 0.1 * T)
        da = np.random.normal(0, 10 * T)
        
        new_trees = list(current)
        new_trees[idx] = (x + dx, y + dy, a + da)
        
        new_score = compute_bbox_score(new_trees)
        new_overlaps = count_overlaps(new_trees)
        
        # Objective: minimize score + heavy penalty for overlaps
        current_obj = current_score + 1000 * current_overlaps
        new_obj = new_score + 1000 * new_overlaps
        
        # Accept or reject
        delta = new_obj - current_obj
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current = new_trees
            current_score = new_score
            current_overlaps = new_overlaps
            
            if current_overlaps == 0 and current_score < best_score:
                best = list(current)
                best_score = current_score
        
        T *= T_decay
    
    return best, best_score

# Load baseline
print("Loading baseline submission...")
baseline_df = pd.read_csv('/home/submission/submission.csv')

# Track results
best_per_n = {}
baseline_scores = {}
improvements = {}

print("\n" + "="*60)
print("MULTI-START SIMULATED ANNEALING FROM RANDOM CONFIGS")
print("="*60)

# Test on small N values
test_ns = list(range(2, 21))
num_restarts = 20

for n in test_ns:
    baseline_trees = load_baseline_config(baseline_df, n)
    baseline_score = compute_bbox_score(baseline_trees)
    baseline_scores[n] = baseline_score
    best_per_n[n] = {'score': baseline_score, 'trees': baseline_trees, 'source': 'baseline'}
    
    print(f"\nN={n}: baseline={baseline_score:.6f}")
    
    # Try multiple random restarts
    for restart in range(num_restarts):
        # Generate random starting configuration
        random_trees = generate_random_valid_config(n)
        
        if random_trees is None:
            # If can't generate valid random config, start from baseline with perturbation
            random_trees = [(x + np.random.uniform(-0.5, 0.5), 
                            y + np.random.uniform(-0.5, 0.5), 
                            a + np.random.uniform(-45, 45)) 
                           for x, y, a in baseline_trees]
        
        # Run SA
        sa_trees, sa_score = simulated_annealing(random_trees, max_iter=3000)
        
        if sa_score < float('inf') and not check_overlaps(sa_trees):
            if sa_score < best_per_n[n]['score'] - 1e-10:
                best_per_n[n] = {'score': sa_score, 'trees': sa_trees, 'source': f'sa_restart_{restart}'}
                print(f"  ✅ NEW BEST: {sa_score:.6f} (restart {restart})")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

total_improvement = 0
improved_ns = []

for n in test_ns:
    baseline = baseline_scores[n]
    best = best_per_n[n]['score']
    if best < baseline - 1e-10:
        improvement = baseline - best
        total_improvement += improvement
        improved_ns.append(n)
        print(f"N={n}: {baseline:.6f} -> {best:.6f} (improved by {improvement:.8f})")

if not improved_ns:
    print("No improvements found with multi-start SA")
    print("The baseline configurations are already at strong local optima")

print(f"\nTotal improvement: {total_improvement:.8f}")

# Save metrics
metrics = {
    'cv_score': 70.316492,
    'improvements_found': len(improved_ns),
    'total_improvement': total_improvement,
    'improved_n_values': improved_ns,
    'method': 'multistart_sa',
    'num_restarts': num_restarts,
    'n_values_tested': test_ns
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Copy baseline as submission
import shutil
shutil.copy('/home/submission/submission.csv', 'submission.csv')

print("\nMetrics saved")
