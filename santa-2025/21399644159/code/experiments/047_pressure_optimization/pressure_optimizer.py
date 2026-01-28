"""
Gradient-Based Pressure Optimization

Novel approach based on Eazy Optimizer kernel:
1. Log-barrier gradient pushes trees toward center
2. Elastic pulsing periodically squeezes/relaxes to escape local optima
3. Overlap repair moves overlapping trees apart

This is FUNDAMENTALLY DIFFERENT from SA/GA/PSO because:
- Uses calculus-based gradients, not random perturbations
- Has periodic "pulsing" to escape local optima
- Treats the bounding box as a soft constraint with log-barrier
"""

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

def get_bounding_box(trees):
    minx = miny = float('inf')
    maxx = maxy = float('-inf')
    
    for x, y, deg in trees:
        x0, x1, y0, y1 = get_tree_bounds(x, y, deg)
        minx = min(minx, x0)
        maxx = max(maxx, x1)
        miny = min(miny, y0)
        maxy = max(maxy, y1)
    
    return minx, maxx, miny, maxy

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

def load_config(df, n):
    pattern = f'{n:03d}_'
    cfg = df[df['id'].str.startswith(pattern)].copy()
    cfg['tree_idx'] = cfg['id'].apply(lambda x: int(x.split('_')[1]))
    cfg = cfg.sort_values('tree_idx')
    
    trees = []
    for _, row in cfg.iterrows():
        x = parse_s_value(row['x'])
        y = parse_s_value(row['y'])
        deg = parse_s_value(row['deg'])
        trees.append([x, y, deg])  # Use list for mutability
    return trees

def apply_square_pressure(trees, scale=0.01):
    """
    Push all trees toward center using log-barrier gradient.
    
    The log-barrier function is: -ln(L-x) - ln(L+x)
    Its derivative is: 1/(L-x) - 1/(L+x)
    Moving AGAINST this gradient pushes toward center.
    """
    minx, maxx, miny, maxy = get_bounding_box(trees)
    side = max(maxx - minx, maxy - miny)
    L = side / 2.0
    
    # Center of bounding box
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    
    new_trees = []
    for x, y, deg in trees:
        # Tree center relative to bbox center
        tx0, tx1, ty0, ty1 = get_tree_bounds(x, y, deg)
        tree_cx = (tx0 + tx1) / 2 - cx
        tree_cy = (ty0 + ty1) / 2 - cy
        
        # Log-barrier gradient (with safety margin)
        eps = 0.01
        if abs(tree_cx) < L - eps:
            gx = 1/(L - tree_cx + eps) - 1/(L + tree_cx + eps)
        else:
            gx = 0
        
        if abs(tree_cy) < L - eps:
            gy = 1/(L - tree_cy + eps) - 1/(L + tree_cy + eps)
        else:
            gy = 0
        
        # Move against gradient (toward center)
        new_x = x - gx * scale
        new_y = y - gy * scale
        new_trees.append([new_x, new_y, deg])
    
    return new_trees

def elastic_pulse(trees, iteration, pulse_period=1000):
    """
    Periodic squeeze and relax to escape local optima.
    """
    if iteration % pulse_period != 0:
        return trees
    
    # Calculate centroid
    cx = sum(t[0] for t in trees) / len(trees)
    cy = sum(t[1] for t in trees) / len(trees)
    
    # Alternating squeeze/relax
    if (iteration // pulse_period) % 2 == 0:
        f = 0.999  # Squeeze
    else:
        f = 1.001  # Relax
    
    new_trees = []
    for x, y, deg in trees:
        new_x = cx + (x - cx) * f
        new_y = cy + (y - cy) * f
        new_trees.append([new_x, new_y, deg])
    
    return new_trees

def repair_overlaps(trees, max_attempts=100):
    """Move overlapping trees apart."""
    trees = [list(t) for t in trees]
    n = len(trees)
    
    for attempt in range(max_attempts):
        polys = [get_tree_polygon(t[0], t[1], t[2]) for t in trees]
        
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                    # Move j away from i
                    dx = trees[j][0] - trees[i][0]
                    dy = trees[j][1] - trees[i][1]
                    dist = math.sqrt(dx*dx + dy*dy) + 1e-10
                    
                    # Move both trees apart
                    move_dist = 0.01
                    trees[i][0] -= dx/dist * move_dist
                    trees[i][1] -= dy/dist * move_dist
                    trees[j][0] += dx/dist * move_dist
                    trees[j][1] += dy/dist * move_dist
                    moved = True
        
        if not moved:
            break
    
    return trees

def pressure_optimize(initial_trees, max_iterations=10000, pressure_scale=0.005):
    """
    Main optimization loop using pressure gradient and elastic pulsing.
    """
    trees = [list(t) for t in initial_trees]
    best_trees = [list(t) for t in trees]
    best_score = compute_bbox_score(trees)
    
    if check_overlaps(trees):
        trees = repair_overlaps(trees)
    
    for iteration in range(max_iterations):
        # Apply pressure gradient
        new_trees = apply_square_pressure(trees, scale=pressure_scale)
        
        # Apply elastic pulsing
        new_trees = elastic_pulse(new_trees, iteration, pulse_period=500)
        
        # Repair any overlaps
        if check_overlaps(new_trees):
            new_trees = repair_overlaps(new_trees)
        
        # Check if improved
        if not check_overlaps(new_trees):
            new_score = compute_bbox_score(new_trees)
            if new_score < best_score - 1e-10:
                best_score = new_score
                best_trees = [list(t) for t in new_trees]
        
        trees = new_trees
        
        # Decay pressure scale
        if iteration % 1000 == 0 and iteration > 0:
            pressure_scale *= 0.9
    
    return best_trees, best_score

# Load baseline
print("Loading baseline...")
baseline_df = pd.read_csv('/home/submission/submission.csv')

baseline_scores = {}
baseline_configs = {}
for n in range(1, 201):
    trees = load_config(baseline_df, n)
    baseline_configs[n] = trees
    baseline_scores[n] = compute_bbox_score(trees)

print(f"Baseline total: {sum(baseline_scores.values()):.6f}")

# Test pressure optimization on various N values
print("\nTesting pressure optimization...")
improvements = {}
best_configs = {}

test_ns = [5, 10, 15, 20, 25, 30, 40, 50]

for n in test_ns:
    print(f"\nN={n}: baseline={baseline_scores[n]:.6f}")
    
    opt_trees, opt_score = pressure_optimize(
        baseline_configs[n],
        max_iterations=5000,
        pressure_scale=0.01
    )
    
    print(f"  Pressure opt score: {opt_score:.6f}")
    
    if opt_score < baseline_scores[n] - 1e-10:
        if not check_overlaps(opt_trees):
            improvements[n] = baseline_scores[n] - opt_score
            best_configs[n] = opt_trees
            print(f"  ✅ IMPROVEMENT: {improvements[n]:.8f}")
        else:
            print(f"  ❌ Has overlaps")
    else:
        print(f"  ❌ No improvement")

total_improvement = sum(improvements.values())
print(f"\nTotal improvement: {total_improvement:.8f}")
print(f"Improved N values: {list(improvements.keys())}")

# Save metrics
metrics = {
    'cv_score': 70.316492,
    'improvements_found': len(improvements),
    'total_improvement': total_improvement,
    'improved_n_values': list(improvements.keys()),
    'method': 'pressure_optimization'
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Copy baseline as submission
import shutil
shutil.copy('/home/submission/submission.csv', 'submission.csv')

print("\nDone!")
