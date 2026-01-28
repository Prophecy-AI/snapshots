"""
Simulated Annealing with Overlap Relaxation

Key difference from previous SA attempts:
- ALLOWS temporary overlaps during search
- Uses overlap penalty in objective function
- Repairs overlaps at the end
- This enables exploration of configurations that would be rejected by strict SA

The hypothesis is that the current solution is at a boundary optimum where
strict no-overlap constraint prevents exploration. By relaxing this constraint
temporarily, we might find better configurations.
"""

import numpy as np
import pandas as pd
import math
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

def count_overlaps(trees):
    """Count number of overlapping tree pairs."""
    polys = [get_tree_polygon(x, y, deg) for x, y, deg in trees]
    count = 0
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                count += 1
    return count

def compute_overlap_area(trees):
    """Compute total overlap area."""
    polys = [get_tree_polygon(x, y, deg) for x, y, deg in trees]
    total_area = 0
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                try:
                    total_area += polys[i].intersection(polys[j]).area
                except:
                    total_area += 0.01
    return total_area

def check_overlaps(trees):
    return count_overlaps(trees) > 0

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

def perturb(trees, step_size=0.05, angle_step=5):
    """Perturb a random tree."""
    n = len(trees)
    idx = np.random.randint(n)
    x, y, deg = trees[idx]
    
    # Random perturbation
    dx = np.random.normal(0, step_size)
    dy = np.random.normal(0, step_size)
    da = np.random.normal(0, angle_step)
    
    new_trees = list(trees)
    new_trees[idx] = (x + dx, y + dy, deg + da)
    return new_trees

def repair_overlaps(trees, max_attempts=1000):
    """Try to repair overlaps by moving overlapping trees apart."""
    trees = list(trees)
    n = len(trees)
    
    for attempt in range(max_attempts):
        overlaps = count_overlaps(trees)
        if overlaps == 0:
            return trees, True
        
        # Find overlapping pairs
        polys = [get_tree_polygon(x, y, deg) for x, y, deg in trees]
        for i in range(n):
            for j in range(i+1, n):
                if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                    # Move tree j away from tree i
                    xi, yi, _ = trees[i]
                    xj, yj, degj = trees[j]
                    
                    # Direction from i to j
                    dx = xj - xi
                    dy = yj - yi
                    dist = math.sqrt(dx*dx + dy*dy) + 1e-10
                    
                    # Move j slightly away
                    move_dist = 0.01
                    trees[j] = (xj + dx/dist * move_dist, yj + dy/dist * move_dist, degj)
                    break
            else:
                continue
            break
    
    return trees, count_overlaps(trees) == 0

def sa_with_overlap_relaxation(initial_trees, max_iter=50000, T_start=1.0, T_end=0.001, 
                                overlap_penalty=0.5, step_size=0.03, angle_step=3):
    """
    SA that allows temporary overlaps during search.
    
    Key features:
    - Overlap penalty added to objective function
    - Accepts moves that create overlaps (with penalty)
    - Tracks best VALID solution (no overlaps)
    - Attempts to repair overlaps at the end
    """
    n = len(initial_trees)
    current = list(initial_trees)
    current_score = compute_bbox_score(current)
    current_overlaps = count_overlaps(current)
    
    best_valid = list(initial_trees)
    best_valid_score = current_score if current_overlaps == 0 else float('inf')
    
    T = T_start
    T_decay = (T_end / T_start) ** (1.0 / max_iter)
    
    accepted_with_overlaps = 0
    found_better_valid = 0
    
    for iteration in range(max_iter):
        # Perturb
        new_trees = perturb(current, step_size, angle_step)
        new_score = compute_bbox_score(new_trees)
        new_overlaps = count_overlaps(new_trees)
        
        # Augmented objective with overlap penalty
        current_obj = current_score + overlap_penalty * current_overlaps
        new_obj = new_score + overlap_penalty * new_overlaps
        
        # Accept or reject
        delta = new_obj - current_obj
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current = new_trees
            current_score = new_score
            current_overlaps = new_overlaps
            
            if current_overlaps > 0:
                accepted_with_overlaps += 1
            
            # Track best valid solution
            if current_overlaps == 0 and current_score < best_valid_score:
                best_valid = list(current)
                best_valid_score = current_score
                found_better_valid += 1
        
        T *= T_decay
    
    # Try to repair final solution if it has overlaps
    if current_overlaps > 0:
        repaired, success = repair_overlaps(current)
        if success:
            repaired_score = compute_bbox_score(repaired)
            if repaired_score < best_valid_score:
                best_valid = repaired
                best_valid_score = repaired_score
    
    return best_valid, best_valid_score, {
        'accepted_with_overlaps': accepted_with_overlaps,
        'found_better_valid': found_better_valid
    }

# Load baseline
print("Loading baseline submission...")
baseline_df = pd.read_csv('/home/submission/submission.csv')

# Load baseline scores
baseline_scores = {}
baseline_configs = {}
for n in range(1, 201):
    trees = load_baseline_config(baseline_df, n)
    baseline_configs[n] = trees
    baseline_scores[n] = compute_bbox_score(trees)

print(f"Total baseline score: {sum(baseline_scores.values()):.6f}")

# Track best per-N
best_per_n = {n: {'score': baseline_scores[n], 'trees': baseline_configs[n], 'source': 'baseline'} 
              for n in range(1, 201)}

print("\n" + "="*60)
print("SA WITH OVERLAP RELAXATION")
print("="*60)

improvements = {}

# Test on various N values
test_ns = [5, 10, 15, 20, 25, 30, 40, 50]

for n in test_ns:
    print(f"\nN={n}: baseline={baseline_scores[n]:.6f}")
    
    # Run SA with overlap relaxation
    sa_trees, sa_score, stats = sa_with_overlap_relaxation(
        baseline_configs[n], 
        max_iter=30000,
        T_start=0.5,
        T_end=0.001,
        overlap_penalty=0.3,
        step_size=0.02,
        angle_step=2
    )
    
    print(f"  SA score: {sa_score:.6f}")
    print(f"  Accepted with overlaps: {stats['accepted_with_overlaps']}")
    print(f"  Found better valid: {stats['found_better_valid']}")
    
    if sa_score < best_per_n[n]['score'] - 1e-10:
        best_per_n[n] = {
            'score': sa_score, 
            'trees': sa_trees, 
            'source': 'sa_overlap_relaxation'
        }
        improvements[n] = baseline_scores[n] - sa_score
        print(f"  ✅ IMPROVEMENT: {baseline_scores[n] - sa_score:.8f}")
    else:
        print(f"  ❌ No improvement")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

total_improvement = 0
improved_ns = []

for n in test_ns:
    if n in improvements:
        total_improvement += improvements[n]
        improved_ns.append(n)
        print(f"N={n}: improved by {improvements[n]:.8f}")

if not improved_ns:
    print("\nNo improvements found with SA + overlap relaxation")
    print("Even allowing temporary overlaps doesn't help escape the boundary optimum")

print(f"\nTotal improvement: {total_improvement:.8f}")

# Save metrics
metrics = {
    'cv_score': 70.316492,
    'improvements_found': len(improved_ns),
    'total_improvement': total_improvement,
    'improved_n_values': improved_ns,
    'method': 'sa_overlap_relaxation',
    'n_values_tested': test_ns,
    'max_iterations': 30000,
    'overlap_penalty': 0.3
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Copy baseline as submission
import shutil
shutil.copy('/home/submission/submission.csv', 'submission.csv')

print("\nMetrics saved")
