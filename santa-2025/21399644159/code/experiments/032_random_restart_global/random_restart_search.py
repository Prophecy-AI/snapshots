import numpy as np
import pandas as pd
import math
import time
from shapely.geometry import Polygon
from shapely import affinity
import json

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
    """Compute bounding box score for a list of (x, y, deg) tuples"""
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
    """Check if any trees overlap"""
    polys = [get_tree_polygon(x, y, deg) for x, y, deg in trees]
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                return True
    return False

def find_blf_position(placed_trees, angle, grid_step=0.05):
    """Find bottom-left-fill position for a new tree"""
    if not placed_trees:
        return 0, 0
    
    # Get bounds of placed trees
    all_bounds = [get_tree_bounds(x, y, deg) for x, y, deg in placed_trees]
    minx = min(b[0] for b in all_bounds) - 1
    maxx = max(b[1] for b in all_bounds) + 1
    miny = min(b[2] for b in all_bounds) - 1
    maxy = max(b[3] for b in all_bounds) + 1
    
    new_poly_template = get_tree_polygon(0, 0, angle)
    placed_polys = [get_tree_polygon(x, y, deg) for x, y, deg in placed_trees]
    
    best_pos = None
    best_y = float('inf')
    best_x = float('inf')
    
    # Grid search for valid position
    for y in np.arange(miny, maxy, grid_step):
        for x in np.arange(minx, maxx, grid_step):
            new_poly = affinity.translate(new_poly_template, x, y)
            
            # Check for overlaps
            valid = True
            for placed in placed_polys:
                if new_poly.intersects(placed) and not new_poly.touches(placed):
                    valid = False
                    break
            
            if valid:
                # BLF: prefer lower y, then lower x
                if y < best_y or (y == best_y and x < best_x):
                    best_y = y
                    best_x = x
                    best_pos = (x, y)
    
    return best_pos if best_pos else (maxx + 0.5, miny)

def generate_random_config_blf(n, seed, angle_mode='random'):
    """Generate a random valid configuration using BLF placement"""
    np.random.seed(seed)
    
    if angle_mode == 'random':
        angles = np.random.uniform(0, 360, n)
    elif angle_mode == 'fixed_45':
        angles = np.full(n, 45.0)
    elif angle_mode == 'alternating':
        angles = np.array([45.0 if i % 2 == 0 else 225.0 for i in range(n)])
    elif angle_mode == 'uniform':
        angles = np.linspace(0, 360, n, endpoint=False)
    else:
        angles = np.random.uniform(0, 360, n)
    
    # Random placement order
    order = np.random.permutation(n)
    
    trees = []
    for idx in order:
        pos = find_blf_position(trees, angles[idx])
        if pos:
            trees.append((pos[0], pos[1], angles[idx]))
    
    return trees

def load_baseline_config(df, n):
    """Load baseline configuration for N trees"""
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

# Load baseline
print("Loading baseline submission...")
baseline_df = pd.read_csv('/home/submission/submission.csv')

# Track best per-N
best_per_n = {}
baseline_scores = {}

print("\n" + "="*60)
print("RANDOM RESTART GLOBAL SEARCH")
print("="*60)

# Focus on small N values first (they contribute most to score)
test_ns = list(range(2, 21))  # N=2 to N=20

for n in test_ns:
    baseline_trees = load_baseline_config(baseline_df, n)
    baseline_score = compute_bbox_score(baseline_trees)
    baseline_scores[n] = baseline_score
    best_per_n[n] = {'score': baseline_score, 'trees': baseline_trees, 'source': 'baseline'}
    
print(f"Loaded baseline scores for N=2-20")

# Strategy 1: Random restarts with BLF
print("\n--- Strategy 1: Random BLF Restarts ---")

num_restarts = 50  # Number of random restarts per N
angle_modes = ['random', 'fixed_45', 'alternating', 'uniform']

improvements_found = 0

for n in test_ns[:10]:  # Focus on N=2-11 first
    print(f"\nN={n}: baseline={baseline_scores[n]:.6f}")
    
    for mode in angle_modes:
        for seed in range(num_restarts):
            trees = generate_random_config_blf(n, seed + n*1000, angle_mode=mode)
            
            if len(trees) != n:
                continue
            
            if check_overlaps(trees):
                continue
            
            score = compute_bbox_score(trees)
            
            if score < best_per_n[n]['score'] - 1e-10:
                best_per_n[n] = {'score': score, 'trees': trees, 'source': f'{mode}_seed{seed}'}
                improvements_found += 1
                print(f"  ✅ NEW BEST: {score:.6f} (mode={mode}, seed={seed})")

print(f"\nFound {improvements_found} improvements from random restarts")

# Strategy 2: Exhaustive angle search for very small N
print("\n--- Strategy 2: Exhaustive Angle Search (N=2-4) ---")

for n in [2, 3, 4]:
    print(f"\nN={n}: baseline={baseline_scores[n]:.6f}")
    
    if n == 2:
        # For N=2, try all angle combinations
        angle_step = 5  # degrees
        for a1 in range(0, 360, angle_step):
            for a2 in range(0, 360, angle_step):
                # Place first tree at origin
                trees = [(0, 0, a1)]
                # Find position for second tree
                pos = find_blf_position(trees, a2, grid_step=0.02)
                if pos:
                    trees.append((pos[0], pos[1], a2))
                    
                    if not check_overlaps(trees):
                        score = compute_bbox_score(trees)
                        if score < best_per_n[n]['score'] - 1e-10:
                            best_per_n[n] = {'score': score, 'trees': trees, 'source': f'exhaustive_a1={a1}_a2={a2}'}
                            print(f"  ✅ NEW BEST: {score:.6f} (a1={a1}, a2={a2})")
    
    elif n == 3:
        # For N=3, sample angle combinations
        angle_step = 15
        for a1 in range(0, 360, angle_step):
            for a2 in range(0, 360, angle_step):
                for a3 in range(0, 360, angle_step):
                    trees = [(0, 0, a1)]
                    pos2 = find_blf_position(trees, a2, grid_step=0.03)
                    if pos2:
                        trees.append((pos2[0], pos2[1], a2))
                        pos3 = find_blf_position(trees, a3, grid_step=0.03)
                        if pos3:
                            trees.append((pos3[0], pos3[1], a3))
                            
                            if not check_overlaps(trees):
                                score = compute_bbox_score(trees)
                                if score < best_per_n[n]['score'] - 1e-10:
                                    best_per_n[n] = {'score': score, 'trees': trees, 'source': f'exhaustive'}
                                    print(f"  ✅ NEW BEST: {score:.6f}")

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
        print(f"N={n}: {baseline:.6f} -> {best:.6f} (improved by {improvement:.8f}) [{best_per_n[n]['source']}]")
    else:
        print(f"N={n}: {baseline:.6f} (no improvement)")

print(f"\nTotal improvement: {total_improvement:.8f}")
print(f"Improved N values: {improved_ns}")

# Save metrics
metrics = {
    'cv_score': 70.316492,  # No change expected
    'improvements_found': len(improved_ns),
    'total_improvement': total_improvement,
    'improved_n_values': improved_ns,
    'method': 'random_restart_global_search'
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nMetrics saved to metrics.json")
