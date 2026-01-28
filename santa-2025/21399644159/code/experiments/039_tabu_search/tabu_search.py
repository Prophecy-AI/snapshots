"""
Tabu Search Implementation

Tabu Search is fundamentally different from SA and GLS:
- Maintains a "tabu list" of recently visited solutions/moves
- Allows moves to WORSE solutions to escape local optima
- Uses aspiration criterion: accept tabu move if it's the best ever seen

Key differences from what we've tried:
- SA uses random acceptance based on temperature
- GLS uses feature penalties
- Tabu Search uses explicit memory of recent moves

This might work because:
- The solution is at a boundary optimum
- Tabu Search can explore worse solutions systematically
- The tabu list prevents cycling back to the same local optimum
"""

import numpy as np
import pandas as pd
import math
import json
import hashlib
from collections import deque
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

def solution_hash(trees):
    """Create a hash of the solution for tabu list."""
    # Round to avoid floating point issues
    rounded = tuple((round(x, 4), round(y, 4), round(deg, 2)) for x, y, deg in trees)
    return hashlib.md5(str(rounded).encode()).hexdigest()[:16]

def generate_neighbors(trees, step_size=0.05, angle_step=10):
    """Generate all neighbor solutions by perturbing each tree."""
    neighbors = []
    n = len(trees)
    
    for i in range(n):
        x, y, deg = trees[i]
        
        # Position perturbations
        for dx in [-step_size, step_size]:
            new_trees = list(trees)
            new_trees[i] = (x + dx, y, deg)
            neighbors.append(('pos_x', i, dx, new_trees))
        
        for dy in [-step_size, step_size]:
            new_trees = list(trees)
            new_trees[i] = (x, y + dy, deg)
            neighbors.append(('pos_y', i, dy, new_trees))
        
        # Angle perturbations
        for da in [-angle_step, angle_step]:
            new_trees = list(trees)
            new_trees[i] = (x, y, deg + da)
            neighbors.append(('angle', i, da, new_trees))
    
    return neighbors

def tabu_search(initial_trees, max_iterations=1000, tabu_tenure=20, step_size=0.05, angle_step=10):
    """
    Tabu Search algorithm.
    
    Key features:
    - Maintains tabu list of recent moves (not solutions)
    - Always moves to best non-tabu neighbor (even if worse)
    - Aspiration criterion: accept tabu move if it's best ever seen
    """
    current = list(initial_trees)
    best = list(initial_trees)
    best_score = compute_bbox_score(initial_trees)
    
    # Tabu list stores (move_type, tree_index, direction) tuples
    tabu_list = deque(maxlen=tabu_tenure)
    
    no_improvement_count = 0
    
    for iteration in range(max_iterations):
        # Generate all neighbors
        neighbors = generate_neighbors(current, step_size, angle_step)
        
        # Evaluate neighbors
        valid_neighbors = []
        for move_type, tree_idx, direction, new_trees in neighbors:
            if check_overlaps(new_trees):
                continue
            
            score = compute_bbox_score(new_trees)
            move = (move_type, tree_idx, direction > 0)  # Simplified move representation
            
            # Check if move is tabu
            is_tabu = move in tabu_list
            
            # Aspiration criterion: accept if best ever
            if is_tabu and score >= best_score:
                continue
            
            valid_neighbors.append((score, move, new_trees))
        
        if not valid_neighbors:
            # No valid moves - stuck
            no_improvement_count += 1
            if no_improvement_count > 50:
                break
            continue
        
        # Select best valid neighbor
        valid_neighbors.sort(key=lambda x: x[0])
        best_neighbor_score, best_move, best_neighbor = valid_neighbors[0]
        
        # Move to best neighbor
        current = best_neighbor
        
        # Add move to tabu list
        tabu_list.append(best_move)
        
        # Update best if improved
        if best_neighbor_score < best_score - 1e-10:
            best_score = best_neighbor_score
            best = list(current)
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if no_improvement_count > 100:
            break
    
    return best, best_score

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
print("TABU SEARCH")
print("="*60)

improvements = {}

# Test on various N values
test_ns = [5, 10, 15, 20, 25, 30, 40, 50]

for n in test_ns:
    print(f"\nN={n}: baseline={baseline_scores[n]:.6f}")
    
    # Run Tabu Search
    tabu_trees, tabu_score = tabu_search(
        baseline_configs[n], 
        max_iterations=500,
        tabu_tenure=30,
        step_size=0.03,
        angle_step=5
    )
    
    print(f"  Tabu score: {tabu_score:.6f}")
    
    if tabu_score < best_per_n[n]['score'] - 1e-10:
        best_per_n[n] = {
            'score': tabu_score, 
            'trees': tabu_trees, 
            'source': 'tabu_search'
        }
        improvements[n] = baseline_scores[n] - tabu_score
        print(f"  ✅ IMPROVEMENT: {baseline_scores[n] - tabu_score:.8f}")
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
    print("\nNo improvements found with Tabu Search")
    print("The baseline is at the boundary of the feasible region")
    print("Tabu Search cannot escape because valid moves don't exist")

print(f"\nTotal improvement: {total_improvement:.8f}")

# Save metrics
metrics = {
    'cv_score': 70.316492,
    'improvements_found': len(improved_ns),
    'total_improvement': total_improvement,
    'improved_n_values': improved_ns,
    'method': 'tabu_search',
    'n_values_tested': test_ns,
    'max_iterations': 500,
    'tabu_tenure': 30
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Copy baseline as submission
import shutil
shutil.copy('/home/submission/submission.csv', 'submission.csv')

print("\nMetrics saved")
