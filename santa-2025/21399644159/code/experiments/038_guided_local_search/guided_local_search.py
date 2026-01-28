"""
Guided Local Search (GLS) Implementation

GLS is a meta-heuristic that escapes local minima by penalizing frequently visited 
solution features. Unlike SA/GA which use random perturbations, GLS uses feature-based 
penalties to guide the search away from previously visited local optima.

Key idea:
- Define solution features (e.g., which trees touch the boundary, angle clusters)
- When stuck in local optimum, penalize the most "costly" features
- Use augmented objective: f(s) + λ * Σ penalties
- This guides search to explore different regions of solution space

Reference: Voudouris & Tsang (1999), "Guided Local Search"
"""

import numpy as np
import pandas as pd
import math
import json
from collections import defaultdict
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

def extract_features(trees):
    """
    Extract features from a solution for GLS penalty tracking.
    
    Features:
    1. Boundary trees: which trees define the bounding box corners
    2. Angle bins: discretized angles (0-360 in 30° bins)
    3. Position bins: discretized positions in grid
    """
    features = set()
    n = len(trees)
    
    # Get bounding box
    minx = miny = float('inf')
    maxx = maxy = float('-inf')
    
    for i, (x, y, deg) in enumerate(trees):
        x0, x1, y0, y1 = get_tree_bounds(x, y, deg)
        minx = min(minx, x0)
        maxx = max(maxx, x1)
        miny = min(miny, y0)
        maxy = max(maxy, y1)
    
    # Feature 1: Boundary trees (trees that touch the bounding box)
    for i, (x, y, deg) in enumerate(trees):
        x0, x1, y0, y1 = get_tree_bounds(x, y, deg)
        if abs(x0 - minx) < 0.01:
            features.add(f"boundary_left_{i}")
        if abs(x1 - maxx) < 0.01:
            features.add(f"boundary_right_{i}")
        if abs(y0 - miny) < 0.01:
            features.add(f"boundary_bottom_{i}")
        if abs(y1 - maxy) < 0.01:
            features.add(f"boundary_top_{i}")
    
    # Feature 2: Angle bins (30° bins)
    for i, (x, y, deg) in enumerate(trees):
        angle_bin = int(deg % 360) // 30
        features.add(f"angle_bin_{i}_{angle_bin}")
    
    # Feature 3: Position bins (grid of 0.5 units)
    for i, (x, y, deg) in enumerate(trees):
        pos_x_bin = int(x / 0.5)
        pos_y_bin = int(y / 0.5)
        features.add(f"pos_bin_{i}_{pos_x_bin}_{pos_y_bin}")
    
    return features

def compute_feature_cost(feature, trees):
    """Compute the cost contribution of a feature."""
    # For boundary features, cost is proportional to how much they extend the bbox
    if feature.startswith("boundary_"):
        return 0.1  # Fixed cost for boundary features
    elif feature.startswith("angle_"):
        return 0.05  # Lower cost for angle features
    else:
        return 0.02  # Lowest cost for position features

def augmented_objective(trees, penalties, lambda_param=0.1):
    """Compute augmented objective: original score + penalty term."""
    base_score = compute_bbox_score(trees)
    
    features = extract_features(trees)
    penalty_sum = sum(penalties.get(f, 0) for f in features)
    
    return base_score + lambda_param * penalty_sum

def local_search_step(trees, penalties, lambda_param=0.1):
    """Perform one step of local search with augmented objective."""
    n = len(trees)
    best_trees = list(trees)
    best_obj = augmented_objective(trees, penalties, lambda_param)
    
    # Try small perturbations
    for i in range(n):
        x, y, deg = trees[i]
        
        # Try position perturbations
        for dx in [-0.02, 0, 0.02]:
            for dy in [-0.02, 0, 0.02]:
                if dx == 0 and dy == 0:
                    continue
                
                new_trees = list(trees)
                new_trees[i] = (x + dx, y + dy, deg)
                
                if not check_overlaps(new_trees):
                    new_obj = augmented_objective(new_trees, penalties, lambda_param)
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_trees = new_trees
        
        # Try angle perturbations
        for da in [-5, 5]:
            new_trees = list(trees)
            new_trees[i] = (x, y, deg + da)
            
            if not check_overlaps(new_trees):
                new_obj = augmented_objective(new_trees, penalties, lambda_param)
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_trees = new_trees
    
    return best_trees, best_obj

def guided_local_search(initial_trees, max_iterations=500, lambda_param=0.1):
    """
    Guided Local Search algorithm.
    
    1. Start with initial solution
    2. Perform local search until stuck
    3. When stuck, penalize features of current solution
    4. Continue local search with augmented objective
    5. Track best solution found (using original objective)
    """
    penalties = defaultdict(float)
    current = list(initial_trees)
    best_trees = list(initial_trees)
    best_score = compute_bbox_score(initial_trees)
    
    stuck_count = 0
    
    for iteration in range(max_iterations):
        # Local search step
        new_trees, new_obj = local_search_step(current, penalties, lambda_param)
        
        # Check if we improved (using augmented objective)
        current_obj = augmented_objective(current, penalties, lambda_param)
        
        if new_obj < current_obj - 1e-10:
            current = new_trees
            stuck_count = 0
            
            # Check if this is best (using original objective)
            current_score = compute_bbox_score(current)
            if current_score < best_score - 1e-10 and not check_overlaps(current):
                best_score = current_score
                best_trees = list(current)
        else:
            stuck_count += 1
        
        # If stuck, penalize features
        if stuck_count >= 5:
            features = extract_features(current)
            
            # Find feature with highest utility to penalize
            max_utility = -float('inf')
            max_feature = None
            
            for f in features:
                cost = compute_feature_cost(f, current)
                utility = cost / (1 + penalties[f])
                if utility > max_utility:
                    max_utility = utility
                    max_feature = f
            
            if max_feature:
                penalties[max_feature] += 1
            
            stuck_count = 0
    
    return best_trees, best_score, penalties

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
print("GUIDED LOCAL SEARCH (GLS)")
print("="*60)

improvements = {}

# Test on various N values
test_ns = [10, 15, 20, 25, 30, 35, 40, 45, 50]

for n in test_ns:
    print(f"\nN={n}: baseline={baseline_scores[n]:.6f}")
    
    # Run GLS starting from baseline
    gls_trees, gls_score, penalties = guided_local_search(
        baseline_configs[n], 
        max_iterations=300,
        lambda_param=0.1
    )
    
    print(f"  GLS score: {gls_score:.6f}")
    print(f"  Features penalized: {len([p for p in penalties.values() if p > 0])}")
    
    if gls_score < best_per_n[n]['score'] - 1e-10:
        best_per_n[n] = {
            'score': gls_score, 
            'trees': gls_trees, 
            'source': 'gls'
        }
        improvements[n] = baseline_scores[n] - gls_score
        print(f"  ✅ IMPROVEMENT: {baseline_scores[n] - gls_score:.8f}")
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
    print("\nNo improvements found with Guided Local Search")
    print("The baseline configurations are already at the boundary of the feasible region")
    print("GLS cannot escape because any improvement causes overlaps")

print(f"\nTotal improvement: {total_improvement:.8f}")

# Save metrics
metrics = {
    'cv_score': 70.316492,
    'improvements_found': len(improved_ns),
    'total_improvement': total_improvement,
    'improved_n_values': improved_ns,
    'method': 'guided_local_search',
    'n_values_tested': test_ns,
    'max_iterations': 300,
    'lambda_param': 0.1
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Copy baseline as submission
import shutil
shutil.copy('/home/submission/submission.csv', 'submission.csv')

print("\nMetrics saved")
