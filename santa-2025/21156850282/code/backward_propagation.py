"""
Backward Propagation: Start from N=200 and work backwards.
For each N, try removing each tree from N+1 configuration to see if it improves N.
This is the technique from egortrushin kernel.
"""
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely import affinity
from shapely.ops import unary_union
from decimal import Decimal, getcontext
import os

getcontext().prec = 25

TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])
BASE_TREE = Polygon(zip(TX, TY))

def parse_value(val):
    if isinstance(val, str):
        if val.startswith('s'):
            return float(val[1:])
        return float(val)
    return float(val)

def ensure_s_prefix(val):
    if isinstance(val, str):
        if val.startswith('s'):
            return val
        return f's{val}'
    return f's{val}'

def create_tree(x, y, deg):
    tree = affinity.rotate(BASE_TREE, deg, origin=(0, 0))
    tree = affinity.translate(tree, x, y)
    return tree

def get_side(trees):
    if not trees:
        return 0
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    for tree in trees:
        bounds = tree.bounds
        min_x = min(min_x, bounds[0])
        min_y = min(min_y, bounds[1])
        max_x = max(max_x, bounds[2])
        max_y = max(max_y, bounds[3])
    return max(max_x - min_x, max_y - min_y)

def check_any_overlap(trees, tolerance=1e-9):
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            if trees[i].intersects(trees[j]):
                intersection = trees[i].intersection(trees[j])
                if intersection.area > tolerance:
                    return True
    return False

# Load current best submission
print("Loading current submission...")
df = pd.read_csv('/home/submission/submission.csv')

# Parse all configurations
configs = {}  # {n: [(x, y, deg), ...]}
for n in range(1, 201):
    prefix = f"{n:03d}_"
    n_rows = df[df['id'].str.startswith(prefix)]
    configs[n] = []
    for _, row in n_rows.iterrows():
        x = parse_value(row['x'])
        y = parse_value(row['y'])
        deg = parse_value(row['deg'])
        configs[n].append((x, y, deg))

# Calculate initial scores
initial_scores = {}
for n in range(1, 201):
    trees = [create_tree(x, y, deg) for x, y, deg in configs[n]]
    side = get_side(trees)
    initial_scores[n] = side**2 / n

initial_total = sum(initial_scores.values())
print(f"Initial total score: {initial_total:.6f}")

# Backward propagation
print("\nRunning backward propagation...")
improvements = []

for n in range(200, 1, -1):
    # Current score for N-1
    current_score = initial_scores[n-1]
    
    # Try removing each tree from N configuration
    best_score = current_score
    best_config = None
    
    for tree_to_remove in range(n):
        # Create candidate configuration by removing one tree
        candidate = [configs[n][i] for i in range(n) if i != tree_to_remove]
        
        # Create trees and check validity
        trees = [create_tree(x, y, deg) for x, y, deg in candidate]
        
        # Check for overlaps
        if check_any_overlap(trees):
            continue
        
        # Calculate score
        side = get_side(trees)
        score = side**2 / (n-1)
        
        if score < best_score:
            best_score = score
            best_config = candidate
    
    # Update if improved
    if best_config is not None:
        improvement = current_score - best_score
        improvements.append((n-1, improvement, current_score, best_score))
        configs[n-1] = best_config
        initial_scores[n-1] = best_score
        print(f"N={n-1}: {current_score:.6f} -> {best_score:.6f} (improvement: {improvement:.6f})")

# Calculate final score
final_total = sum(initial_scores.values())
print(f"\nFinal total score: {final_total:.6f}")
print(f"Total improvement: {initial_total - final_total:.6f}")

# Save if improved
if final_total < initial_total:
    os.makedirs('/home/code/experiments/005_backward_prop', exist_ok=True)
    
    rows = []
    for n in range(1, 201):
        for i, (x, y, deg) in enumerate(configs[n]):
            rows.append({
                'id': f'{n:03d}_{i}',
                'x': ensure_s_prefix(x),
                'y': ensure_s_prefix(y),
                'deg': ensure_s_prefix(deg)
            })
    
    df_out = pd.DataFrame(rows)
    df_out.to_csv('/home/code/experiments/005_backward_prop/submission.csv', index=False)
    print(f"\nSaved improved submission to /home/code/experiments/005_backward_prop/submission.csv")
else:
    print("\nNo improvement from backward propagation")
