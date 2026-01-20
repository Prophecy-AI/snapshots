"""
Backward Propagation (BackPacking) for tree packing.
Use larger N configurations to improve smaller N by removing trees.
"""
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import pandas as pd

TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])
BASE_TREE = Polygon(zip(TX, TY))

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

def parse_value(val):
    if isinstance(val, str):
        if val.startswith('s'):
            return float(val[1:])
        return float(val)
    return float(val)

def load_config(df, n):
    """Load configuration for N from dataframe."""
    prefix = f"{n:03d}_"
    n_rows = df[df['id'].str.startswith(prefix)]
    
    configs = []
    for _, row in n_rows.iterrows():
        x = parse_value(row['x'])
        y = parse_value(row['y'])
        deg = parse_value(row['deg'])
        configs.append((x, y, deg))
    
    return configs

def get_score(configs):
    """Calculate score for a configuration."""
    n = len(configs)
    if n == 0:
        return float('inf')
    trees = [create_tree(x, y, deg) for x, y, deg in configs]
    side = get_side(trees)
    return side**2 / n

def get_bbox_touching_indices(configs):
    """Find indices of trees that touch the bounding box."""
    trees = [create_tree(x, y, deg) for x, y, deg in configs]
    
    min_x = min(t.bounds[0] for t in trees)
    min_y = min(t.bounds[1] for t in trees)
    max_x = max(t.bounds[2] for t in trees)
    max_y = max(t.bounds[3] for t in trees)
    
    tolerance = 0.001
    touching = []
    
    for i, tree in enumerate(trees):
        bounds = tree.bounds
        if (abs(bounds[0] - min_x) < tolerance or 
            abs(bounds[1] - min_y) < tolerance or
            abs(bounds[2] - max_x) < tolerance or
            abs(bounds[3] - max_y) < tolerance):
            touching.append(i)
    
    return touching

def backward_propagation(df):
    """
    Backward propagation: for each N from 200 down to 2,
    try removing each tree from N+1 config and keep best result.
    """
    # Load all configurations
    configs = {}
    for n in range(1, 201):
        configs[n] = load_config(df, n)
    
    # Calculate initial scores
    scores = {n: get_score(configs[n]) for n in range(1, 201)}
    
    improvements = {}
    
    # Backward propagation
    for n in range(199, 0, -1):
        current_score = scores[n]
        parent_config = configs[n + 1]
        
        if len(parent_config) != n + 1:
            continue
        
        # Get indices of trees touching the bounding box
        touching_indices = get_bbox_touching_indices(parent_config)
        
        best_new_config = None
        best_new_score = current_score
        
        # Try removing each touching tree
        for idx in touching_indices:
            new_config = [c for i, c in enumerate(parent_config) if i != idx]
            new_score = get_score(new_config)
            
            if new_score < best_new_score:
                best_new_config = new_config
                best_new_score = new_score
        
        if best_new_config is not None and best_new_score < current_score - 0.0001:
            improvement = current_score - best_new_score
            configs[n] = best_new_config
            scores[n] = best_new_score
            improvements[n] = {
                'old_score': current_score,
                'new_score': best_new_score,
                'improvement': improvement
            }
            print(f"N={n}: {current_score:.6f} -> {best_new_score:.6f} (improvement: {improvement:.6f})")
    
    return improvements, configs, scores

def main():
    print("Backward Propagation Optimization")
    print("=" * 60)
    
    # Load baseline
    baseline_df = pd.read_csv('/home/submission/submission.csv')
    
    improvements, new_configs, new_scores = backward_propagation(baseline_df)
    
    print(f"\n{'=' * 60}")
    print(f"Total improvements found: {len(improvements)}")
    
    if improvements:
        total_improvement = sum(data['improvement'] for data in improvements.values())
        print(f"Total improvement: {total_improvement:.6f}")
        
        # Calculate new total score
        old_total = sum(get_score(load_config(baseline_df, n)) for n in range(1, 201))
        new_total = sum(new_scores.values())
        print(f"\nOld total score: {old_total:.6f}")
        print(f"New total score: {new_total:.6f}")
        print(f"Total improvement: {old_total - new_total:.6f}")
    else:
        print("No improvements found through backward propagation.")
    
    return improvements

if __name__ == "__main__":
    improvements = main()
