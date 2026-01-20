"""
Fix overlapping configurations from snapshot 21145966992.
This snapshot has better scores but overlaps - try to fix them.
"""
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import pandas as pd
import random
import math

# Tree polygon vertices
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

def find_overlapping_pairs(trees, tolerance=1e-9):
    """Find all overlapping pairs."""
    pairs = []
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            if trees[i].intersects(trees[j]):
                intersection = trees[i].intersection(trees[j])
                if intersection.area > tolerance:
                    pairs.append((i, j, intersection.area))
    return pairs

def check_any_overlap(trees, tolerance=1e-9):
    return len(find_overlapping_pairs(trees, tolerance)) > 0

def fix_overlaps_local_search(x, y, deg, max_iter=1000):
    """Try to fix overlaps using local search."""
    n = len(x)
    x = x.copy()
    y = y.copy()
    deg = deg.copy()
    
    trees = [create_tree(x[i], y[i], deg[i]) for i in range(n)]
    
    for iteration in range(max_iter):
        pairs = find_overlapping_pairs(trees)
        if not pairs:
            return x, y, deg, True  # Fixed!
        
        # Pick a random overlapping pair
        i, j, area = random.choice(pairs)
        
        # Try to fix by moving one of the trees
        for tree_idx in [i, j]:
            original_x, original_y, original_deg = x[tree_idx], y[tree_idx], deg[tree_idx]
            
            # Try small moves in many directions
            for angle in range(0, 360, 15):
                for distance in [0.005, 0.01, 0.02, 0.03, 0.05]:
                    dx = distance * math.cos(math.radians(angle))
                    dy = distance * math.sin(math.radians(angle))
                    
                    x[tree_idx] = original_x + dx
                    y[tree_idx] = original_y + dy
                    trees[tree_idx] = create_tree(x[tree_idx], y[tree_idx], deg[tree_idx])
                    
                    # Check if this fixes the overlap without creating new ones
                    new_pairs = find_overlapping_pairs(trees)
                    if len(new_pairs) < len(pairs):
                        break  # Keep this move
                else:
                    continue
                break
            else:
                # Try rotation
                for da in range(0, 360, 10):
                    deg[tree_idx] = (original_deg + da) % 360
                    trees[tree_idx] = create_tree(x[tree_idx], y[tree_idx], deg[tree_idx])
                    
                    new_pairs = find_overlapping_pairs(trees)
                    if len(new_pairs) < len(pairs):
                        break
                else:
                    # Restore original
                    x[tree_idx] = original_x
                    y[tree_idx] = original_y
                    deg[tree_idx] = original_deg
                    trees[tree_idx] = create_tree(x[tree_idx], y[tree_idx], deg[tree_idx])
                    continue
                break
            break
    
    return x, y, deg, not check_any_overlap(trees)

def main():
    print("Fixing overlaps in snapshot 21145966992")
    print("=" * 50)
    
    # Load the overlapping snapshot
    overlap_df = pd.read_csv('/home/nonroot/snapshots/santa-2025/21145966992/submission/submission.csv')
    
    # Load clean baseline for comparison
    clean_df = pd.read_csv('/home/submission/submission.csv')
    
    fixed_count = 0
    improved_count = 0
    results = {}
    
    for n in range(1, 201):
        prefix = f"{n:03d}_"
        
        # Get overlapping config
        overlap_rows = overlap_df[overlap_df['id'].str.startswith(prefix)]
        if len(overlap_rows) != n:
            continue
        
        x = np.array([parse_value(row['x']) for _, row in overlap_rows.iterrows()])
        y = np.array([parse_value(row['y']) for _, row in overlap_rows.iterrows()])
        deg = np.array([parse_value(row['deg']) for _, row in overlap_rows.iterrows()])
        
        trees = [create_tree(x[i], y[i], deg[i]) for i in range(n)]
        
        # Check if this config has overlaps
        has_overlap = check_any_overlap(trees)
        overlap_side = get_side(trees)
        overlap_score = overlap_side**2 / n
        
        # Get clean config score
        clean_rows = clean_df[clean_df['id'].str.startswith(prefix)]
        clean_trees = []
        for _, row in clean_rows.iterrows():
            cx = parse_value(row['x'])
            cy = parse_value(row['y'])
            cdeg = parse_value(row['deg'])
            clean_trees.append(create_tree(cx, cy, cdeg))
        clean_side = get_side(clean_trees)
        clean_score = clean_side**2 / n
        
        if has_overlap:
            # Try to fix
            fixed_x, fixed_y, fixed_deg, success = fix_overlaps_local_search(x, y, deg, max_iter=500)
            
            if success:
                fixed_trees = [create_tree(fixed_x[i], fixed_y[i], fixed_deg[i]) for i in range(n)]
                fixed_side = get_side(fixed_trees)
                fixed_score = fixed_side**2 / n
                
                fixed_count += 1
                
                if fixed_score < clean_score:
                    improved_count += 1
                    results[n] = {
                        'x': fixed_x,
                        'y': fixed_y,
                        'deg': fixed_deg,
                        'score': fixed_score
                    }
                    print(f"N={n}: FIXED & IMPROVED! {clean_score:.6f} -> {fixed_score:.6f} (diff: {clean_score - fixed_score:+.6f})")
                else:
                    print(f"N={n}: Fixed but not better ({fixed_score:.6f} vs {clean_score:.6f})")
            else:
                print(f"N={n}: Could not fix overlaps")
        else:
            # No overlap - check if better than clean
            if overlap_score < clean_score:
                results[n] = {
                    'x': x,
                    'y': y,
                    'deg': deg,
                    'score': overlap_score
                }
                print(f"N={n}: Already valid & better! {clean_score:.6f} -> {overlap_score:.6f}")
    
    print(f"\nFixed {fixed_count} configurations")
    print(f"Improved {improved_count} configurations")
    print(f"Total improvements: {len(results)}")
    
    return results

if __name__ == "__main__":
    results = main()
