"""
Fix configurations with small overlaps from snapshot 21145966992.
Focus on cases where overlaps are tiny and might be fixable.
"""
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import pandas as pd
import random
import math

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

def fix_overlaps_aggressive(x, y, deg, max_iter=2000):
    """Aggressively try to fix overlaps."""
    n = len(x)
    x = x.copy()
    y = y.copy()
    deg = deg.copy()
    
    trees = [create_tree(x[i], y[i], deg[i]) for i in range(n)]
    initial_side = get_side(trees)
    
    for iteration in range(max_iter):
        pairs = find_overlapping_pairs(trees)
        if not pairs:
            return x, y, deg, True, get_side(trees)
        
        # Sort by overlap area (fix smallest first)
        pairs.sort(key=lambda p: p[2])
        
        for i, j, area in pairs:
            # Try moving tree i away from tree j
            ci = trees[i].centroid
            cj = trees[j].centroid
            
            # Direction from j to i
            dx = ci.x - cj.x
            dy = ci.y - cj.y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 0:
                dx /= dist
                dy /= dist
            else:
                dx, dy = 1, 0
            
            # Try moving tree i in this direction
            for step in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]:
                new_x = x[i] + dx * step
                new_y = y[i] + dy * step
                new_tree = create_tree(new_x, new_y, deg[i])
                
                # Check if this fixes the overlap
                old_tree = trees[i]
                trees[i] = new_tree
                new_pairs = find_overlapping_pairs(trees)
                
                if len(new_pairs) < len(pairs):
                    x[i] = new_x
                    y[i] = new_y
                    break
                else:
                    trees[i] = old_tree
            else:
                # Try rotating
                for da in [1, -1, 2, -2, 5, -5, 10, -10]:
                    new_deg = (deg[i] + da) % 360
                    new_tree = create_tree(x[i], y[i], new_deg)
                    
                    old_tree = trees[i]
                    trees[i] = new_tree
                    new_pairs = find_overlapping_pairs(trees)
                    
                    if len(new_pairs) < len(pairs):
                        deg[i] = new_deg
                        break
                    else:
                        trees[i] = old_tree
    
    return x, y, deg, not check_any_overlap(trees), get_side(trees)

# Load snapshots
overlap_df = pd.read_csv('/home/nonroot/snapshots/santa-2025/21145966992/submission/submission.csv')
clean_df = pd.read_csv('/home/submission/submission.csv')

print("Fixing small overlaps")
print("=" * 70)

# Find groups with small total overlap area
fixable = []
for n in range(1, 201):
    prefix = f"{n:03d}_"
    
    overlap_rows = overlap_df[overlap_df['id'].str.startswith(prefix)]
    if len(overlap_rows) != n:
        continue
    
    x = np.array([parse_value(row['x']) for _, row in overlap_rows.iterrows()])
    y = np.array([parse_value(row['y']) for _, row in overlap_rows.iterrows()])
    deg = np.array([parse_value(row['deg']) for _, row in overlap_rows.iterrows()])
    trees = [create_tree(x[i], y[i], deg[i]) for i in range(n)]
    
    pairs = find_overlapping_pairs(trees)
    if pairs:
        total_area = sum(p[2] for p in pairs)
        overlap_side = get_side(trees)
        overlap_score = overlap_side**2 / n
        
        # Get clean score
        clean_rows = clean_df[clean_df['id'].str.startswith(prefix)]
        clean_trees = []
        for _, row in clean_rows.iterrows():
            cx = parse_value(row['x'])
            cy = parse_value(row['y'])
            cdeg = parse_value(row['deg'])
            clean_trees.append(create_tree(cx, cy, cdeg))
        clean_score = get_side(clean_trees)**2 / n
        
        if overlap_score < clean_score:
            fixable.append((n, total_area, len(pairs), overlap_score, clean_score, x, y, deg))

# Sort by total overlap area (smallest first)
fixable.sort(key=lambda x: x[1])

print(f"Found {len(fixable)} groups with overlaps that are better than clean")
print("\nAttempting to fix (sorted by overlap area):")

fixed_results = {}
for n, total_area, num_pairs, overlap_score, clean_score, x, y, deg in fixable[:30]:
    print(f"\nN={n}: {num_pairs} pairs, total_area={total_area:.6f}, potential gain={clean_score - overlap_score:.6f}")
    
    fixed_x, fixed_y, fixed_deg, success, fixed_side = fix_overlaps_aggressive(x, y, deg, max_iter=3000)
    
    if success:
        fixed_score = fixed_side**2 / n
        if fixed_score < clean_score:
            print(f"  ✓ FIXED & IMPROVED! {clean_score:.6f} -> {fixed_score:.6f}")
            fixed_results[n] = {
                'x': fixed_x,
                'y': fixed_y,
                'deg': fixed_deg,
                'score': fixed_score,
                'improvement': clean_score - fixed_score
            }
        else:
            print(f"  Fixed but not better: {fixed_score:.6f} vs {clean_score:.6f}")
    else:
        print(f"  ✗ Could not fix")

print(f"\n\nTotal fixed & improved: {len(fixed_results)}")
if fixed_results:
    total_improvement = sum(r['improvement'] for r in fixed_results.values())
    print(f"Total improvement: {total_improvement:.6f}")
    
    # Save the improved configurations
    for n, data in fixed_results.items():
        print(f"  N={n}: improvement={data['improvement']:.6f}")
