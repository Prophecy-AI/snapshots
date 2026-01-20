"""
Analyze overlaps in snapshot 21145966992 to understand the pattern.
"""
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import pandas as pd

TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])
BASE_TREE = Polygon(zip(TX, TY))

def parse_value(val):
    if isinstance(val, str):
        if val.startswith('s'):
            return float(val[1:])
        return float(val)
    return float(val)

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

# Load both snapshots
overlap_df = pd.read_csv('/home/nonroot/snapshots/santa-2025/21145966992/submission/submission.csv')
clean_df = pd.read_csv('/home/submission/submission.csv')

print("Comparing overlapping vs clean snapshot")
print("=" * 70)

overlap_groups = []
better_groups = []
total_improvement = 0

for n in range(1, 201):
    prefix = f"{n:03d}_"
    
    # Overlapping config
    overlap_rows = overlap_df[overlap_df['id'].str.startswith(prefix)]
    if len(overlap_rows) != n:
        continue
    
    x = [parse_value(row['x']) for _, row in overlap_rows.iterrows()]
    y = [parse_value(row['y']) for _, row in overlap_rows.iterrows()]
    deg = [parse_value(row['deg']) for _, row in overlap_rows.iterrows()]
    trees = [create_tree(x[i], y[i], deg[i]) for i in range(n)]
    
    pairs = find_overlapping_pairs(trees)
    overlap_side = get_side(trees)
    overlap_score = overlap_side**2 / n
    
    # Clean config
    clean_rows = clean_df[clean_df['id'].str.startswith(prefix)]
    clean_trees = []
    for _, row in clean_rows.iterrows():
        cx = parse_value(row['x'])
        cy = parse_value(row['y'])
        cdeg = parse_value(row['deg'])
        clean_trees.append(create_tree(cx, cy, cdeg))
    clean_side = get_side(clean_trees)
    clean_score = clean_side**2 / n
    
    if pairs:
        total_area = sum(p[2] for p in pairs)
        overlap_groups.append((n, len(pairs), total_area, overlap_score, clean_score))
        if overlap_score < clean_score:
            better_groups.append((n, overlap_score, clean_score, clean_score - overlap_score))
            total_improvement += clean_score - overlap_score

print(f"\nGroups with overlaps: {len(overlap_groups)}")
print(f"Groups where overlap version is better: {len(better_groups)}")
print(f"Total potential improvement: {total_improvement:.6f}")

print("\nTop 20 groups where overlap version is better:")
better_groups.sort(key=lambda x: -x[3])
for n, overlap_score, clean_score, diff in better_groups[:20]:
    print(f"  N={n}: overlap={overlap_score:.6f}, clean={clean_score:.6f}, diff={diff:+.6f}")

print("\nOverlap details for top improvement groups:")
for n, overlap_score, clean_score, diff in better_groups[:10]:
    prefix = f"{n:03d}_"
    overlap_rows = overlap_df[overlap_df['id'].str.startswith(prefix)]
    x = [parse_value(row['x']) for _, row in overlap_rows.iterrows()]
    y = [parse_value(row['y']) for _, row in overlap_rows.iterrows()]
    deg = [parse_value(row['deg']) for _, row in overlap_rows.iterrows()]
    trees = [create_tree(x[i], y[i], deg[i]) for i in range(n)]
    pairs = find_overlapping_pairs(trees)
    
    print(f"  N={n}: {len(pairs)} overlapping pairs, areas: {[f'{p[2]:.6f}' for p in pairs[:5]]}")
