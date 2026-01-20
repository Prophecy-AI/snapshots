"""
Find the absolute best score for each N across ALL snapshots (including overlapping ones).
Then identify which N values have room for improvement.
"""
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import pandas as pd
from pathlib import Path

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

def check_any_overlap(trees, tolerance=1e-9):
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            if trees[i].intersects(trees[j]):
                intersection = trees[i].intersection(trees[j])
                if intersection.area > tolerance:
                    return True
    return False

# Load all snapshots
snapshot_dir = Path('/home/nonroot/snapshots/santa-2025')
snapshots = {}

for snapshot_path in sorted(snapshot_dir.iterdir()):
    if snapshot_path.is_dir():
        submission_path = snapshot_path / 'submission' / 'submission.csv'
        if submission_path.exists():
            try:
                df = pd.read_csv(submission_path)
                if all(col in df.columns for col in ['id', 'x', 'y', 'deg']):
                    snapshots[snapshot_path.name] = df
            except:
                pass

print(f"Loaded {len(snapshots)} snapshots")

# Find best score for each N (with and without overlap checking)
best_any = {}  # Best score regardless of overlaps
best_valid = {}  # Best score with no overlaps

for n in range(1, 201):
    prefix = f"{n:03d}_"
    
    best_any[n] = {'score': float('inf'), 'source': None}
    best_valid[n] = {'score': float('inf'), 'source': None}
    
    for source_name, df in snapshots.items():
        n_rows = df[df['id'].str.startswith(prefix)]
        if len(n_rows) != n:
            continue
        
        trees = []
        for _, row in n_rows.iterrows():
            x = parse_value(row['x'])
            y = parse_value(row['y'])
            deg = parse_value(row['deg'])
            trees.append(create_tree(x, y, deg))
        
        side = get_side(trees)
        score = side**2 / n
        
        if score < best_any[n]['score']:
            best_any[n] = {'score': score, 'source': source_name}
        
        if not check_any_overlap(trees) and score < best_valid[n]['score']:
            best_valid[n] = {'score': score, 'source': source_name}

# Calculate totals
total_any = sum(best_any[n]['score'] for n in range(1, 201))
total_valid = sum(best_valid[n]['score'] for n in range(1, 201))

print(f"\nBest possible score (ignoring overlaps): {total_any:.6f}")
print(f"Best valid score (no overlaps): {total_valid:.6f}")
print(f"Gap due to overlaps: {total_valid - total_any:.6f}")

# Find N values with biggest gaps
gaps = []
for n in range(1, 201):
    gap = best_valid[n]['score'] - best_any[n]['score']
    if gap > 0.0001:
        gaps.append((n, gap, best_any[n]['score'], best_valid[n]['score']))

gaps.sort(key=lambda x: -x[1])

print(f"\nTop 20 N values with biggest gaps (valid vs any):")
for n, gap, any_score, valid_score in gaps[:20]:
    print(f"  N={n}: gap={gap:.6f}, any={any_score:.6f}, valid={valid_score:.6f}")

# Check current submission score
current_df = pd.read_csv('/home/submission/submission.csv')
current_total = 0
for n in range(1, 201):
    prefix = f"{n:03d}_"
    n_rows = current_df[current_df['id'].str.startswith(prefix)]
    trees = []
    for _, row in n_rows.iterrows():
        x = parse_value(row['x'])
        y = parse_value(row['y'])
        deg = parse_value(row['deg'])
        trees.append(create_tree(x, y, deg))
    side = get_side(trees)
    current_total += side**2 / n

print(f"\nCurrent submission score: {current_total:.6f}")
print(f"Best valid score: {total_valid:.6f}")
print(f"Room for improvement: {current_total - total_valid:.6f}")
