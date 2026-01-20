"""
Create mega ensemble from ALL available sources (snapshots + external data).
Focus on finding the best VALID (no overlap) configuration for each N.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from shapely.geometry import Polygon
from shapely import affinity
import os

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

# Collect all CSV files
csv_files = []

# External data
for path in Path('/home/code/external_data').rglob('*.csv'):
    csv_files.append(str(path))

# Snapshots
for path in Path('/home/nonroot/snapshots/santa-2025').rglob('submission.csv'):
    csv_files.append(str(path))

print(f"Found {len(csv_files)} CSV files")

# Load all valid configurations per N
best_per_n = {}  # {n: {'score': score, 'rows': rows, 'source': source}}

for csv_path in csv_files:
    try:
        df = pd.read_csv(csv_path)
        if not all(col in df.columns for col in ['id', 'x', 'y', 'deg']):
            continue
        
        for n in range(1, 201):
            prefix = f"{n:03d}_"
            n_rows = df[df['id'].str.startswith(prefix)]
            if len(n_rows) != n:
                continue
            
            # Parse and create trees
            trees = []
            rows_data = []
            for _, row in n_rows.iterrows():
                x = parse_value(row['x'])
                y = parse_value(row['y'])
                deg = parse_value(row['deg'])
                trees.append(create_tree(x, y, deg))
                rows_data.append({
                    'id': row['id'],
                    'x': x,
                    'y': y,
                    'deg': deg
                })
            
            # Check for overlaps
            if check_any_overlap(trees):
                continue
            
            # Calculate score
            side = get_side(trees)
            score = side**2 / n
            
            # Update best if better
            if n not in best_per_n or score < best_per_n[n]['score']:
                best_per_n[n] = {
                    'score': score,
                    'rows': rows_data,
                    'source': csv_path
                }
    except Exception as e:
        pass

print(f"\nFound valid configurations for {len(best_per_n)} N values")

# Calculate total score
total_score = sum(best_per_n[n]['score'] for n in range(1, 201) if n in best_per_n)
print(f"Total ensemble score: {total_score:.6f}")

# Create submission
rows = []
for n in range(1, 201):
    if n in best_per_n:
        for row in best_per_n[n]['rows']:
            rows.append({
                'id': row['id'],
                'x': ensure_s_prefix(row['x']),
                'y': ensure_s_prefix(row['y']),
                'deg': ensure_s_prefix(row['deg'])
            })
    else:
        print(f"WARNING: No valid configuration for N={n}")

# Save
os.makedirs('/home/code/experiments/005_mega_ensemble_v2', exist_ok=True)
df_out = pd.DataFrame(rows)
df_out.to_csv('/home/code/experiments/005_mega_ensemble_v2/submission.csv', index=False)
print(f"\nSaved to /home/code/experiments/005_mega_ensemble_v2/submission.csv")

# Show sources used
source_counts = {}
for n in range(1, 201):
    if n in best_per_n:
        source = Path(best_per_n[n]['source']).name
        source_counts[source] = source_counts.get(source, 0) + 1

print("\nSources used:")
for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
    print(f"  {source}: {count} N values")

# Compare with current
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

print(f"\nCurrent submission: {current_total:.6f}")
print(f"New ensemble: {total_score:.6f}")
print(f"Improvement: {current_total - total_score:.6f}")
