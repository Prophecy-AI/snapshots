"""
Score all external data sources and find the best per-N configurations.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import math
from shapely.geometry import Polygon
from shapely import affinity

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

# Find all CSV files
csv_files = list(Path('/home/code/external_data').rglob('*.csv'))
csv_files += list(Path('/home/nonroot/snapshots/santa-2025').rglob('submission.csv'))

print(f"Found {len(csv_files)} CSV files")

# Score each file
file_scores = {}
all_n_scores = {}  # {n: {source: (score, has_overlap)}}

for csv_path in csv_files:
    try:
        df = pd.read_csv(csv_path)
        if not all(col in df.columns for col in ['id', 'x', 'y', 'deg']):
            continue
        
        total_score = 0
        n_scores = {}
        valid = True
        
        for n in range(1, 201):
            prefix = f"{n:03d}_"
            n_rows = df[df['id'].str.startswith(prefix)]
            if len(n_rows) != n:
                valid = False
                break
            
            trees = []
            for _, row in n_rows.iterrows():
                x = parse_value(row['x'])
                y = parse_value(row['y'])
                deg = parse_value(row['deg'])
                trees.append(create_tree(x, y, deg))
            
            side = get_side(trees)
            score = side**2 / n
            has_overlap = check_any_overlap(trees)
            
            n_scores[n] = (score, has_overlap)
            total_score += score
        
        if valid:
            file_scores[str(csv_path)] = total_score
            for n, (score, has_overlap) in n_scores.items():
                if n not in all_n_scores:
                    all_n_scores[n] = {}
                all_n_scores[n][str(csv_path)] = (score, has_overlap)
            
            print(f"{csv_path.name}: {total_score:.6f}")
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")

# Find best per-N (valid only)
print("\n" + "="*70)
print("Best valid score per N:")
best_valid_total = 0
best_valid_sources = {}

for n in range(1, 201):
    if n not in all_n_scores:
        continue
    
    best_score = float('inf')
    best_source = None
    
    for source, (score, has_overlap) in all_n_scores[n].items():
        if not has_overlap and score < best_score:
            best_score = score
            best_source = source
    
    if best_source:
        best_valid_total += best_score
        best_valid_sources[n] = (best_source, best_score)

print(f"\nBest valid ensemble score: {best_valid_total:.6f}")

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

print(f"Current submission score: {current_total:.6f}")
print(f"Potential improvement: {current_total - best_valid_total:.6f}")

# Show top 10 improvements
print("\nTop 20 N values with biggest improvements:")
improvements = []
for n in range(1, 201):
    if n in best_valid_sources:
        best_score = best_valid_sources[n][1]
        
        # Get current score
        prefix = f"{n:03d}_"
        n_rows = current_df[current_df['id'].str.startswith(prefix)]
        trees = []
        for _, row in n_rows.iterrows():
            x = parse_value(row['x'])
            y = parse_value(row['y'])
            deg = parse_value(row['deg'])
            trees.append(create_tree(x, y, deg))
        current_score = get_side(trees)**2 / n
        
        if best_score < current_score - 0.0001:
            improvements.append((n, current_score - best_score, current_score, best_score, best_valid_sources[n][0]))

improvements.sort(key=lambda x: -x[1])
for n, diff, current, best, source in improvements[:20]:
    source_name = Path(source).name
    print(f"  N={n}: current={current:.6f}, best={best:.6f}, diff={diff:.6f} from {source_name}")
