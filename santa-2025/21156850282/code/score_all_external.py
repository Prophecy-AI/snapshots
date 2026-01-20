"""
Score ALL external datasets and find the best configurations per N.
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

def check_any_overlap(trees, tolerance=1e-9):
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            if trees[i].intersects(trees[j]):
                intersection = trees[i].intersection(trees[j])
                if intersection.area > tolerance:
                    return True
    return False

def get_side_fast(xs, ys, degs):
    """Fast bounding box calculation."""
    n = len(xs)
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    
    for i in range(n):
        r = degs[i] * math.pi / 180
        c, s = math.cos(r), math.sin(r)
        
        for j in range(len(TX)):
            x = TX[j] * c - TY[j] * s + xs[i]
            y = TX[j] * s + TY[j] * c + ys[i]
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
    
    return max(max_x - min_x, max_y - min_y)

def score_csv(path, check_overlaps=False):
    """Calculate total score for a CSV file."""
    try:
        df = pd.read_csv(path)
        if not all(col in df.columns for col in ['id', 'x', 'y', 'deg']):
            return None, {}, {}
        
        total = 0
        scores = {}
        overlaps = {}
        for n in range(1, 201):
            prefix = f"{n:03d}_"
            n_rows = df[df['id'].str.startswith(prefix)]
            if len(n_rows) != n:
                continue
            
            xs = [parse_value(row['x']) for _, row in n_rows.iterrows()]
            ys = [parse_value(row['y']) for _, row in n_rows.iterrows()]
            degs = [parse_value(row['deg']) for _, row in n_rows.iterrows()]
            
            side = get_side_fast(xs, ys, degs)
            score = side**2 / n
            scores[n] = score
            total += score
            
            if check_overlaps:
                trees = [create_tree(xs[i], ys[i], degs[i]) for i in range(n)]
                overlaps[n] = check_any_overlap(trees)
        
        return total, scores, overlaps
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None, {}, {}

# Find all CSVs
external_dirs = [
    '/home/code/external_data',
    '/home/code/external_data/jonathanchan',
    '/home/code/external_data/seowoohyeon',
    '/home/code/external_data/telegram',
    '/home/code/external_data/bucket_of_chump',
]

all_csvs = []
for d in external_dirs:
    p = Path(d)
    if p.exists():
        all_csvs.extend(p.glob('*.csv'))

print(f"Found {len(all_csvs)} CSV files")

# Score all
results = {}
for csv_path in all_csvs:
    print(f"Scoring {csv_path.name}...", end=" ")
    total, scores, _ = score_csv(csv_path, check_overlaps=False)
    if total is not None and len(scores) == 200:
        results[str(csv_path)] = {'total': total, 'scores': scores}
        print(f"{total:.6f}")
    else:
        print("SKIP (incomplete)")

# Also add current submission and snapshots
current_total, current_scores, _ = score_csv('/home/submission/submission.csv')
results['current'] = {'total': current_total, 'scores': current_scores}
print(f"\nCurrent submission: {current_total:.6f}")

# Find best per N
best_per_n = {}
for n in range(1, 201):
    best_score = float('inf')
    best_source = None
    for source_name, data in results.items():
        if n in data['scores'] and data['scores'][n] < best_score:
            best_score = data['scores'][n]
            best_source = source_name
    best_per_n[n] = (best_score, best_source)

# Calculate potential improvement
current_total_check = sum(current_scores.values())
best_total = sum(best_per_n[n][0] for n in range(1, 201))

print(f"\nCurrent total: {current_total_check:.6f}")
print(f"Best possible (combining all sources): {best_total:.6f}")
print(f"Potential improvement: {current_total_check - best_total:.6f}")

# Show N values where external sources are better
improvements = []
for n in range(1, 201):
    if n in current_scores:
        best_score, best_source = best_per_n[n]
        if best_source != 'current' and best_score < current_scores[n] - 0.0001:
            improvements.append((n, current_scores[n] - best_score, best_source, best_score, current_scores[n]))

improvements.sort(key=lambda x: -x[1])
print(f"\nN values where external sources are better: {len(improvements)}")
print("Top 30:")
for n, diff, source, new_score, old_score in improvements[:30]:
    source_short = Path(source).name if '/' in source else source
    print(f"  N={n}: improvement={diff:.6f} from {source_short} ({old_score:.6f} -> {new_score:.6f})")

# Check for overlaps in the best configurations
print("\n" + "="*70)
print("Checking for overlaps in best configurations...")
print("="*70)

overlap_count = 0
for n in range(1, 201):
    best_score, best_source = best_per_n[n]
    if best_source != 'current':
        # Load and check
        df = pd.read_csv(best_source)
        prefix = f"{n:03d}_"
        n_rows = df[df['id'].str.startswith(prefix)]
        xs = [parse_value(row['x']) for _, row in n_rows.iterrows()]
        ys = [parse_value(row['y']) for _, row in n_rows.iterrows()]
        degs = [parse_value(row['deg']) for _, row in n_rows.iterrows()]
        trees = [create_tree(xs[i], ys[i], degs[i]) for i in range(n)]
        if check_any_overlap(trees):
            overlap_count += 1
            print(f"  N={n}: OVERLAP in {Path(best_source).name}")

print(f"\nTotal configurations with overlaps: {overlap_count}")
