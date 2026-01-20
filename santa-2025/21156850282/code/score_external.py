"""
Score external datasets and find the best configurations per N.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import math

TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def parse_value(val):
    if isinstance(val, str):
        if val.startswith('s'):
            return float(val[1:])
        return float(val)
    return float(val)

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

def score_csv(path):
    """Calculate total score for a CSV file."""
    try:
        df = pd.read_csv(path)
        if not all(col in df.columns for col in ['id', 'x', 'y', 'deg']):
            return None, {}
        
        total = 0
        scores = {}
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
        
        return total, scores
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None, {}

# Score all external CSVs
external_dir = Path('/home/code/external_data')
results = {}

for csv_path in external_dir.glob('*.csv'):
    print(f"\nScoring {csv_path.name}...")
    total, scores = score_csv(csv_path)
    if total is not None:
        results[csv_path.name] = {'total': total, 'scores': scores}
        print(f"  Total score: {total:.6f}")

# Also score current submission
current_total, current_scores = score_csv('/home/submission/submission.csv')
print(f"\nCurrent submission: {current_total:.6f}")

# Find best per N across all sources
print("\n" + "="*70)
print("Best scores per N across all sources:")
print("="*70)

all_sources = {**results, 'current': {'total': current_total, 'scores': current_scores}}

best_per_n = {}
for n in range(1, 201):
    best_score = float('inf')
    best_source = None
    for source_name, data in all_sources.items():
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
            improvements.append((n, current_scores[n] - best_score, best_source, best_score))

improvements.sort(key=lambda x: -x[1])
print(f"\nN values where external sources are better: {len(improvements)}")
print("Top 20:")
for n, diff, source, score in improvements[:20]:
    print(f"  N={n}: improvement={diff:.6f} from {source} (score={score:.6f})")
