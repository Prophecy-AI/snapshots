"""
Fast version - just calculate scores without overlap checking for analysis.
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
    """Fast bounding box calculation using numpy."""
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

# Calculate score for each N in each snapshot
all_scores = {}  # {n: {source: score}}

for source_name, df in snapshots.items():
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
        
        if n not in all_scores:
            all_scores[n] = {}
        all_scores[n][source_name] = score

# Find best and worst for each N
print("\nScore range for each N:")
for n in [1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 200]:
    if n in all_scores:
        scores = list(all_scores[n].values())
        best = min(scores)
        worst = max(scores)
        print(f"  N={n}: best={best:.6f}, worst={worst:.6f}, range={worst-best:.6f}")

# Current submission
current_df = pd.read_csv('/home/submission/submission.csv')
current_scores = {}
for n in range(1, 201):
    prefix = f"{n:03d}_"
    n_rows = current_df[current_df['id'].str.startswith(prefix)]
    xs = [parse_value(row['x']) for _, row in n_rows.iterrows()]
    ys = [parse_value(row['y']) for _, row in n_rows.iterrows()]
    degs = [parse_value(row['deg']) for _, row in n_rows.iterrows()]
    side = get_side_fast(xs, ys, degs)
    current_scores[n] = side**2 / n

current_total = sum(current_scores.values())
print(f"\nCurrent submission total: {current_total:.6f}")

# Find N values where current is not the best
improvements = []
for n in range(1, 201):
    if n in all_scores:
        best = min(all_scores[n].values())
        current = current_scores[n]
        if best < current - 0.0001:
            improvements.append((n, current - best, current, best))

improvements.sort(key=lambda x: -x[1])
print(f"\nN values where better scores exist: {len(improvements)}")
print("Top 20:")
for n, diff, current, best in improvements[:20]:
    print(f"  N={n}: current={current:.6f}, best={best:.6f}, diff={diff:.6f}")

total_possible_improvement = sum(x[1] for x in improvements)
print(f"\nTotal possible improvement: {total_possible_improvement:.6f}")
