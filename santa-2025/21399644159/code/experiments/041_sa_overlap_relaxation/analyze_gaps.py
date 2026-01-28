"""
Analyze per-N scores to identify where improvements might be possible.
"""

import numpy as np
import pandas as pd
import math
import json

TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def parse_s_value(s):
    if isinstance(s, str) and s.startswith('s'):
        return float(s[1:])
    return float(s)

def get_tree_bounds(x, y, deg):
    rad = math.radians(deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    rx = TX * cos_a - TY * sin_a + x
    ry = TX * sin_a + TY * cos_a + y
    return rx.min(), rx.max(), ry.min(), ry.max()

def compute_bbox_score(trees):
    if not trees:
        return float('inf')
    
    minx = miny = float('inf')
    maxx = maxy = float('-inf')
    
    for x, y, deg in trees:
        x0, x1, y0, y1 = get_tree_bounds(x, y, deg)
        minx = min(minx, x0)
        maxx = max(maxx, x1)
        miny = min(miny, y0)
        maxy = max(maxy, y1)
    
    side = max(maxx - minx, maxy - miny)
    n = len(trees)
    return side**2 / n

def load_baseline_config(df, n):
    pattern = f'{n:03d}_'
    cfg = df[df['id'].str.startswith(pattern)].copy()
    cfg['tree_idx'] = cfg['id'].apply(lambda x: int(x.split('_')[1]))
    cfg = cfg.sort_values('tree_idx')
    
    trees = []
    for _, row in cfg.iterrows():
        x = parse_s_value(row['x'])
        y = parse_s_value(row['y'])
        deg = parse_s_value(row['deg'])
        trees.append((x, y, deg))
    return trees

# Load baseline
print("Loading baseline...")
baseline_df = pd.read_csv('/home/submission/submission.csv')

# Calculate per-N scores
per_n_scores = {}
for n in range(1, 201):
    trees = load_baseline_config(baseline_df, n)
    per_n_scores[n] = compute_bbox_score(trees)

# Calculate theoretical lower bounds
# For N trees, the minimum area is N * tree_area
# Tree area is approximately 0.35 * 0.8 = 0.28 (rough estimate)
tree_area = 0.28  # Approximate

print("\nPer-N Analysis:")
print("="*60)
print(f"{'N':>4} {'Score':>10} {'Side':>10} {'Efficiency':>10}")
print("="*60)

# Sort by score contribution (highest first)
sorted_ns = sorted(per_n_scores.keys(), key=lambda n: per_n_scores[n], reverse=True)

for n in sorted_ns[:30]:  # Top 30 contributors
    score = per_n_scores[n]
    side = math.sqrt(score * n)
    # Efficiency = how much of the bounding box is used
    efficiency = (n * tree_area) / (side * side)
    print(f"{n:>4} {score:>10.6f} {side:>10.4f} {efficiency:>10.2%}")

print("\n" + "="*60)
print("Total score:", sum(per_n_scores.values()))
print("Target score: 68.870074")
print("Gap:", sum(per_n_scores.values()) - 68.870074)

# Find N values with lowest efficiency (most room for improvement)
print("\n" + "="*60)
print("N values with LOWEST efficiency (most room for improvement):")
print("="*60)

efficiencies = {}
for n in range(1, 201):
    score = per_n_scores[n]
    side = math.sqrt(score * n)
    efficiencies[n] = (n * tree_area) / (side * side)

sorted_by_eff = sorted(efficiencies.keys(), key=lambda n: efficiencies[n])

for n in sorted_by_eff[:20]:
    score = per_n_scores[n]
    side = math.sqrt(score * n)
    print(f"N={n:>3}: score={score:.6f}, side={side:.4f}, efficiency={efficiencies[n]:.2%}")
