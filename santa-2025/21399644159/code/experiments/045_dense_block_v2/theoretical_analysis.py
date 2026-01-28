"""
Theoretical Analysis of Tree Packing

The tree polygon has a specific area. For N trees, the minimum possible
bounding box area is N * tree_area (if trees could be packed perfectly).

The score is side^2 / N, so the minimum score is:
- If trees pack perfectly: tree_area (constant for all N)
- In practice: depends on how well trees can interlock

Let's calculate the theoretical minimum and compare to baseline.
"""

import numpy as np
import pandas as pd
import math
from shapely.geometry import Polygon
from shapely import affinity

TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def parse_s_value(s):
    if isinstance(s, str) and s.startswith('s'):
        return float(s[1:])
    return float(s)

def get_tree_polygon(x, y, deg):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = affinity.rotate(poly, deg, origin=(0, 0))
    poly = affinity.translate(poly, x, y)
    return poly

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

def load_config(df, n):
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

# Calculate tree area
tree_poly = get_tree_polygon(0, 0, 0)
tree_area = tree_poly.area
print(f"Tree area: {tree_area:.6f}")

# Calculate minimum bounding box for a single tree at different angles
print("\nMinimum bounding box for single tree at different angles:")
min_single_score = float('inf')
best_angle = 0
for deg in range(0, 360, 1):
    x0, x1, y0, y1 = get_tree_bounds(0, 0, deg)
    width = x1 - x0
    height = y1 - y0
    side = max(width, height)
    score = side**2
    if score < min_single_score:
        min_single_score = score
        best_angle = deg

print(f"Best angle for N=1: {best_angle}° with score {min_single_score:.6f}")

# Load baseline
baseline_df = pd.read_csv('/home/submission/submission.csv')

baseline_scores = {}
for n in range(1, 201):
    trees = load_config(baseline_df, n)
    baseline_scores[n] = compute_bbox_score(trees)

# Theoretical analysis
print("\n" + "="*70)
print("THEORETICAL ANALYSIS")
print("="*70)

# For each N, calculate:
# 1. Baseline score
# 2. Theoretical minimum (if trees could pack perfectly)
# 3. Gap

print(f"\n{'N':>4} {'Baseline':>12} {'Theo Min':>12} {'Gap':>12} {'Efficiency':>12}")
print("-"*56)

total_baseline = 0
total_theo_min = 0

for n in range(1, 201):
    baseline = baseline_scores[n]
    
    # Theoretical minimum: if trees pack perfectly, area = N * tree_area
    # Score = side^2 / N = area / N = tree_area (constant)
    # But this ignores that trees can't overlap
    
    # A more realistic theoretical minimum:
    # For N trees, minimum side = sqrt(N * tree_area / packing_efficiency)
    # Typical packing efficiency for irregular shapes is ~0.7-0.8
    packing_eff = 0.75
    theo_min_area = n * tree_area / packing_eff
    theo_min_side = math.sqrt(theo_min_area)
    theo_min_score = theo_min_side**2 / n
    
    gap = baseline - theo_min_score
    efficiency = theo_min_score / baseline * 100
    
    total_baseline += baseline
    total_theo_min += theo_min_score
    
    if n <= 20 or n % 20 == 0:
        print(f"{n:>4} {baseline:>12.6f} {theo_min_score:>12.6f} {gap:>12.6f} {efficiency:>11.1f}%")

print("-"*56)
print(f"Total: {total_baseline:.6f} vs theoretical {total_theo_min:.6f}")
print(f"Gap: {total_baseline - total_theo_min:.6f}")

# The gap to target
target = 68.870074
print(f"\nTarget score: {target:.6f}")
print(f"Current score: {total_baseline:.6f}")
print(f"Gap to target: {total_baseline - target:.6f}")
print(f"Theoretical minimum: {total_theo_min:.6f}")
print(f"Target vs theoretical: {target - total_theo_min:.6f}")

# Is the target achievable?
if target < total_theo_min:
    print("\n⚠️ TARGET IS BELOW THEORETICAL MINIMUM!")
else:
    print(f"\n✅ Target is {(target - total_theo_min) / total_theo_min * 100:.1f}% above theoretical minimum")
    print(f"   This means the target IS theoretically achievable")
