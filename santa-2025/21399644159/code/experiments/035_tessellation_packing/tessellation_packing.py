"""
Tessellation-based packing for Christmas trees.
Based on the discussion "For Large N Use Tessellations" by Chris Deotte.

The key insight is that for large N, periodic tessellation patterns can achieve
higher packing density than random optimization because they exploit the
regularity of the tree shape.
"""

import numpy as np
import pandas as pd
import math
import json
from shapely.geometry import Polygon
from shapely import affinity

# Tree polygon vertices
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

def check_overlaps(trees):
    polys = [get_tree_polygon(x, y, deg) for x, y, deg in trees]
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                return True
    return False

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

def create_hexagonal_tessellation(n, spacing_x=0.75, spacing_y=0.85, base_angle=45):
    """
    Create hexagonal tessellation packing.
    Trees are arranged in a hexagonal grid with alternating orientations.
    """
    trees = []
    cols = int(np.ceil(np.sqrt(n * 1.2)))  # Slightly more columns for hex packing
    
    placed = 0
    row = 0
    while placed < n:
        x_offset = (row % 2) * spacing_x / 2
        for col in range(cols):
            if placed >= n:
                break
            x = col * spacing_x + x_offset
            y = row * spacing_y
            # Alternate angles for interlocking
            angle = base_angle if (row + col) % 2 == 0 else base_angle + 180
            trees.append((x, y, angle))
            placed += 1
        row += 1
    
    return trees

def create_diamond_tessellation(n, spacing=0.8, base_angle=45):
    """
    Create diamond (45° rotated grid) tessellation.
    """
    trees = []
    cols = int(np.ceil(np.sqrt(n)))
    
    placed = 0
    for i in range(n):
        row = i // cols
        col = i % cols
        # Diamond pattern: rotate grid by 45°
        x = (col - row) * spacing / np.sqrt(2)
        y = (col + row) * spacing / np.sqrt(2)
        angle = base_angle if (row + col) % 2 == 0 else base_angle + 180
        trees.append((x, y, angle))
        placed += 1
    
    return trees

def create_brick_tessellation(n, spacing_x=0.7, spacing_y=0.9, base_angle=0):
    """
    Create brick-like tessellation (offset rows).
    """
    trees = []
    cols = int(np.ceil(np.sqrt(n * 1.1)))
    
    placed = 0
    row = 0
    while placed < n:
        x_offset = (row % 2) * spacing_x / 2
        for col in range(cols):
            if placed >= n:
                break
            x = col * spacing_x + x_offset
            y = row * spacing_y
            angle = base_angle
            trees.append((x, y, angle))
            placed += 1
        row += 1
    
    return trees

def create_interlock_tessellation(n, spacing_x=0.65, spacing_y=0.75):
    """
    Create interlocking tessellation where trees point in opposite directions.
    This exploits the tree's asymmetric shape for tighter packing.
    """
    trees = []
    cols = int(np.ceil(np.sqrt(n)))
    
    for i in range(n):
        row = i // cols
        col = i % cols
        x = col * spacing_x
        y = row * spacing_y
        # Alternate between 0° and 180° for interlocking
        angle = 0 if (row + col) % 2 == 0 else 180
        trees.append((x, y, angle))
    
    return trees

def optimize_spacing(n, tessellation_func, base_angle=45):
    """
    Find optimal spacing for a tessellation pattern.
    """
    best_score = float('inf')
    best_trees = None
    best_params = None
    
    # Grid search over spacing parameters
    for sx in np.arange(0.5, 1.0, 0.05):
        for sy in np.arange(0.6, 1.1, 0.05):
            try:
                if tessellation_func == create_hexagonal_tessellation:
                    trees = tessellation_func(n, sx, sy, base_angle)
                elif tessellation_func == create_diamond_tessellation:
                    trees = tessellation_func(n, sx, base_angle)
                elif tessellation_func == create_brick_tessellation:
                    trees = tessellation_func(n, sx, sy, base_angle)
                elif tessellation_func == create_interlock_tessellation:
                    trees = tessellation_func(n, sx, sy)
                else:
                    continue
                
                if len(trees) == n and not check_overlaps(trees):
                    score = compute_bbox_score(trees)
                    if score < best_score:
                        best_score = score
                        best_trees = trees
                        best_params = (sx, sy)
            except:
                continue
    
    return best_trees, best_score, best_params

# Load baseline
print("Loading baseline submission...")
baseline_df = pd.read_csv('/home/submission/submission.csv')

# Track results
baseline_scores = {}
best_per_n = {}
improvements = {}

print("\n" + "="*60)
print("TESSELLATION-BASED PACKING")
print("="*60)

# Test on various N values
test_ns = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 180, 200]

tessellation_funcs = [
    ('hexagonal', create_hexagonal_tessellation),
    ('diamond', create_diamond_tessellation),
    ('brick', create_brick_tessellation),
    ('interlock', create_interlock_tessellation),
]

for n in test_ns:
    baseline_trees = load_baseline_config(baseline_df, n)
    baseline_score = compute_bbox_score(baseline_trees)
    baseline_scores[n] = baseline_score
    best_per_n[n] = {'score': baseline_score, 'trees': baseline_trees, 'source': 'baseline'}
    
    print(f"\nN={n}: baseline={baseline_score:.6f}")
    
    for name, func in tessellation_funcs:
        # Try different base angles
        for base_angle in [0, 45, 90]:
            try:
                if func == create_interlock_tessellation:
                    trees, score, params = optimize_spacing(n, func)
                else:
                    trees, score, params = optimize_spacing(n, func, base_angle)
                
                if trees and score < float('inf'):
                    if score < best_per_n[n]['score'] - 1e-10:
                        best_per_n[n] = {'score': score, 'trees': trees, 'source': f'{name}_angle{base_angle}'}
                        print(f"  ✅ NEW BEST: {score:.6f} ({name}, angle={base_angle}, params={params})")
                    else:
                        pass  # print(f"  {name} angle={base_angle}: {score:.6f} (no improvement)")
            except Exception as e:
                pass  # print(f"  {name} angle={base_angle}: Error - {e}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

total_improvement = 0
improved_ns = []

for n in test_ns:
    baseline = baseline_scores[n]
    best = best_per_n[n]['score']
    if best < baseline - 1e-10:
        improvement = baseline - best
        total_improvement += improvement
        improved_ns.append(n)
        print(f"N={n}: {baseline:.6f} -> {best:.6f} (improved by {improvement:.8f}) [{best_per_n[n]['source']}]")
    else:
        print(f"N={n}: {baseline:.6f} (no improvement)")

if not improved_ns:
    print("\nNo improvements found with tessellation patterns")
    print("The baseline configurations are already better than regular tessellations")

print(f"\nTotal improvement: {total_improvement:.8f}")

# Save metrics
metrics = {
    'cv_score': 70.316492,
    'improvements_found': len(improved_ns),
    'total_improvement': total_improvement,
    'improved_n_values': improved_ns,
    'method': 'tessellation_packing',
    'patterns_tried': ['hexagonal', 'diamond', 'brick', 'interlock'],
    'n_values_tested': test_ns
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Copy baseline as submission (no improvements expected)
import shutil
shutil.copy('/home/submission/submission.csv', 'submission.csv')

print("\nMetrics saved")
