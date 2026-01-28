"""
Asymmetric Solution Generation

Based on the discussion "Why winning solutions will be Asymmetric":
- Symmetric solutions have theoretical limits
- Asymmetric arrangements can achieve better packing density
- Use golden angle (137.5°) for natural asymmetry

This generates solutions FROM SCRATCH, not optimizing existing ones.
"""

import numpy as np
import pandas as pd
import math
import json
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

def check_overlaps(trees):
    polys = [get_tree_polygon(x, y, deg) for x, y, deg in trees]
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                return True
    return False

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

def generate_golden_spiral(n, scale=0.4):
    """Generate asymmetric placement using golden angle spiral."""
    golden_angle = 137.5077640500378  # degrees
    trees = []
    
    for i in range(n):
        # Spiral placement
        angle = i * golden_angle
        r = scale * math.sqrt(i + 1)
        x = r * math.cos(math.radians(angle))
        y = r * math.sin(math.radians(angle))
        
        # Tree rotation follows placement angle for asymmetry
        tree_angle = (angle * 1.5) % 360
        
        trees.append((x, y, tree_angle))
    
    return trees

def generate_fibonacci_lattice(n, scale=0.5):
    """Generate placement using Fibonacci lattice."""
    phi = (1 + math.sqrt(5)) / 2  # Golden ratio
    trees = []
    
    for i in range(n):
        # Fibonacci lattice
        theta = 2 * math.pi * i / phi
        r = scale * math.sqrt(i + 1)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        
        # Asymmetric angle
        tree_angle = (i * 137.5) % 360
        
        trees.append((x, y, tree_angle))
    
    return trees

def generate_phyllotaxis(n, scale=0.35):
    """Generate placement using phyllotaxis pattern (sunflower seeds)."""
    golden_angle = 137.5077640500378
    trees = []
    
    for i in range(n):
        angle = i * golden_angle
        r = scale * math.sqrt(i)
        x = r * math.cos(math.radians(angle))
        y = r * math.sin(math.radians(angle))
        
        # Alternate tree angles for interlocking
        tree_angle = 45 + (i % 2) * 180
        
        trees.append((x, y, tree_angle))
    
    return trees

def repair_overlaps_greedy(trees, max_attempts=500):
    """Greedily repair overlaps by moving trees apart."""
    trees = [list(t) for t in trees]
    n = len(trees)
    
    for attempt in range(max_attempts):
        polys = [get_tree_polygon(t[0], t[1], t[2]) for t in trees]
        
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                    dx = trees[j][0] - trees[i][0]
                    dy = trees[j][1] - trees[i][1]
                    dist = math.sqrt(dx*dx + dy*dy) + 1e-10
                    
                    move_dist = 0.02
                    trees[j][0] += dx/dist * move_dist
                    trees[j][1] += dy/dist * move_dist
                    moved = True
        
        if not moved:
            break
    
    return [(t[0], t[1], t[2]) for t in trees]

# Load baseline
print("Loading baseline...")
baseline_df = pd.read_csv('/home/submission/submission.csv')

baseline_scores = {}
baseline_configs = {}
for n in range(1, 201):
    trees = load_config(baseline_df, n)
    baseline_configs[n] = trees
    baseline_scores[n] = compute_bbox_score(trees)

print(f"Baseline total: {sum(baseline_scores.values()):.6f}")

# Test asymmetric generation
print("\nTesting asymmetric generation...")
improvements = {}

test_ns = [10, 15, 20, 25, 30, 40, 50]

generators = [
    ('golden_spiral', generate_golden_spiral),
    ('fibonacci', generate_fibonacci_lattice),
    ('phyllotaxis', generate_phyllotaxis),
]

for n in test_ns:
    print(f"\nN={n}: baseline={baseline_scores[n]:.6f}")
    
    best_score = baseline_scores[n]
    best_method = None
    
    for method_name, generator in generators:
        # Try different scales
        for scale in [0.3, 0.35, 0.4, 0.45, 0.5]:
            trees = generator(n, scale)
            trees = repair_overlaps_greedy(trees)
            
            if not check_overlaps(trees):
                score = compute_bbox_score(trees)
                if score < best_score - 1e-10:
                    best_score = score
                    best_method = f"{method_name}_scale{scale}"
    
    if best_method:
        improvements[n] = baseline_scores[n] - best_score
        print(f"  ✅ IMPROVEMENT: {improvements[n]:.8f} ({best_method})")
    else:
        print(f"  ❌ No improvement")

total_improvement = sum(improvements.values())
print(f"\nTotal improvement: {total_improvement:.8f}")
print(f"Improved N values: {list(improvements.keys())}")

# Save metrics
metrics = {
    'cv_score': 70.316492,
    'improvements_found': len(improvements),
    'total_improvement': total_improvement,
    'improved_n_values': list(improvements.keys()),
    'method': 'asymmetric_generation'
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\nDone!")
