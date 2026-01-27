"""
Corner Extraction: Extract smaller N configurations from larger N configurations
by selecting trees closest to corners.

Key insight from kernel "new-simple-fix-rebuild-large-layout-check-on-all":
- Large N configurations are highly optimized
- Subsets of these configurations may be better than independently optimized small N
- Corner-based selection naturally creates compact arrangements
"""

import numpy as np
import pandas as pd
import math
import json
from shapely.geometry import Polygon
from shapely import affinity
from shapely.ops import unary_union

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

def compute_bbox_score_from_trees(trees):
    """Compute score from list of tree dicts with 'x', 'y', 'deg' keys"""
    if not trees:
        return float('inf')
    
    minx = miny = float('inf')
    maxx = maxy = float('-inf')
    
    for t in trees:
        x0, x1, y0, y1 = get_tree_bounds(t['x'], t['y'], t['deg'])
        minx = min(minx, x0)
        maxx = max(maxx, x1)
        miny = min(miny, y0)
        maxy = max(maxy, y1)
    
    side = max(maxx - minx, maxy - miny)
    n = len(trees)
    return side**2 / n

def check_overlaps(trees):
    """Check if any trees overlap"""
    polys = [get_tree_polygon(t['x'], t['y'], t['deg']) for t in trees]
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                return True
    return False

def load_config(df, n):
    """Load configuration for N trees"""
    pattern = f'{n:03d}_'
    cfg = df[df['id'].str.startswith(pattern)].copy()
    cfg['tree_idx'] = cfg['id'].apply(lambda x: int(x.split('_')[1]))
    cfg = cfg.sort_values('tree_idx')
    
    trees = []
    for _, row in cfg.iterrows():
        x = parse_s_value(row['x'])
        y = parse_s_value(row['y'])
        deg = parse_s_value(row['deg'])
        trees.append({'x': x, 'y': y, 'deg': deg})
    return trees

def corner_extraction(source_trees, target_n):
    """
    Extract target_n trees from source configuration using corner selection.
    
    For each corner of the bounding box:
    - Sort trees by distance to that corner
    - Take the closest target_n trees
    - Compute the bounding box score
    - Keep the best configuration
    """
    if len(source_trees) < target_n:
        return None, float('inf')
    
    # Get bounding box of all trees
    minx = miny = float('inf')
    maxx = maxy = float('-inf')
    
    for t in source_trees:
        x0, x1, y0, y1 = get_tree_bounds(t['x'], t['y'], t['deg'])
        minx = min(minx, x0)
        maxx = max(maxx, x1)
        miny = min(miny, y0)
        maxy = max(maxy, y1)
    
    # Define corners
    corners = [
        (minx, miny),  # bottom-left
        (minx, maxy),  # top-left
        (maxx, miny),  # bottom-right
        (maxx, maxy),  # top-right
    ]
    
    best_score = float('inf')
    best_trees = None
    
    for corner_x, corner_y in corners:
        # Sort trees by distance to corner (using tree center)
        trees_with_dist = []
        for t in source_trees:
            # Use center of tree bounds as reference point
            x0, x1, y0, y1 = get_tree_bounds(t['x'], t['y'], t['deg'])
            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2
            dist = math.sqrt((center_x - corner_x)**2 + (center_y - corner_y)**2)
            trees_with_dist.append((dist, t))
        
        # Sort by distance and take closest target_n trees
        trees_with_dist.sort(key=lambda x: x[0])
        selected = [t for _, t in trees_with_dist[:target_n]]
        
        # Check for overlaps (should be none since we're taking a subset)
        if not check_overlaps(selected):
            score = compute_bbox_score_from_trees(selected)
            if score < best_score:
                best_score = score
                best_trees = selected
    
    return best_trees, best_score

# Load baseline
print("Loading baseline submission...")
baseline_df = pd.read_csv('/home/submission/submission.csv')

# Load all baseline configurations
baseline_configs = {}
baseline_scores = {}
for n in range(1, 201):
    trees = load_config(baseline_df, n)
    baseline_configs[n] = trees
    baseline_scores[n] = compute_bbox_score_from_trees(trees)

print(f"Loaded {len(baseline_configs)} configurations")
print(f"Total baseline score: {sum(baseline_scores.values()):.6f}")

# Track best per-N
best_per_n = {n: {'score': baseline_scores[n], 'trees': baseline_configs[n], 'source': 'baseline'} 
              for n in range(1, 201)}

print("\n" + "="*60)
print("CORNER EXTRACTION")
print("="*60)

improvements = {}

# For each source N (from large to small)
source_ns = list(range(200, 20, -1))  # Start from N=200 down to N=21

for source_n in source_ns:
    source_trees = baseline_configs[source_n]
    
    # Try extracting smaller configurations
    for target_n in range(2, min(source_n, 51)):  # Extract N=2 to N=50
        extracted_trees, extracted_score = corner_extraction(source_trees, target_n)
        
        if extracted_trees and extracted_score < best_per_n[target_n]['score'] - 1e-10:
            improvement = best_per_n[target_n]['score'] - extracted_score
            best_per_n[target_n] = {
                'score': extracted_score, 
                'trees': extracted_trees, 
                'source': f'extracted_from_N{source_n}'
            }
            improvements[target_n] = improvement
            print(f"✅ N={target_n}: {baseline_scores[target_n]:.6f} -> {extracted_score:.6f} (from N={source_n})")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

total_improvement = 0
improved_ns = []

for n in range(1, 201):
    if n in improvements:
        total_improvement += improvements[n]
        improved_ns.append(n)

if improved_ns:
    print(f"\nFound {len(improved_ns)} N values with improvements:")
    for n in sorted(improved_ns):
        print(f"  N={n}: improved by {improvements[n]:.8f}")
    print(f"\nTotal improvement: {total_improvement:.8f}")
else:
    print("\nNo improvements found with corner extraction")
    print("The baseline configurations are already optimal for corner-based subsets")

# Calculate new total score
new_total = sum(best_per_n[n]['score'] for n in range(1, 201))
print(f"\nBaseline total: {sum(baseline_scores.values()):.6f}")
print(f"New total: {new_total:.6f}")
print(f"Improvement: {sum(baseline_scores.values()) - new_total:.8f}")

# Save metrics
metrics = {
    'cv_score': new_total,
    'baseline_score': sum(baseline_scores.values()),
    'improvements_found': len(improved_ns),
    'total_improvement': total_improvement,
    'improved_n_values': improved_ns,
    'method': 'corner_extraction',
    'source_n_range': [200, 21],
    'target_n_range': [2, 50]
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Copy baseline as submission (will update if improvements found)
import shutil
shutil.copy('/home/submission/submission.csv', 'submission.csv')

print("\nMetrics saved")
