"""
Row-based constructive algorithm for tree packing.
Implements zaburo's approach: alternating angles (0° and 180°) with row offsets.
"""

import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import pandas as pd
import math
import warnings
warnings.filterwarnings('ignore')

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

# Tree dimensions
TREE_WIDTH = 0.7  # max(TX) - min(TX) = 0.35 - (-0.35) = 0.7
TREE_HEIGHT = 1.0  # max(TY) - min(TY) = 0.8 - (-0.2) = 1.0

def parse_coord(val):
    if isinstance(val, str):
        if val.startswith('s'):
            return float(val[1:])
        return float(val)
    return float(val)

def get_tree_vertices(x, y, angle_deg):
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rx = TX * cos_a - TY * sin_a
    ry = TX * sin_a + TY * cos_a
    return rx + x, ry + y

def create_tree_polygon(x, y, angle_deg):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = affinity.rotate(poly, angle_deg, origin=(0, 0))
    poly = affinity.translate(poly, x, y)
    return poly

def compute_bbox_size(trees):
    all_x, all_y = [], []
    for x, y, angle in trees:
        vx, vy = get_tree_vertices(x, y, angle)
        all_x.extend(vx)
        all_y.extend(vy)
    if not all_x:
        return float('inf')
    return max(max(all_x) - min(all_x), max(all_y) - min(all_y))

def compute_score_for_n(trees, n):
    size = compute_bbox_size(trees)
    return (size ** 2) / n

def check_overlap(trees, threshold=1e-10):
    polygons = [create_tree_polygon(x, y, a) for x, y, a in trees]
    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > threshold:
                    return True
    return False

def load_submission(path):
    df = pd.read_csv(path)
    if 'id' in df.columns:
        df['n'] = df['id'].apply(lambda x: int(x.split('_')[0]))
        df['i'] = df['id'].apply(lambda x: int(x.split('_')[1]))
    df['x'] = df['x'].apply(parse_coord)
    df['y'] = df['y'].apply(parse_coord)
    if 'deg' in df.columns:
        df['deg'] = df['deg'].apply(parse_coord)
    else:
        df['deg'] = 0
    
    result = {}
    for n in range(1, 201):
        n_df = df[df['n'] == n]
        if len(n_df) == n:
            trees = [(row['x'], row['y'], row['deg']) for _, row in n_df.iterrows()]
            result[n] = trees
    return result

def build_row_based_solution(n, row_width=None, x_spacing=0.7, y_spacing=0.5, x_offset=0.35):
    """
    Build solution using row-based placement with alternating angles.
    
    Args:
        n: Number of trees
        row_width: Trees per row (if None, use sqrt(n))
        x_spacing: Horizontal spacing between trees
        y_spacing: Vertical spacing between rows
        x_offset: Horizontal offset for odd rows
    """
    if row_width is None:
        row_width = max(1, int(math.sqrt(n)))
    
    trees = []
    remaining = n
    row = 0
    
    while remaining > 0:
        # Alternate row angles (0° and 180°)
        angle = 0 if row % 2 == 0 else 180
        
        # Offset every other row
        offset = 0 if row % 2 == 0 else x_offset
        
        # Trees in this row
        m = min(remaining, row_width)
        
        # Y position
        y = row * y_spacing
        
        for i in range(m):
            x = offset + i * x_spacing
            trees.append((x, y, angle))
        
        remaining -= m
        row += 1
    
    return trees

def optimize_row_parameters(n, baseline_score):
    """
    Try different row parameters to find the best configuration.
    """
    best_score = float('inf')
    best_config = None
    best_params = None
    
    # Try different row widths
    for row_width in range(1, n + 1):
        # Try different spacings
        for x_spacing in np.linspace(0.5, 1.0, 11):
            for y_spacing in np.linspace(0.3, 1.0, 15):
                for x_offset in np.linspace(0, 0.5, 11):
                    trees = build_row_based_solution(n, row_width, x_spacing, y_spacing, x_offset)
                    
                    # Check for overlaps
                    if check_overlap(trees):
                        continue
                    
                    score = compute_score_for_n(trees, n)
                    if score < best_score:
                        best_score = score
                        best_config = trees
                        best_params = (row_width, x_spacing, y_spacing, x_offset)
    
    return best_config, best_score, best_params

# Load baseline
print("Loading baseline (exp_039)...")
baseline_path = "/home/code/experiments/039_per_n_analysis/safe_ensemble.csv"
baseline = load_submission(baseline_path)

# Compute baseline scores
baseline_scores = {n: compute_score_for_n(baseline[n], n) for n in range(1, 201)}
total_baseline = sum(baseline_scores.values())
print(f"Baseline total: {total_baseline:.6f}")

# Test on small N values first
print("\n" + "="*60)
print("TESTING ROW-BASED CONSTRUCTIVE ON SMALL N VALUES")
print("="*60)

test_n_values = [10, 15, 20, 25, 30]
results = {}

for n in test_n_values:
    print(f"\nN={n}: Optimizing row parameters...")
    best_config, best_score, best_params = optimize_row_parameters(n, baseline_scores[n])
    
    if best_config is None:
        print(f"  No valid configuration found")
        continue
    
    improvement = baseline_scores[n] - best_score
    results[n] = {
        'baseline': baseline_scores[n],
        'row_based': best_score,
        'improvement': improvement,
        'params': best_params
    }
    
    if improvement > 0:
        print(f"  ✅ IMPROVED: {baseline_scores[n]:.6f} -> {best_score:.6f} (improvement: {improvement:.6f})")
        print(f"     Params: row_width={best_params[0]}, x_spacing={best_params[1]:.2f}, y_spacing={best_params[2]:.2f}, x_offset={best_params[3]:.2f}")
    else:
        print(f"  ❌ WORSE: {baseline_scores[n]:.6f} -> {best_score:.6f} (diff: {improvement:.6f})")
        print(f"     Params: row_width={best_params[0]}, x_spacing={best_params[1]:.2f}, y_spacing={best_params[2]:.2f}, x_offset={best_params[3]:.2f}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

improvements = [r['improvement'] for r in results.values() if r['improvement'] > 0]
if improvements:
    print(f"Improvements found: {len(improvements)}/{len(results)}")
    print(f"Total improvement: {sum(improvements):.6f}")
else:
    print("No improvements found - row-based constructive produces WORSE results than baseline")
    print("This confirms the baseline is at a strong local optimum")
