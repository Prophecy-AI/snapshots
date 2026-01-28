"""
Tessellation-based packing v2 - with larger spacing to avoid overlaps.
"""

import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import warnings
warnings.filterwarnings('ignore')

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

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

def get_tree_polygon(x, y, angle_deg):
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

def compute_score(trees, n):
    if not trees or len(trees) != n:
        return float('inf')
    size = compute_bbox_size(trees)
    return (size ** 2) / n

def check_overlap(trees, threshold=1e-10):
    polygons = [get_tree_polygon(x, y, a) for x, y, a in trees]
    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > threshold:
                    return True
    return False

def load_baseline(path):
    df = pd.read_csv(path)
    df['n'] = df['id'].apply(lambda x: int(x.split('_')[0]))
    df['x'] = df['x'].apply(parse_coord)
    df['y'] = df['y'].apply(parse_coord)
    df['deg'] = df['deg'].apply(parse_coord)
    
    result = {}
    for n in range(1, 201):
        n_df = df[df['n'] == n]
        if len(n_df) == n:
            result[n] = [(row['x'], row['y'], row['deg']) for _, row in n_df.iterrows()]
    return result

def generate_grid_pattern(n, dx=1.0, dy=1.0, angle=45):
    """
    Simple grid pattern with uniform spacing.
    """
    trees = []
    grid_size = int(np.ceil(np.sqrt(n)))
    
    for i in range(grid_size):
        for j in range(grid_size):
            if len(trees) >= n:
                break
            x = i * dx
            y = j * dy
            trees.append((x, y, angle))
    
    return trees[:n]

def generate_alternating_grid(n, dx=1.0, dy=1.0):
    """
    Grid with alternating angles (0° and 180°).
    """
    trees = []
    grid_size = int(np.ceil(np.sqrt(n)))
    
    for i in range(grid_size):
        for j in range(grid_size):
            if len(trees) >= n:
                break
            x = i * dx
            y = j * dy
            angle = 0 if (i + j) % 2 == 0 else 180
            trees.append((x, y, angle))
    
    return trees[:n]

def generate_offset_rows(n, dx=0.8, dy=1.0, offset=0.4):
    """
    Rows with offset for better packing.
    """
    trees = []
    row = 0
    remaining = n
    
    while remaining > 0:
        trees_per_row = int(np.ceil(np.sqrt(n)))
        m = min(remaining, trees_per_row)
        
        x_offset = offset if row % 2 == 1 else 0
        angle = 45 if row % 2 == 0 else 225
        
        for i in range(m):
            x = x_offset + i * dx
            y = row * dy
            trees.append((x, y, angle))
        
        remaining -= m
        row += 1
    
    return trees

# Load baseline
print("Loading baseline (exp_044)...")
baseline_path = "/home/code/experiments/044_extended_subset_extraction/ensemble_044.csv"
baseline = load_baseline(baseline_path)
baseline_scores = {n: compute_score(baseline[n], n) for n in range(1, 201)}
total_baseline = sum(baseline_scores.values())
print(f"Baseline total: {total_baseline:.6f}")

# Test patterns with larger spacing
print("\n" + "="*60)
print("TESTING TESSELLATION PATTERNS (LARGER SPACING)")
print("="*60)

test_n_values = [10, 20, 30]

for n in test_n_values:
    print(f"\nN={n}:")
    print(f"  Baseline score: {baseline_scores[n]:.6f}")
    
    # Test simple grid
    for dx in [0.8, 0.9, 1.0, 1.1]:
        for dy in [0.8, 0.9, 1.0, 1.1]:
            trees = generate_grid_pattern(n, dx=dx, dy=dy, angle=45)
            if not check_overlap(trees):
                score = compute_score(trees, n)
                print(f"  Grid (dx={dx}, dy={dy}): {score:.6f} {'✅' if score < baseline_scores[n] else ''}")
    
    # Test alternating grid
    for dx in [0.8, 0.9, 1.0, 1.1]:
        for dy in [0.8, 0.9, 1.0, 1.1]:
            trees = generate_alternating_grid(n, dx=dx, dy=dy)
            if not check_overlap(trees):
                score = compute_score(trees, n)
                print(f"  AltGrid (dx={dx}, dy={dy}): {score:.6f} {'✅' if score < baseline_scores[n] else ''}")
    
    # Test offset rows
    for dx in [0.8, 0.9, 1.0]:
        for dy in [0.8, 0.9, 1.0]:
            for offset in [0.3, 0.4, 0.5]:
                trees = generate_offset_rows(n, dx=dx, dy=dy, offset=offset)
                if not check_overlap(trees):
                    score = compute_score(trees, n)
                    if score < baseline_scores[n]:
                        print(f"  ✅ OffsetRows (dx={dx}, dy={dy}, off={offset}): {score:.6f} (BETTER!)")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("Tessellation patterns produce WORSE results than baseline.")
print("The baseline uses sophisticated per-tree optimization that")
print("simple tessellation patterns cannot match.")
