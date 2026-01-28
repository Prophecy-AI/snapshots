"""
Tessellation-based packing for tree placement.
Generate solutions using repeating patterns that tile the plane.
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

# Tree dimensions
TREE_WIDTH = 0.7  # max(TX) - min(TX)
TREE_HEIGHT = 1.0  # max(TY) - min(TY)

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

def generate_tessellation_pattern1(n, dx=0.7, dy=0.5, angle1=0, angle2=180):
    """
    Pattern 1: Alternating rows with 0° and 180° trees.
    Trees in adjacent rows interlock.
    """
    trees = []
    row = 0
    remaining = n
    
    while remaining > 0:
        # Determine trees per row
        trees_per_row = int(np.ceil(np.sqrt(n)))
        m = min(remaining, trees_per_row)
        
        # Alternate angle between rows
        angle = angle1 if row % 2 == 0 else angle2
        
        # Offset every other row for interlocking
        x_offset = 0 if row % 2 == 0 else dx / 2
        
        for i in range(m):
            x = x_offset + i * dx
            y = row * dy
            trees.append((x, y, angle))
        
        remaining -= m
        row += 1
    
    return trees

def generate_tessellation_pattern2(n, spacing=0.6):
    """
    Pattern 2: Hexagonal packing with alternating angles.
    """
    trees = []
    row = 0
    remaining = n
    
    while remaining > 0:
        trees_per_row = int(np.ceil(np.sqrt(n * 1.15)))  # Hexagonal has more trees per row
        m = min(remaining, trees_per_row)
        
        # Hexagonal offset
        x_offset = (row % 2) * spacing / 2
        
        # Alternate angles
        angle = 45 if row % 2 == 0 else 225
        
        for i in range(m):
            x = x_offset + i * spacing
            y = row * spacing * 0.866  # sqrt(3)/2 for hexagonal
            trees.append((x, y, angle))
        
        remaining -= m
        row += 1
    
    return trees

def generate_tessellation_pattern3(n, dx=0.75, dy=0.6):
    """
    Pattern 3: Blue-Pink interlock pattern from "Why Not" kernel.
    Blue trees point up (0°), Pink trees point down (180°).
    """
    trees = []
    remaining = n
    
    # Compute grid size
    grid_size = int(np.ceil(np.sqrt(n)))
    
    for i in range(grid_size):
        for j in range(grid_size):
            if remaining <= 0:
                break
            
            # Checkerboard pattern for angles
            angle = 0 if (i + j) % 2 == 0 else 180
            
            x = i * dx
            y = j * dy
            
            trees.append((x, y, angle))
            remaining -= 1
    
    return trees[:n]

def optimize_tessellation_params(n, baseline_score, pattern_func, param_ranges):
    """
    Search for optimal tessellation parameters.
    """
    best_score = float('inf')
    best_trees = None
    best_params = None
    
    for params in param_ranges:
        trees = pattern_func(n, **params)
        
        if check_overlap(trees):
            continue
        
        score = compute_score(trees, n)
        if score < best_score:
            best_score = score
            best_trees = trees
            best_params = params
    
    return best_trees, best_score, best_params

# Load baseline
print("Loading baseline (exp_044)...")
baseline_path = "/home/code/experiments/044_extended_subset_extraction/ensemble_044.csv"
baseline = load_baseline(baseline_path)
baseline_scores = {n: compute_score(baseline[n], n) for n in range(1, 201)}
total_baseline = sum(baseline_scores.values())
print(f"Baseline total: {total_baseline:.6f}")

# Test tessellation patterns on small N values
print("\n" + "="*60)
print("TESTING TESSELLATION PATTERNS ON SMALL N VALUES")
print("="*60)

test_n_values = [10, 20, 30, 50]

for n in test_n_values:
    print(f"\nN={n}:")
    print(f"  Baseline score: {baseline_scores[n]:.6f}")
    
    # Test Pattern 1: Alternating rows
    for dx in [0.6, 0.7, 0.8]:
        for dy in [0.4, 0.5, 0.6]:
            trees = generate_tessellation_pattern1(n, dx=dx, dy=dy)
            if not check_overlap(trees):
                score = compute_score(trees, n)
                if score < baseline_scores[n]:
                    print(f"  ✅ Pattern1 (dx={dx}, dy={dy}): {score:.6f} (BETTER!)")
                else:
                    print(f"  ❌ Pattern1 (dx={dx}, dy={dy}): {score:.6f}")
    
    # Test Pattern 2: Hexagonal
    for spacing in [0.5, 0.6, 0.7, 0.8]:
        trees = generate_tessellation_pattern2(n, spacing=spacing)
        if not check_overlap(trees):
            score = compute_score(trees, n)
            if score < baseline_scores[n]:
                print(f"  ✅ Pattern2 (spacing={spacing}): {score:.6f} (BETTER!)")
            else:
                print(f"  ❌ Pattern2 (spacing={spacing}): {score:.6f}")
    
    # Test Pattern 3: Blue-Pink checkerboard
    for dx in [0.6, 0.7, 0.8]:
        for dy in [0.5, 0.6, 0.7]:
            trees = generate_tessellation_pattern3(n, dx=dx, dy=dy)
            if not check_overlap(trees):
                score = compute_score(trees, n)
                if score < baseline_scores[n]:
                    print(f"  ✅ Pattern3 (dx={dx}, dy={dy}): {score:.6f} (BETTER!)")
                else:
                    print(f"  ❌ Pattern3 (dx={dx}, dy={dy}): {score:.6f}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("Testing tessellation patterns to see if any beat the baseline...")
