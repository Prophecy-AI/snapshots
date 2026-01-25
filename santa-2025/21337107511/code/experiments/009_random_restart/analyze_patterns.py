"""
Analyze the baseline solution patterns to understand how it achieves tight packing.
"""

import numpy as np
import pandas as pd
import math
from shapely import Polygon
from shapely.affinity import rotate, translate
import json

# Tree polygon vertices
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]
TREE_COORDS = list(zip(TX, TY))

def get_tree_polygon(x, y, angle):
    poly = Polygon(TREE_COORDS)
    poly = rotate(poly, angle, origin=(0, 0))
    poly = translate(poly, x, y)
    return poly

def calculate_bounding_box(trees):
    all_coords = []
    for x, y, angle in trees:
        poly = get_tree_polygon(x, y, angle)
        all_coords.extend(poly.exterior.coords)
    
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    
    return max(max(xs) - min(xs), max(ys) - min(ys))

def load_baseline(csv_path):
    df = pd.read_csv(csv_path)
    
    solutions = {}
    for n in range(1, 201):
        n_df = df[df['id'].str.startswith(f'{n:03d}_')]
        trees = []
        for _, row in n_df.iterrows():
            x = float(str(row['x']).replace('s', ''))
            y = float(str(row['y']).replace('s', ''))
            angle = float(str(row['deg']).replace('s', ''))
            trees.append([x, y, angle])
        solutions[n] = trees
    
    return solutions

def analyze_packing_density(trees, n):
    """Analyze how efficiently trees are packed."""
    # Calculate bounding box
    all_coords = []
    for x, y, angle in trees:
        poly = get_tree_polygon(x, y, angle)
        all_coords.extend(poly.exterior.coords)
    
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    
    bbox_width = max(xs) - min(xs)
    bbox_height = max(ys) - min(ys)
    bbox_area = bbox_width * bbox_height
    
    # Calculate total tree area
    tree_poly = Polygon(TREE_COORDS)
    single_tree_area = tree_poly.area
    total_tree_area = n * single_tree_area
    
    # Packing density
    density = total_tree_area / bbox_area
    
    return {
        'bbox_width': bbox_width,
        'bbox_height': bbox_height,
        'bbox_area': bbox_area,
        'total_tree_area': total_tree_area,
        'packing_density': density
    }

def analyze_angle_patterns(trees):
    """Analyze the angle distribution."""
    angles = [t[2] for t in trees]
    
    # Round to nearest 45 degrees
    rounded = [round(a / 45) * 45 % 360 for a in angles]
    
    # Count occurrences
    counts = {}
    for a in rounded:
        counts[a] = counts.get(a, 0) + 1
    
    return counts

def analyze_position_patterns(trees):
    """Analyze position patterns."""
    xs = [t[0] for t in trees]
    ys = [t[1] for t in trees]
    
    # Calculate pairwise distances
    distances = []
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            d = math.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            distances.append(d)
    
    return {
        'min_distance': min(distances) if distances else 0,
        'max_distance': max(distances) if distances else 0,
        'mean_distance': np.mean(distances) if distances else 0,
        'std_distance': np.std(distances) if distances else 0
    }

if __name__ == "__main__":
    baseline_path = "/home/nonroot/snapshots/santa-2025/21337353543/submission/submission.csv"
    print(f"Loading baseline from {baseline_path}")
    baseline_solutions = load_baseline(baseline_path)
    
    print("\n" + "=" * 60)
    print("BASELINE PATTERN ANALYSIS")
    print("=" * 60)
    
    # Analyze selected N values
    for n in [2, 3, 4, 5, 10, 20, 50, 100]:
        trees = baseline_solutions[n]
        
        print(f"\nN={n}:")
        
        # Packing density
        density = analyze_packing_density(trees, n)
        print(f"  Packing density: {density['packing_density']:.4f}")
        print(f"  BBox: {density['bbox_width']:.4f} x {density['bbox_height']:.4f}")
        
        # Angle patterns
        angles = analyze_angle_patterns(trees)
        print(f"  Angle distribution: {angles}")
        
        # Position patterns
        positions = analyze_position_patterns(trees)
        print(f"  Min tree distance: {positions['min_distance']:.6f}")
        print(f"  Mean tree distance: {positions['mean_distance']:.4f}")
    
    # Calculate theoretical minimum for comparison
    print("\n" + "=" * 60)
    print("THEORETICAL ANALYSIS")
    print("=" * 60)
    
    tree_poly = Polygon(TREE_COORDS)
    tree_area = tree_poly.area
    print(f"Single tree area: {tree_area:.6f}")
    
    # For N trees with perfect packing (density ~0.9 for irregular shapes)
    for n in [5, 10, 20, 50, 100]:
        trees = baseline_solutions[n]
        side = calculate_bounding_box(trees)
        score = (side ** 2) / n
        
        # Theoretical minimum assuming 90% packing density
        theoretical_area = n * tree_area / 0.9
        theoretical_side = math.sqrt(theoretical_area)
        theoretical_score = (theoretical_side ** 2) / n
        
        gap = score - theoretical_score
        print(f"N={n}: actual={score:.6f}, theoretical={theoretical_score:.6f}, gap={gap:.6f} ({gap/score*100:.1f}%)")
