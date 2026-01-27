"""
Analyze the bimodal tessellation pattern in the baseline solution.

Key questions:
1. Why do trees cluster at ~67° and ~247°?
2. What is the optimal angle pair for interlocking trees?
3. How does the spacing vary with N?
"""

import numpy as np
import pandas as pd
from shapely import Polygon
import math
import json
from collections import Counter

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def get_tree_vertices(x, y, angle_deg):
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    vertices = []
    for tx, ty in zip(TX, TY):
        rx = tx * cos_a - ty * sin_a
        ry = tx * sin_a + ty * cos_a
        vertices.append((rx + x, ry + y))
    
    return vertices

def load_baseline_solution(csv_path):
    df = pd.read_csv(csv_path)
    
    solutions = {}
    for _, row in df.iterrows():
        id_parts = row['id'].split('_')
        n = int(id_parts[0])
        i = int(id_parts[1])
        
        x = float(str(row['x']).replace('s', ''))
        y = float(str(row['y']).replace('s', ''))
        deg = float(str(row['deg']).replace('s', ''))
        
        if n not in solutions:
            solutions[n] = []
        solutions[n].append((x, y, deg))
    
    return solutions

def analyze_angle_distribution(angles):
    """Analyze the distribution of angles."""
    # Normalize to 0-360
    angles_norm = [a % 360 for a in angles]
    
    # Bin into 10-degree buckets
    bins = [0] * 36
    for a in angles_norm:
        bin_idx = int(a / 10)
        bins[bin_idx] += 1
    
    # Find peaks
    peaks = []
    for i in range(36):
        if bins[i] > len(angles) * 0.1:  # More than 10% of trees
            peaks.append((i * 10, bins[i]))
    
    return angles_norm, bins, peaks

def compute_tree_dimensions(angle_deg):
    """Compute the width and height of a tree at a given angle."""
    vertices = get_tree_vertices(0, 0, angle_deg)
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    
    return width, height

def main():
    print("=" * 60)
    print("BIMODAL TESSELLATION PATTERN ANALYSIS")
    print("=" * 60)
    
    baseline_path = "/home/submission/submission.csv"
    solutions = load_baseline_solution(baseline_path)
    
    # Analyze large N values
    test_ns = [50, 100, 150, 200]
    
    print("\n" + "=" * 60)
    print("ANGLE DISTRIBUTION FOR LARGE N")
    print("=" * 60)
    
    for n in test_ns:
        if n not in solutions:
            continue
            
        trees = solutions[n]
        angles = [t[2] for t in trees]
        
        angles_norm, bins, peaks = analyze_angle_distribution(angles)
        
        print(f"\nN={n}:")
        print(f"  Number of trees: {len(trees)}")
        print(f"  Angle range: {min(angles_norm):.1f}° - {max(angles_norm):.1f}°")
        print(f"  Peaks (>10% of trees): {peaks}")
        
        # Count trees in two main clusters (around 67° and 247°)
        cluster1 = sum(1 for a in angles_norm if 50 <= a <= 90)
        cluster2 = sum(1 for a in angles_norm if 230 <= a <= 270)
        other = len(angles) - cluster1 - cluster2
        
        print(f"  Cluster 50-90°: {cluster1} trees ({cluster1/len(angles)*100:.1f}%)")
        print(f"  Cluster 230-270°: {cluster2} trees ({cluster2/len(angles)*100:.1f}%)")
        print(f"  Other: {other} trees ({other/len(angles)*100:.1f}%)")
    
    # Analyze tree dimensions at different angles
    print("\n" + "=" * 60)
    print("TREE DIMENSIONS AT DIFFERENT ANGLES")
    print("=" * 60)
    
    for angle in [0, 30, 45, 60, 67, 75, 90]:
        width, height = compute_tree_dimensions(angle)
        bbox = max(width, height)
        print(f"  {angle:3d}°: width={width:.4f}, height={height:.4f}, bbox={bbox:.4f}")
    
    # Why 67° and 247°?
    print("\n" + "=" * 60)
    print("WHY 67° AND 247°?")
    print("=" * 60)
    
    # Find the angle that minimizes width (for horizontal packing)
    best_angle_width = 0
    best_width = float('inf')
    
    for angle in range(0, 180):
        width, height = compute_tree_dimensions(angle)
        if width < best_width:
            best_width = width
            best_angle_width = angle
    
    print(f"  Angle that minimizes width: {best_angle_width}° (width={best_width:.4f})")
    
    # Find the angle that minimizes height (for vertical packing)
    best_angle_height = 0
    best_height = float('inf')
    
    for angle in range(0, 180):
        width, height = compute_tree_dimensions(angle)
        if height < best_height:
            best_height = height
            best_angle_height = angle
    
    print(f"  Angle that minimizes height: {best_angle_height}° (height={best_height:.4f})")
    
    # Find the angle that minimizes bbox (for square packing)
    best_angle_bbox = 0
    best_bbox = float('inf')
    
    for angle in range(0, 180):
        width, height = compute_tree_dimensions(angle)
        bbox = max(width, height)
        if bbox < best_bbox:
            best_bbox = bbox
            best_angle_bbox = angle
    
    print(f"  Angle that minimizes bbox: {best_angle_bbox}° (bbox={best_bbox:.4f})")
    
    # Analyze spacing in baseline
    print("\n" + "=" * 60)
    print("SPACING ANALYSIS FOR LARGE N")
    print("=" * 60)
    
    for n in [100, 150, 200]:
        if n not in solutions:
            continue
            
        trees = solutions[n]
        xs = [t[0] for t in trees]
        ys = [t[1] for t in trees]
        
        # Compute pairwise distances
        distances = []
        for i in range(len(trees)):
            for j in range(i + 1, len(trees)):
                dist = math.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                distances.append(dist)
        
        min_dist = min(distances)
        avg_dist = sum(distances) / len(distances)
        
        # Estimate grid spacing
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        side = int(math.ceil(math.sqrt(n)))
        
        print(f"\nN={n}:")
        print(f"  Min distance: {min_dist:.4f}")
        print(f"  Avg distance: {avg_dist:.4f}")
        print(f"  X range: {x_range:.4f}")
        print(f"  Y range: {y_range:.4f}")
        print(f"  Estimated grid spacing: x={x_range/(side-1):.4f}, y={y_range/(side-1):.4f}")

if __name__ == "__main__":
    main()
