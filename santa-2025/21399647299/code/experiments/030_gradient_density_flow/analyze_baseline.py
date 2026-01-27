"""
Analyze the baseline solution to understand why it's so good.

Key questions:
1. What is the packing efficiency?
2. How close are trees to each other?
3. What angles are used?
4. What is the theoretical minimum?
"""

import numpy as np
import pandas as pd
from shapely import Polygon
from shapely.ops import unary_union
import math
import json

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def get_tree_polygon(x, y, angle_deg):
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    vertices = []
    for tx, ty in zip(TX, TY):
        rx = tx * cos_a - ty * sin_a
        ry = tx * sin_a + ty * cos_a
        vertices.append((rx + x, ry + y))
    
    return Polygon(vertices)

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

def compute_bbox(xs, ys, angles):
    all_x = []
    all_y = []
    
    for x, y, angle in zip(xs, ys, angles):
        vertices = get_tree_vertices(x, y, angle)
        for vx, vy in vertices:
            all_x.append(vx)
            all_y.append(vy)
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    return max(max_x - min_x, max_y - min_y), min_x, max_x, min_y, max_y

def compute_score(xs, ys, angles, n):
    bbox, _, _, _, _ = compute_bbox(xs, ys, angles)
    return bbox ** 2 / n

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

def analyze_solution(xs, ys, angles, n):
    """Analyze a solution in detail."""
    # Compute tree area
    tree_poly = get_tree_polygon(0, 0, 0)
    tree_area = tree_poly.area
    total_tree_area = n * tree_area
    
    # Compute bounding box
    bbox, min_x, max_x, min_y, max_y = compute_bbox(xs, ys, angles)
    bbox_area = bbox ** 2
    
    # Packing efficiency
    efficiency = total_tree_area / bbox_area
    
    # Compute actual union area (accounting for any overlaps)
    polygons = [get_tree_polygon(x, y, a) for x, y, a in zip(xs, ys, angles)]
    union = unary_union(polygons)
    actual_area = union.area
    
    # Angle distribution
    angle_counts = {}
    for a in angles:
        a_norm = a % 360
        if a_norm not in angle_counts:
            angle_counts[a_norm] = 0
        angle_counts[a_norm] += 1
    
    # Minimum distance between trees
    min_dist = float('inf')
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            dist = math.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            min_dist = min(min_dist, dist)
    
    return {
        'n': n,
        'tree_area': tree_area,
        'total_tree_area': total_tree_area,
        'bbox': bbox,
        'bbox_area': bbox_area,
        'actual_union_area': actual_area,
        'efficiency': efficiency,
        'score': bbox ** 2 / n,
        'min_distance': min_dist,
        'angle_distribution': angle_counts,
        'width': max_x - min_x,
        'height': max_y - min_y
    }

def main():
    print("=" * 60)
    print("BASELINE SOLUTION ANALYSIS")
    print("=" * 60)
    
    baseline_path = "/home/submission/submission.csv"
    solutions = load_baseline_solution(baseline_path)
    
    # Compute tree area
    tree_poly = get_tree_polygon(0, 0, 0)
    tree_area = tree_poly.area
    print(f"\nSingle tree area: {tree_area:.6f}")
    
    # Theoretical minimum: if we could pack with 100% efficiency
    print("\nTheoretical minimum scores (100% efficiency):")
    for n in [1, 5, 10, 20, 50, 100, 200]:
        # bbox^2 / n = total_tree_area / efficiency
        # At 100% efficiency: bbox^2 = n * tree_area
        # So bbox = sqrt(n * tree_area)
        # Score = bbox^2 / n = tree_area
        theoretical_min = tree_area
        print(f"  N={n}: {theoretical_min:.6f} (same for all N at 100% efficiency)")
    
    print("\n" + "=" * 60)
    print("ACTUAL BASELINE ANALYSIS")
    print("=" * 60)
    
    test_ns = [1, 5, 10, 20, 30, 50, 100, 150, 200]
    
    for n in test_ns:
        if n not in solutions:
            continue
            
        trees = solutions[n]
        xs = [t[0] for t in trees]
        ys = [t[1] for t in trees]
        angles = [t[2] for t in trees]
        
        analysis = analyze_solution(xs, ys, angles, n)
        
        print(f"\nN={n}:")
        print(f"  Score: {analysis['score']:.6f}")
        print(f"  Bbox: {analysis['bbox']:.4f} (width={analysis['width']:.4f}, height={analysis['height']:.4f})")
        print(f"  Efficiency: {analysis['efficiency']*100:.2f}%")
        print(f"  Min tree distance: {analysis['min_distance']:.4f}")
        print(f"  Theoretical min (100% eff): {tree_area:.6f}")
        print(f"  Gap to theoretical: {analysis['score'] - tree_area:.6f} ({(analysis['score']/tree_area - 1)*100:.1f}%)")
        
        # Top 3 angles used
        sorted_angles = sorted(analysis['angle_distribution'].items(), key=lambda x: -x[1])[:3]
        print(f"  Top angles: {sorted_angles}")
    
    # Compute total score
    total_score = 0
    for n in range(1, 201):
        if n in solutions:
            trees = solutions[n]
            xs = [t[0] for t in trees]
            ys = [t[1] for t in trees]
            angles = [t[2] for t in trees]
            score = compute_score(xs, ys, angles, n)
            total_score += score
    
    print("\n" + "=" * 60)
    print(f"TOTAL SCORE: {total_score:.6f}")
    print("=" * 60)
    
    # Theoretical minimum total score
    theoretical_total = 200 * tree_area
    print(f"Theoretical minimum (100% efficiency for all N): {theoretical_total:.6f}")
    print(f"Gap to theoretical: {total_score - theoretical_total:.6f}")
    print(f"Current efficiency: {theoretical_total / total_score * 100:.2f}%")

if __name__ == "__main__":
    main()
