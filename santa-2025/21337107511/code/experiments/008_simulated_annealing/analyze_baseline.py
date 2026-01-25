"""
Analyze the baseline solution to understand its structure.
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

def analyze_solution(trees, n):
    """Analyze a solution for N trees."""
    angles = [t[2] for t in trees]
    xs = [t[0] for t in trees]
    ys = [t[1] for t in trees]
    
    # Angle distribution
    angle_counts = {}
    for a in angles:
        a_rounded = round(a / 45) * 45  # Round to nearest 45 degrees
        angle_counts[a_rounded] = angle_counts.get(a_rounded, 0) + 1
    
    # Position spread
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    
    # Bounding box
    side = calculate_bounding_box(trees)
    score = (side ** 2) / n
    
    return {
        'n': n,
        'side': side,
        'score': score,
        'x_range': x_range,
        'y_range': y_range,
        'angle_distribution': angle_counts,
        'unique_angles': len(set(angles)),
        'mean_angle': np.mean(angles),
        'std_angle': np.std(angles)
    }

if __name__ == "__main__":
    baseline_path = "/home/nonroot/snapshots/santa-2025/21337353543/submission/submission.csv"
    print(f"Loading baseline from {baseline_path}")
    baseline_solutions = load_baseline(baseline_path)
    
    print("\n" + "=" * 60)
    print("BASELINE ANALYSIS")
    print("=" * 60)
    
    # Analyze small N values
    for n in [1, 2, 3, 4, 5, 10, 20, 50, 100]:
        trees = baseline_solutions[n]
        analysis = analyze_solution(trees, n)
        
        print(f"\nN={n}:")
        print(f"  Side: {analysis['side']:.6f}")
        print(f"  Score: {analysis['score']:.6f}")
        print(f"  X range: {analysis['x_range']:.6f}")
        print(f"  Y range: {analysis['y_range']:.6f}")
        print(f"  Unique angles: {analysis['unique_angles']}")
        print(f"  Angle distribution: {analysis['angle_distribution']}")
        print(f"  Mean angle: {analysis['mean_angle']:.2f}")
        print(f"  Std angle: {analysis['std_angle']:.2f}")
        
        # Print actual tree positions for small N
        if n <= 5:
            print(f"  Trees:")
            for i, (x, y, angle) in enumerate(trees):
                print(f"    {i}: x={x:.6f}, y={y:.6f}, angle={angle:.2f}")
    
    # Calculate theoretical minimum for N=1
    print("\n" + "=" * 60)
    print("THEORETICAL ANALYSIS FOR N=1")
    print("=" * 60)
    
    # For N=1, find optimal rotation
    best_angle = None
    best_side = float('inf')
    
    for angle_int in range(0, 3600):  # 0.1 degree increments
        angle = angle_int / 10.0
        poly = get_tree_polygon(0, 0, angle)
        coords = list(poly.exterior.coords)
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        side = max(max(xs) - min(xs), max(ys) - min(ys))
        
        if side < best_side:
            best_side = side
            best_angle = angle
    
    print(f"Optimal N=1: angle={best_angle:.1f}, side={best_side:.6f}, score={best_side**2:.6f}")
    print(f"Baseline N=1: angle={baseline_solutions[1][0][2]:.2f}, side={calculate_bounding_box(baseline_solutions[1]):.6f}")
