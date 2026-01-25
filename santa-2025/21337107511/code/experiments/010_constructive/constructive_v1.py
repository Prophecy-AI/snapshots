"""
Constructive Algorithm for Tree Packing

Based on zaburo kernel's alternating row approach, but with improvements:
1. Try different angle combinations (not just 0°/180°)
2. Optimize spacing for each N
3. Try different row configurations

The zaburo kernel achieves 88.33 with 0°/180° and 0.7/1.0 spacing.
The baseline achieves 70.6 with sophisticated interlocking.
Our goal: Find a constructive approach that can beat the baseline.
"""

import numpy as np
import pandas as pd
import math
from shapely import Polygon
from shapely.affinity import rotate, translate
from decimal import Decimal, getcontext
import time
import json

getcontext().prec = 30

# Tree polygon vertices
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]
TREE_COORDS = list(zip(TX, TY))

def get_tree_polygon(x, y, angle):
    """Create tree polygon at position (x, y) with rotation angle (degrees)."""
    poly = Polygon(TREE_COORDS)
    poly = rotate(poly, angle, origin=(0, 0))
    poly = translate(poly, x, y)
    return poly

def calculate_bounding_box(trees):
    """Calculate the bounding box side length for a set of trees."""
    all_coords = []
    for x, y, angle in trees:
        poly = get_tree_polygon(x, y, angle)
        all_coords.extend(poly.exterior.coords)
    
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    
    return max(max(xs) - min(xs), max(ys) - min(ys))

def has_overlap(trees):
    """Check if any trees overlap."""
    n = len(trees)
    if n <= 1:
        return False
    
    polygons = [get_tree_polygon(x, y, angle) for x, y, angle in trees]
    
    for i in range(n):
        for j in range(i + 1, n):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > 1e-15:
                    return True
    return False

def construct_alternating_rows(n, angle_even=0, angle_odd=180, h_spacing=0.7, v_spacing=1.0, offset=0.35):
    """
    Construct solution using alternating rows pattern (zaburo style).
    
    Parameters:
    - angle_even: angle for even rows (0, 2, 4, ...)
    - angle_odd: angle for odd rows (1, 3, 5, ...)
    - h_spacing: horizontal spacing between trees
    - v_spacing: vertical spacing between rows
    - offset: horizontal offset for odd rows
    """
    trees = []
    
    # Calculate optimal row configuration
    # Try different numbers of trees per row
    best_trees = None
    best_side = float('inf')
    
    for n_per_row in range(1, n + 1):
        test_trees = []
        remaining = n
        row = 0
        
        while remaining > 0:
            is_even = (row % 2 == 0)
            angle = angle_even if is_even else angle_odd
            x_offset = 0 if is_even else offset
            y = row * v_spacing / 2  # Interleave rows
            
            # Alternate y position for interlocking
            if not is_even:
                y = 0.8 + (row - 1) // 2 * v_spacing
            else:
                y = row // 2 * v_spacing
            
            count = min(remaining, n_per_row)
            for i in range(count):
                x = i * h_spacing + x_offset
                test_trees.append([x, y, angle])
            
            remaining -= count
            row += 1
        
        # Check for overlaps
        if not has_overlap(test_trees):
            side = calculate_bounding_box(test_trees)
            if side < best_side:
                best_side = side
                best_trees = test_trees
    
    return best_trees, best_side

def construct_interlocking(n, angle1=45, angle2=225, spacing=0.5):
    """
    Construct solution using interlocking pattern.
    
    Trees at angle1 and angle2 (180° apart) can interlock.
    """
    trees = []
    
    # Calculate grid size
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= n:
                break
            
            # Alternate angles in checkerboard pattern
            angle = angle1 if (row + col) % 2 == 0 else angle2
            
            x = col * spacing
            y = row * spacing
            
            trees.append([x, y, angle])
            count += 1
    
    return trees

def optimize_spacing_for_pattern(n, pattern_func, angle1, angle2, verbose=False):
    """
    Binary search for optimal spacing that doesn't cause overlaps.
    """
    spacing_high = 2.0
    spacing_low = 0.3
    
    best_trees = None
    best_side = float('inf')
    
    while spacing_high - spacing_low > 0.01:
        spacing = (spacing_high + spacing_low) / 2
        
        if pattern_func == 'alternating':
            trees, _ = construct_alternating_rows(n, angle1, angle2, h_spacing=spacing, v_spacing=spacing*1.4)
        else:
            trees = construct_interlocking(n, angle1, angle2, spacing)
        
        if trees is None:
            spacing_low = spacing
            continue
        
        if has_overlap(trees):
            spacing_low = spacing
        else:
            spacing_high = spacing
            side = calculate_bounding_box(trees)
            if side < best_side:
                best_side = side
                best_trees = trees
    
    return best_trees, best_side

def find_best_constructive(n, verbose=False):
    """
    Try multiple constructive approaches and return the best.
    """
    best_trees = None
    best_side = float('inf')
    best_config = None
    
    # Angle combinations to try
    angle_pairs = [
        (0, 180),
        (45, 225),
        (90, 270),
        (135, 315),
        (0, 90),
        (45, 135),
        (90, 180),
        (135, 225),
    ]
    
    for angle1, angle2 in angle_pairs:
        # Try alternating rows
        trees, side = construct_alternating_rows(n, angle1, angle2)
        if trees is not None and side < best_side:
            best_side = side
            best_trees = trees
            best_config = f"alternating_{angle1}_{angle2}"
            if verbose:
                print(f"  alternating {angle1}/{angle2}: side={side:.4f}")
        
        # Try interlocking with optimized spacing
        trees, side = optimize_spacing_for_pattern(n, 'interlocking', angle1, angle2)
        if trees is not None and side < best_side:
            best_side = side
            best_trees = trees
            best_config = f"interlocking_{angle1}_{angle2}"
            if verbose:
                print(f"  interlocking {angle1}/{angle2}: side={side:.4f}")
    
    return best_trees, best_side, best_config

def load_baseline(csv_path):
    """Load baseline solution from CSV."""
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

def test_constructive(baseline_solutions, test_ns=[5, 10, 20, 50]):
    """Test constructive approaches on small N values."""
    print("=" * 60)
    print("TESTING CONSTRUCTIVE APPROACHES")
    print("=" * 60)
    
    results = {}
    
    for n in test_ns:
        print(f"\nN={n}:")
        baseline_trees = baseline_solutions[n]
        baseline_side = calculate_bounding_box(baseline_trees)
        baseline_score = (baseline_side ** 2) / n
        
        start_time = time.time()
        best_trees, best_side, best_config = find_best_constructive(n, verbose=True)
        elapsed = time.time() - start_time
        
        if best_trees is None:
            print(f"  ERROR: Could not construct valid solution")
            continue
        
        best_score = (best_side ** 2) / n
        improvement = baseline_score - best_score
        
        results[n] = {
            'baseline_side': baseline_side,
            'baseline_score': baseline_score,
            'best_side': best_side,
            'best_score': best_score,
            'improvement': improvement,
            'config': best_config,
            'time': elapsed
        }
        
        print(f"\n  RESULTS for N={n}:")
        print(f"    Baseline: side={baseline_side:.6f}, score={baseline_score:.6f}")
        print(f"    Constructive: side={best_side:.6f}, score={best_score:.6f}")
        print(f"    Best config: {best_config}")
        print(f"    Improvement: {improvement:.8f} ({improvement/baseline_score*100:.4f}%)")
        
        if improvement > 0:
            print(f"    ✅ BEAT BASELINE!")
        else:
            print(f"    ❌ Did not beat baseline (gap: {-improvement:.6f})")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_improvement = sum(r['improvement'] for r in results.values())
    beat_baseline = sum(1 for r in results.values() if r['improvement'] > 0)
    
    print(f"Total improvement: {total_improvement:.8f}")
    print(f"N values that beat baseline: {beat_baseline}/{len(test_ns)}")
    
    return results

if __name__ == "__main__":
    # Load baseline
    baseline_path = "/home/nonroot/snapshots/santa-2025/21337353543/submission/submission.csv"
    print(f"Loading baseline from {baseline_path}")
    baseline_solutions = load_baseline(baseline_path)
    
    # Test constructive approaches
    test_results = test_constructive(baseline_solutions, test_ns=[5, 10, 20, 50])
    
    # Save results
    with open('/home/code/experiments/010_constructive/test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\nResults saved to test_results.json")
