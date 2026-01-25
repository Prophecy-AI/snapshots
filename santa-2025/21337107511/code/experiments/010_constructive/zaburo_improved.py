"""
Improved zaburo-style constructive algorithm.

The original zaburo achieves 88.33 with fixed spacing.
Let's optimize the spacing for each N to get better results.
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

def has_overlap(trees):
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

def zaburo_pattern(n, n_even, n_odd, h_spacing=0.7, v_spacing=1.0):
    """
    Exact zaburo pattern with configurable spacing.
    
    n_even: trees per even row
    n_odd: trees per odd row
    """
    trees = []
    remaining = n
    row = 0
    
    while remaining > 0:
        is_even = (row % 2 == 0)
        m = min(remaining, n_even if is_even else n_odd)
        
        angle = 0 if is_even else 180
        x_offset = 0 if is_even else h_spacing / 2
        
        if is_even:
            y = row // 2 * v_spacing
        else:
            y = 0.8 + (row - 1) // 2 * v_spacing
        
        for i in range(m):
            x = i * h_spacing + x_offset
            trees.append([x, y, angle])
        
        remaining -= m
        row += 1
    
    return trees

def find_best_zaburo(n, verbose=False):
    """
    Find the best zaburo configuration for n trees.
    Try different row configurations and spacings.
    """
    best_trees = None
    best_side = float('inf')
    best_config = None
    
    # Try different row configurations
    for n_even in range(1, n + 1):
        for n_odd in [n_even, n_even - 1, n_even + 1]:
            if n_odd < 1:
                continue
            
            # Try different spacings
            for h_spacing in [0.5, 0.6, 0.7, 0.8]:
                for v_spacing in [0.8, 0.9, 1.0, 1.1]:
                    trees = zaburo_pattern(n, n_even, n_odd, h_spacing, v_spacing)
                    
                    if len(trees) != n:
                        continue
                    
                    if not has_overlap(trees):
                        side = calculate_bounding_box(trees)
                        if side < best_side:
                            best_side = side
                            best_trees = trees
                            best_config = f"n_even={n_even}, n_odd={n_odd}, h={h_spacing}, v={v_spacing}"
                            if verbose:
                                print(f"    New best: {best_config}, side={side:.4f}")
    
    return best_trees, best_side, best_config

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

def test_zaburo_improved(baseline_solutions, test_ns=[5, 10, 20, 50]):
    """Test improved zaburo on small N values."""
    print("=" * 60)
    print("TESTING IMPROVED ZABURO PATTERN")
    print("=" * 60)
    
    results = {}
    
    for n in test_ns:
        print(f"\nN={n}:")
        baseline_trees = baseline_solutions[n]
        baseline_side = calculate_bounding_box(baseline_trees)
        baseline_score = (baseline_side ** 2) / n
        
        start_time = time.time()
        best_trees, best_side, best_config = find_best_zaburo(n, verbose=True)
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
        print(f"    Zaburo: side={best_side:.6f}, score={best_score:.6f}")
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
    baseline_path = "/home/nonroot/snapshots/santa-2025/21337353543/submission/submission.csv"
    print(f"Loading baseline from {baseline_path}")
    baseline_solutions = load_baseline(baseline_path)
    
    # Test improved zaburo
    test_results = test_zaburo_improved(baseline_solutions, test_ns=[5, 10, 20])
    
    # Save results
    with open('/home/code/experiments/010_constructive/zaburo_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\nResults saved to zaburo_results.json")
