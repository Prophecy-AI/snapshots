"""
Lattice-based placement with optimization.

Instead of random placement, use a structured lattice pattern
that achieves better packing density.
"""

import numpy as np
import pandas as pd
import math
import random
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

def generate_hexagonal_lattice(n, spacing=0.5):
    """
    Generate n positions on a hexagonal lattice.
    Hexagonal packing is optimal for circles and often good for other shapes.
    """
    positions = []
    
    # Calculate grid dimensions
    cols = int(math.ceil(math.sqrt(n * 1.2)))
    rows = int(math.ceil(n / cols)) + 1
    
    # Hexagonal spacing
    dx = spacing
    dy = spacing * math.sqrt(3) / 2
    
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= n:
                break
            
            x = col * dx
            if row % 2 == 1:
                x += dx / 2  # Offset odd rows
            y = row * dy
            
            positions.append((x, y))
            count += 1
        
        if count >= n:
            break
    
    return positions[:n]

def generate_square_lattice(n, spacing=0.5):
    """Generate n positions on a square lattice."""
    positions = []
    
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= n:
                break
            
            x = col * spacing
            y = row * spacing
            positions.append((x, y))
            count += 1
    
    return positions[:n]

def create_lattice_solution(n, lattice_type='hexagonal', spacing=0.5, base_angle=45):
    """
    Create a solution using lattice placement.
    
    All trees use the same angle (base_angle) for simplicity.
    """
    if lattice_type == 'hexagonal':
        positions = generate_hexagonal_lattice(n, spacing)
    else:
        positions = generate_square_lattice(n, spacing)
    
    trees = [[x, y, base_angle] for x, y in positions]
    return trees

def optimize_spacing(n, lattice_type='hexagonal', base_angle=45, verbose=False):
    """
    Find the optimal spacing for a lattice solution.
    
    Binary search for the smallest spacing that doesn't cause overlaps.
    """
    # Start with a large spacing and decrease
    spacing_high = 2.0
    spacing_low = 0.1
    
    best_trees = None
    best_side = float('inf')
    
    while spacing_high - spacing_low > 0.001:
        spacing = (spacing_high + spacing_low) / 2
        
        trees = create_lattice_solution(n, lattice_type, spacing, base_angle)
        
        if has_overlap(trees):
            # Too tight, increase spacing
            spacing_low = spacing
        else:
            # No overlap, try tighter
            spacing_high = spacing
            side = calculate_bounding_box(trees)
            if side < best_side:
                best_side = side
                best_trees = trees
    
    return best_trees, best_side

def try_multiple_configurations(n, verbose=False):
    """
    Try multiple lattice configurations and angles.
    """
    best_trees = None
    best_side = float('inf')
    
    # Try different lattice types and angles
    configs = [
        ('hexagonal', 0),
        ('hexagonal', 45),
        ('hexagonal', 90),
        ('hexagonal', 135),
        ('square', 0),
        ('square', 45),
        ('square', 90),
        ('square', 135),
    ]
    
    for lattice_type, base_angle in configs:
        trees, side = optimize_spacing(n, lattice_type, base_angle)
        
        if trees is not None and side < best_side:
            best_side = side
            best_trees = trees
            if verbose:
                print(f"  {lattice_type} @ {base_angle}°: side={side:.6f}")
    
    return best_trees, best_side

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

def test_lattice(baseline_solutions, test_ns=[5, 10, 15, 20, 30]):
    """Test lattice placement on small N values."""
    print("=" * 60)
    print("TESTING LATTICE PLACEMENT")
    print("=" * 60)
    
    results = {}
    
    for n in test_ns:
        print(f"\nN={n}:")
        baseline_trees = baseline_solutions[n]
        baseline_side = calculate_bounding_box(baseline_trees)
        baseline_score = (baseline_side ** 2) / n
        
        start_time = time.time()
        best_trees, best_side = try_multiple_configurations(n, verbose=True)
        elapsed = time.time() - start_time
        
        best_score = (best_side ** 2) / n
        improvement = baseline_score - best_score
        
        results[n] = {
            'baseline_side': baseline_side,
            'baseline_score': baseline_score,
            'best_side': best_side,
            'best_score': best_score,
            'improvement': improvement,
            'time': elapsed
        }
        
        print(f"\n  RESULTS for N={n}:")
        print(f"    Baseline: side={baseline_side:.6f}, score={baseline_score:.6f}")
        print(f"    Lattice: side={best_side:.6f}, score={best_score:.6f}")
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
    
    # Test lattice placement
    test_results = test_lattice(baseline_solutions, test_ns=[5, 10, 15, 20, 30])
    
    # Save results
    with open('/home/code/experiments/009_random_restart/lattice_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\nResults saved to lattice_results.json")
