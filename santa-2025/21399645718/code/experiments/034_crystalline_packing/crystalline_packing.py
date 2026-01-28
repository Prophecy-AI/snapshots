"""
Crystalline Packing for Large N values (N=101-200).
Based on the insight that top teams use regular geometric lattices for large N.
"""

import numpy as np
from shapely import Polygon
from shapely.affinity import rotate, translate
import pandas as pd
import time
import json

# Tree polygon vertices
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]

def get_tree_polygon(x, y, angle):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = rotate(poly, angle, origin=(0, 0))
    poly = translate(poly, x, y)
    return poly

def get_bbox_size(trees):
    all_coords = []
    for x, y, angle in trees:
        poly = get_tree_polygon(x, y, angle)
        all_coords.extend(poly.exterior.coords)
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    return max(max(xs) - min(xs), max(ys) - min(ys))

def check_overlap(tree1, tree2):
    poly1 = get_tree_polygon(*tree1)
    poly2 = get_tree_polygon(*tree2)
    if poly1.intersects(poly2):
        intersection = poly1.intersection(poly2)
        return intersection.area > 1e-10
    return False

def check_all_overlaps(trees):
    for i in range(len(trees)):
        for j in range(i+1, len(trees)):
            if check_overlap(trees[i], trees[j]):
                return True
    return False

def calculate_score(trees, n):
    bbox = get_bbox_size(trees)
    return bbox**2 / n

def center_trees(trees):
    if not trees:
        return trees
    xs = [t[0] for t in trees]
    ys = [t[1] for t in trees]
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    return [(x - cx, y - cy, a) for x, y, a in trees]

def hexagonal_lattice(n, spacing, angle=0):
    """Create hexagonal lattice positions for n trees"""
    positions = []
    side = int(np.ceil(np.sqrt(n) * 1.5))
    
    for row in range(side):
        for col in range(side):
            if len(positions) >= n:
                break
            x = col * spacing
            y = row * spacing * np.sqrt(3) / 2
            if row % 2 == 1:
                x += spacing / 2
            positions.append((x, y, angle))
        if len(positions) >= n:
            break
    
    return center_trees(positions[:n])

def square_lattice(n, spacing, angle=0):
    """Create square lattice positions for n trees"""
    positions = []
    side = int(np.ceil(np.sqrt(n)))
    
    for i in range(side):
        for j in range(side):
            if len(positions) < n:
                positions.append((i * spacing, j * spacing, angle))
    
    return center_trees(positions[:n])

def rectangular_lattice(n, spacing_x, spacing_y, angle=0):
    """Create rectangular lattice with different x and y spacing"""
    positions = []
    cols = int(np.ceil(np.sqrt(n * spacing_y / spacing_x)))
    rows = int(np.ceil(n / cols))
    
    for i in range(rows):
        for j in range(cols):
            if len(positions) < n:
                positions.append((j * spacing_x, i * spacing_y, angle))
    
    return center_trees(positions[:n])

def alternating_angle_lattice(n, spacing, angle1=0, angle2=180):
    """Create lattice with alternating angles (interlocking pattern)"""
    positions = []
    side = int(np.ceil(np.sqrt(n)))
    
    for i in range(side):
        for j in range(side):
            if len(positions) < n:
                angle = angle1 if (i + j) % 2 == 0 else angle2
                positions.append((i * spacing, j * spacing, angle))
    
    return center_trees(positions[:n])

def find_minimum_spacing(n, lattice_func, angle=0, start=0.3, end=1.5, step=0.01):
    """Find minimum spacing that produces no overlaps"""
    for spacing in np.arange(start, end, step):
        if lattice_func == rectangular_lattice:
            trees = lattice_func(n, spacing, spacing * 0.8, angle)
        elif lattice_func == alternating_angle_lattice:
            trees = lattice_func(n, spacing, angle, angle + 180)
        else:
            trees = lattice_func(n, spacing, angle)
        
        if not check_all_overlaps(trees):
            return spacing, trees
    
    return None, None

def parse_value(s):
    if isinstance(s, str) and s.startswith('s'):
        return float(s[1:])
    return float(s)

def load_baseline():
    df = pd.read_csv('/home/submission/submission.csv')
    df['n'] = df['id'].apply(lambda x: int(x.split('_')[0]))
    for col in ['x', 'y', 'deg']:
        df[col+'_val'] = df[col].apply(parse_value)
    
    baseline = {}
    for n in range(1, 201):
        group = df[df['n'] == n]
        trees = [(row['x_val'], row['y_val'], row['deg_val']) for _, row in group.iterrows()]
        baseline[n] = {
            'trees': trees,
            'score': calculate_score(trees, n)
        }
    return baseline

def main():
    print("Loading baseline...")
    baseline = load_baseline()
    
    # Calculate score contribution from N=101-200
    large_n_score = sum(baseline[n]['score'] for n in range(101, 201))
    total_score = sum(b['score'] for b in baseline.values())
    print(f"Total baseline score: {total_score:.6f}")
    print(f"N=101-200 contribution: {large_n_score:.6f} ({large_n_score/total_score*100:.1f}%)")
    
    # Test crystalline packing for large N
    lattice_funcs = [
        ('hexagonal', hexagonal_lattice),
        ('square', square_lattice),
        ('rectangular', rectangular_lattice),
        ('alternating', alternating_angle_lattice),
    ]
    
    angles = [0, 15, 30, 45, 60, 75, 90]
    
    improvements = {}
    best_per_n = {n: baseline[n].copy() for n in range(1, 201)}
    
    print("\nTesting crystalline packing for N=100-200...")
    print("="*60)
    
    for n in [100, 120, 140, 160, 180, 200]:
        print(f"\nN={n}: baseline score = {baseline[n]['score']:.6f}")
        best_score = baseline[n]['score']
        best_trees = baseline[n]['trees']
        best_method = "baseline"
        
        for lattice_name, lattice_func in lattice_funcs:
            for angle in angles:
                spacing, trees = find_minimum_spacing(n, lattice_func, angle)
                
                if trees is None:
                    continue
                
                score = calculate_score(trees, n)
                
                if score < best_score - 1e-8:
                    best_score = score
                    best_trees = trees
                    best_method = f"{lattice_name}_angle{angle}"
                    print(f"  IMPROVEMENT: {lattice_name} angle={angle} spacing={spacing:.3f} -> {score:.6f} (improvement: {baseline[n]['score'] - score:.6f})")
        
        if best_score < baseline[n]['score'] - 1e-8:
            improvements[n] = {
                'improvement': baseline[n]['score'] - best_score,
                'method': best_method,
                'score': best_score
            }
            best_per_n[n] = {'trees': best_trees, 'score': best_score}
        else:
            print(f"  No improvement found (best crystalline >= baseline)")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if improvements:
        total_improvement = sum(imp['improvement'] for imp in improvements.values())
        print(f"Found {len(improvements)} N values with improvements:")
        for n, data in sorted(improvements.items()):
            print(f"  N={n}: improvement={data['improvement']:.6f} via {data['method']}")
        print(f"\nTotal improvement: {total_improvement:.6f}")
        
        new_total = sum(b['score'] for b in best_per_n.values())
        print(f"New total score: {new_total:.6f}")
        print(f"Baseline total: {total_score:.6f}")
    else:
        print("No improvements found with crystalline packing")
        print("The baseline already uses optimized irregular patterns that beat regular lattices")
    
    # Save results
    results = {
        'baseline_total': total_score,
        'large_n_contribution': large_n_score,
        'improvements': {str(n): data for n, data in improvements.items()} if improvements else {},
        'total_improvement': sum(imp['improvement'] for imp in improvements.values()) if improvements else 0
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return best_per_n, results

if __name__ == '__main__':
    best_per_n, results = main()
