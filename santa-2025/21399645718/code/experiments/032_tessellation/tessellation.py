"""
Tessellation-based construction for Christmas tree packing.
Generate solutions using regular patterns instead of optimization.
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
    """Center trees around origin"""
    if not trees:
        return trees
    xs = [t[0] for t in trees]
    ys = [t[1] for t in trees]
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    return [(x - cx, y - cy, a) for x, y, a in trees]

def hexagonal_tessellation(n, spacing=1.0, angle=0):
    """Generate n trees in hexagonal pattern"""
    trees = []
    # Estimate grid size needed
    side = int(np.ceil(np.sqrt(n) * 1.5))
    
    for row in range(side):
        for col in range(side):
            if len(trees) >= n:
                break
            x = col * spacing * 0.866  # cos(30)
            y = row * spacing + (col % 2) * spacing * 0.5
            trees.append((x, y, angle))
        if len(trees) >= n:
            break
    
    return center_trees(trees[:n])

def square_tessellation(n, spacing=1.0, angle=0):
    """Generate n trees in square pattern"""
    trees = []
    side = int(np.ceil(np.sqrt(n)))
    
    for i in range(side):
        for j in range(side):
            if len(trees) < n:
                trees.append((i * spacing, j * spacing, angle))
    
    return center_trees(trees[:n])

def diagonal_tessellation(n, spacing=1.0, angle=45):
    """Generate n trees in diagonal pattern"""
    trees = []
    side = int(np.ceil(np.sqrt(n) * 1.5))
    
    for i in range(side):
        for j in range(side):
            if len(trees) < n:
                x = (i + j * 0.5) * spacing
                y = j * spacing * 0.866
                trees.append((x, y, angle))
    
    return center_trees(trees[:n])

def triangular_tessellation(n, spacing=1.0, angle=0):
    """Generate n trees in triangular pattern"""
    trees = []
    side = int(np.ceil(np.sqrt(n) * 1.5))
    
    for row in range(side):
        cols_in_row = side - row
        for col in range(cols_in_row):
            if len(trees) < n:
                x = col * spacing + row * spacing * 0.5
                y = row * spacing * 0.866
                trees.append((x, y, angle))
    
    return center_trees(trees[:n])

def compact_spacing(n, base_spacing=1.0):
    """Find minimum spacing that avoids overlaps"""
    for spacing_mult in np.arange(0.5, 2.0, 0.05):
        spacing = base_spacing * spacing_mult
        trees = square_tessellation(n, spacing, 0)
        if not check_all_overlaps(trees):
            return spacing
    return base_spacing * 2.0

def parse_value(s):
    if isinstance(s, str) and s.startswith('s'):
        return float(s[1:])
    return float(s)

def load_baseline():
    """Load baseline scores per N"""
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
    
    print(f"Baseline total score: {sum(b['score'] for b in baseline.values()):.6f}")
    
    # Test tessellation patterns
    tessellation_funcs = [
        ('hexagonal', hexagonal_tessellation),
        ('square', square_tessellation),
        ('diagonal', diagonal_tessellation),
        ('triangular', triangular_tessellation),
    ]
    
    angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
    spacings = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    improvements = {}
    best_per_n = {n: baseline[n].copy() for n in range(1, 201)}
    
    print("\nTesting tessellation patterns...")
    
    for tess_name, tess_func in tessellation_funcs:
        print(f"\n{tess_name.upper()} TESSELLATION:")
        tess_improvements = 0
        
        for n in range(2, 51):  # Focus on small N first
            best_score = baseline[n]['score']
            best_trees = baseline[n]['trees']
            
            for angle in angles:
                for spacing in spacings:
                    try:
                        trees = tess_func(n, spacing, angle)
                        
                        if check_all_overlaps(trees):
                            continue
                        
                        score = calculate_score(trees, n)
                        
                        if score < best_score - 1e-8:
                            best_score = score
                            best_trees = trees
                            improvement = baseline[n]['score'] - score
                            print(f"  N={n}: {tess_name} angle={angle} spacing={spacing:.1f} -> {score:.6f} (improvement: {improvement:.6f})")
                            tess_improvements += 1
                    except Exception as e:
                        continue
            
            if best_score < best_per_n[n]['score']:
                best_per_n[n] = {'trees': best_trees, 'score': best_score}
        
        print(f"  Total improvements from {tess_name}: {tess_improvements}")
    
    # Calculate total improvement
    new_total = sum(b['score'] for b in best_per_n.values())
    baseline_total = sum(b['score'] for b in baseline.values())
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline total: {baseline_total:.6f}")
    print(f"New total: {new_total:.6f}")
    print(f"Total improvement: {baseline_total - new_total:.6f}")
    
    # Save results
    results = {
        'baseline_total': baseline_total,
        'new_total': new_total,
        'improvement': baseline_total - new_total,
        'improvements_per_n': {n: baseline[n]['score'] - best_per_n[n]['score'] 
                               for n in range(1, 201) 
                               if baseline[n]['score'] - best_per_n[n]['score'] > 1e-8}
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return best_per_n, results

if __name__ == '__main__':
    best_per_n, results = main()
