"""
Tessellation-based construction v2 - with tighter packing.
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

def interlocking_pattern(n, angle=0):
    """
    Create interlocking pattern where trees are rotated 180 degrees alternately.
    This allows tighter packing.
    """
    trees = []
    side = int(np.ceil(np.sqrt(n)))
    
    # Find minimum spacing that works
    for spacing in np.arange(0.4, 1.5, 0.01):
        trees = []
        for i in range(side):
            for j in range(side):
                if len(trees) < n:
                    # Alternate rotation by 180 degrees
                    tree_angle = angle if (i + j) % 2 == 0 else angle + 180
                    trees.append((i * spacing, j * spacing, tree_angle))
        
        trees = center_trees(trees[:n])
        if not check_all_overlaps(trees):
            return trees, spacing
    
    return None, None

def brick_pattern(n, angle=0):
    """
    Brick-like pattern with offset rows.
    """
    trees = []
    side = int(np.ceil(np.sqrt(n)))
    
    for spacing in np.arange(0.4, 1.5, 0.01):
        trees = []
        for row in range(side):
            for col in range(side):
                if len(trees) < n:
                    x = col * spacing + (row % 2) * spacing * 0.5
                    y = row * spacing * 0.866
                    trees.append((x, y, angle))
        
        trees = center_trees(trees[:n])
        if not check_all_overlaps(trees):
            return trees, spacing
    
    return None, None

def spiral_pattern(n, angle=0):
    """
    Spiral pattern from center outward.
    """
    trees = [(0, 0, angle)]
    
    if n == 1:
        return trees, 0
    
    # Find minimum spacing
    for spacing in np.arange(0.4, 1.5, 0.01):
        trees = [(0, 0, angle)]
        radius = spacing
        angle_step = 0.5  # radians
        current_angle = 0
        
        while len(trees) < n:
            x = radius * np.cos(current_angle)
            y = radius * np.sin(current_angle)
            new_tree = (x, y, angle)
            
            # Check if this tree overlaps with any existing
            overlaps = False
            for t in trees:
                if check_overlap(new_tree, t):
                    overlaps = True
                    break
            
            if not overlaps:
                trees.append(new_tree)
            
            current_angle += angle_step
            if current_angle > 2 * np.pi:
                current_angle = 0
                radius += spacing * 0.5
            
            if radius > n * spacing:  # Safety limit
                break
        
        if len(trees) >= n:
            trees = center_trees(trees[:n])
            if not check_all_overlaps(trees):
                return trees, spacing
    
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
    
    print(f"Baseline total score: {sum(b['score'] for b in baseline.values()):.6f}")
    
    patterns = [
        ('interlocking', interlocking_pattern),
        ('brick', brick_pattern),
        ('spiral', spiral_pattern),
    ]
    
    angles = [0, 15, 30, 45, 60, 75, 90]
    
    improvements = {}
    best_per_n = {n: baseline[n].copy() for n in range(1, 201)}
    
    print("\nTesting construction patterns...")
    
    for pattern_name, pattern_func in patterns:
        print(f"\n{pattern_name.upper()} PATTERN:")
        pattern_improvements = 0
        
        for n in range(2, 31):  # Focus on small N
            best_score = baseline[n]['score']
            best_trees = baseline[n]['trees']
            
            for angle in angles:
                try:
                    trees, spacing = pattern_func(n, angle)
                    
                    if trees is None:
                        continue
                    
                    if check_all_overlaps(trees):
                        continue
                    
                    score = calculate_score(trees, n)
                    
                    if score < best_score - 1e-8:
                        best_score = score
                        best_trees = trees
                        improvement = baseline[n]['score'] - score
                        print(f"  N={n}: {pattern_name} angle={angle} -> {score:.6f} (improvement: {improvement:.6f})")
                        pattern_improvements += 1
                except Exception as e:
                    continue
            
            if best_score < best_per_n[n]['score']:
                best_per_n[n] = {'trees': best_trees, 'score': best_score}
        
        print(f"  Total improvements from {pattern_name}: {pattern_improvements}")
    
    # Calculate total improvement
    new_total = sum(b['score'] for b in best_per_n.values())
    baseline_total = sum(b['score'] for b in baseline.values())
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline total: {baseline_total:.6f}")
    print(f"New total: {new_total:.6f}")
    print(f"Total improvement: {baseline_total - new_total:.6f}")
    
    return best_per_n

if __name__ == '__main__':
    best_per_n = main()
