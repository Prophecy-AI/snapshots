"""
Smart search for small N values using known optimal structures.
"""

import numpy as np
from shapely import Polygon
from shapely.affinity import rotate, translate
import time
import json
import warnings
warnings.filterwarnings('ignore')

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

def optimize_n2_smart():
    """
    Smart optimization for N=2.
    Key insight: optimal has trees 180 degrees apart.
    """
    print("Smart optimization for N=2")
    
    # Current best
    best_score = 0.450779
    best_config = None
    
    # For N=2, search all base angles with 180-degree offset
    for base_angle in np.arange(0, 180, 0.5):
        a0 = base_angle
        a1 = base_angle + 180
        
        tree0 = (0, 0, a0)
        
        # Search for optimal position
        for dx in np.arange(-0.4, 0.4, 0.01):
            for dy in np.arange(-0.7, 0.1, 0.01):
                tree1 = (dx, dy, a1)
                
                if not check_overlap(tree0, tree1):
                    trees = [tree0, tree1]
                    score = calculate_score(trees, 2)
                    
                    if score < best_score - 1e-8:
                        best_score = score
                        best_config = trees
                        print(f"  New best: {score:.8f} at base_angle={base_angle:.1f}, pos=({dx:.3f}, {dy:.3f})")
    
    print(f"\nBest score: {best_score:.8f}")
    print(f"Improvement: {0.450779 - best_score:.8f}")
    
    return best_config, best_score

def optimize_n2_gradient():
    """
    Gradient-based refinement for N=2.
    """
    print("\nGradient-based refinement for N=2")
    
    # Start from current best
    x0, y0, a0 = 0.154097, -0.038541, 203.629378
    x1, y1, a1 = -0.154097, -0.561459, 23.629378
    
    best_score = 0.450779
    best_config = [(x0, y0, a0), (x1, y1, a1)]
    
    # Gradient descent with finite differences
    step_size = 0.001
    angle_step = 0.01
    
    for iteration in range(1000):
        improved = False
        
        # Try small perturbations
        for var_idx in range(6):
            for delta in [-step_size, step_size] if var_idx < 2 or var_idx in [3, 4] else [-angle_step, angle_step]:
                params = [x0, y0, a0, x1, y1, a1]
                params[var_idx] += delta
                
                tree0 = (params[0], params[1], params[2])
                tree1 = (params[3], params[4], params[5])
                
                if not check_overlap(tree0, tree1):
                    trees = [tree0, tree1]
                    score = calculate_score(trees, 2)
                    
                    if score < best_score - 1e-10:
                        best_score = score
                        x0, y0, a0, x1, y1, a1 = params
                        best_config = [tree0, tree1]
                        improved = True
                        break
            
            if improved:
                break
        
        if not improved:
            break
    
    print(f"Final score: {best_score:.10f}")
    print(f"Improvement: {0.450779 - best_score:.10f}")
    
    return best_config, best_score

def main():
    results = {}
    
    # Method 1: Smart search with 180-degree constraint
    print("="*60)
    print("Method 1: Smart search with 180-degree constraint")
    print("="*60)
    start = time.time()
    config, score = optimize_n2_smart()
    elapsed = time.time() - start
    results['smart_180'] = {
        'score': score,
        'time': elapsed,
        'improvement': 0.450779 - score
    }
    
    # Method 2: Gradient refinement
    print("\n" + "="*60)
    print("Method 2: Gradient refinement")
    print("="*60)
    start = time.time()
    config, score = optimize_n2_gradient()
    elapsed = time.time() - start
    results['gradient'] = {
        'score': score,
        'time': elapsed,
        'improvement': 0.450779 - score
    }
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Current best N=2 score: 0.450779")
    for method, data in results.items():
        print(f"  {method}: {data['score']:.10f} (improvement: {data['improvement']:.10f})")
    
    # Save results
    with open('results_smart.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
