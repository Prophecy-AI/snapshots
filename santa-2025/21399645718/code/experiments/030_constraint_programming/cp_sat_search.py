"""
Use OR-Tools CP-SAT solver with precomputed valid configurations.
For small N, enumerate all valid (non-overlapping) configurations and find the minimum.
"""

import numpy as np
from shapely import Polygon
from shapely.affinity import rotate, translate
from ortools.sat.python import cp_model
import time
import json
from itertools import product
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
    """Check if two trees overlap (not just touch)"""
    poly1 = get_tree_polygon(*tree1)
    poly2 = get_tree_polygon(*tree2)
    if poly1.intersects(poly2):
        intersection = poly1.intersection(poly2)
        return intersection.area > 1e-10
    return False

def check_all_overlaps(trees):
    """Check if any pair of trees overlaps"""
    for i in range(len(trees)):
        for j in range(i+1, len(trees)):
            if check_overlap(trees[i], trees[j]):
                return True
    return False

def calculate_score(trees, n):
    bbox = get_bbox_size(trees)
    return bbox**2 / n

def exhaustive_n2_fine():
    """
    Exhaustive search for N=2 with very fine granularity.
    Fix tree 0 at origin, search for tree 1's optimal position.
    """
    print("Exhaustive search for N=2 with fine granularity")
    
    # Current best
    best_score = 0.450779
    best_config = None
    
    # The key insight: for N=2, the optimal has trees 180 degrees apart
    # Current: angles 203.63 and 23.63
    
    # Search all angle pairs with 1 degree step
    angle_step = 1.0
    pos_step = 0.005
    
    tested = 0
    improved = 0
    
    # Fix tree 0 at origin
    for a0 in np.arange(0, 360, angle_step):
        a1 = (a0 + 180) % 360  # Trees 180 degrees apart
        
        tree0 = (0, 0, a0)
        
        # Search for tree 1 position
        for dx in np.arange(-0.6, 0.6, pos_step):
            for dy in np.arange(-0.8, 0.2, pos_step):
                tree1 = (dx, dy, a1)
                
                if not check_overlap(tree0, tree1):
                    trees = [tree0, tree1]
                    score = calculate_score(trees, 2)
                    tested += 1
                    
                    if score < best_score - 1e-8:
                        best_score = score
                        best_config = trees
                        improved += 1
                        print(f"  IMPROVEMENT #{improved}: {score:.8f} at a0={a0:.1f}, pos=({dx:.3f}, {dy:.3f})")
        
        if a0 % 30 == 0:
            print(f"  Progress: a0={a0:.0f}, tested={tested}, best={best_score:.8f}")
    
    print(f"Tested {tested} configurations, found {improved} improvements")
    return best_config, best_score

def exhaustive_n2_all_angles():
    """
    Exhaustive search for N=2 with all angle combinations.
    """
    print("Exhaustive search for N=2 with all angle combinations")
    
    # Current best
    best_score = 0.450779
    best_config = None
    
    angle_step = 5.0  # Coarser for all combinations
    pos_step = 0.01
    
    tested = 0
    improved = 0
    
    # Fix tree 0 at origin
    for a0 in np.arange(0, 360, angle_step):
        for a1 in np.arange(0, 360, angle_step):
            tree0 = (0, 0, a0)
            
            # Search for tree 1 position
            for dx in np.arange(-0.6, 0.6, pos_step):
                for dy in np.arange(-0.8, 0.2, pos_step):
                    tree1 = (dx, dy, a1)
                    
                    if not check_overlap(tree0, tree1):
                        trees = [tree0, tree1]
                        score = calculate_score(trees, 2)
                        tested += 1
                        
                        if score < best_score - 1e-8:
                            best_score = score
                            best_config = trees
                            improved += 1
                            print(f"  IMPROVEMENT #{improved}: {score:.8f} at angles ({a0:.1f}, {a1:.1f}), pos=({dx:.3f}, {dy:.3f})")
        
        if a0 % 45 == 0:
            print(f"  Progress: a0={a0:.0f}, tested={tested}, best={best_score:.8f}")
    
    print(f"Tested {tested} configurations, found {improved} improvements")
    return best_config, best_score

def search_around_optimal_n2():
    """
    Very fine search around the known optimal for N=2.
    """
    print("Fine search around optimal for N=2")
    
    # Current best configuration
    base_x0, base_y0, base_a0 = 0.154097, -0.038541, 203.629378
    base_x1, base_y1, base_a1 = -0.154097, -0.561459, 23.629378
    
    best_score = 0.450779
    best_config = [(base_x0, base_y0, base_a0), (base_x1, base_y1, base_a1)]
    
    # Very fine search
    angle_step = 0.1
    pos_step = 0.001
    
    tested = 0
    improved = 0
    
    for da0 in np.arange(-2, 2.01, angle_step):
        for da1 in np.arange(-2, 2.01, angle_step):
            a0 = base_a0 + da0
            a1 = base_a1 + da1
            
            # Search positions
            for dx0 in np.arange(-0.05, 0.051, pos_step):
                for dy0 in np.arange(-0.05, 0.051, pos_step):
                    for dx1 in np.arange(-0.05, 0.051, pos_step):
                        for dy1 in np.arange(-0.05, 0.051, pos_step):
                            x0 = base_x0 + dx0
                            y0 = base_y0 + dy0
                            x1 = base_x1 + dx1
                            y1 = base_y1 + dy1
                            
                            tree0 = (x0, y0, a0)
                            tree1 = (x1, y1, a1)
                            
                            if not check_overlap(tree0, tree1):
                                trees = [tree0, tree1]
                                score = calculate_score(trees, 2)
                                tested += 1
                                
                                if score < best_score - 1e-8:
                                    best_score = score
                                    best_config = trees
                                    improved += 1
                                    print(f"  IMPROVEMENT #{improved}: {score:.10f}")
        
        if da0 % 0.5 < 0.01:
            print(f"  Progress: da0={da0:.1f}, tested={tested}, best={best_score:.8f}")
    
    print(f"Tested {tested} configurations, found {improved} improvements")
    return best_config, best_score

def main():
    results = {}
    
    # Method 1: Exhaustive with 180-degree constraint
    print("="*60)
    print("Method 1: Exhaustive with 180-degree constraint")
    print("="*60)
    start = time.time()
    config, score = exhaustive_n2_fine()
    elapsed = time.time() - start
    results['exhaustive_180'] = {
        'score': score,
        'time': elapsed,
        'improvement': 0.450779 - score
    }
    print(f"Time: {elapsed:.2f}s, Score: {score:.8f}, Improvement: {0.450779 - score:.8f}")
    
    # Method 2: All angle combinations (coarser)
    print("\n" + "="*60)
    print("Method 2: All angle combinations")
    print("="*60)
    start = time.time()
    config, score = exhaustive_n2_all_angles()
    elapsed = time.time() - start
    results['exhaustive_all'] = {
        'score': score,
        'time': elapsed,
        'improvement': 0.450779 - score
    }
    print(f"Time: {elapsed:.2f}s, Score: {score:.8f}, Improvement: {0.450779 - score:.8f}")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Current best N=2 score: 0.450779")
    for method, data in results.items():
        print(f"  {method}: {data['score']:.8f} (improvement: {data['improvement']:.8f})")
    
    # Save results
    with open('results_exhaustive.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
