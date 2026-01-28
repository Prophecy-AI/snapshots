"""
CP-SAT solver for N=2 with precomputed valid configurations.
"""

import numpy as np
from shapely import Polygon
from shapely.affinity import rotate, translate
from ortools.sat.python import cp_model
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

def calculate_score(trees, n):
    bbox = get_bbox_size(trees)
    return bbox**2 / n

def enumerate_valid_n2_configs(angle_step=5, pos_step=0.02):
    """
    Enumerate all valid N=2 configurations with given granularity.
    Returns list of (config, score) tuples.
    """
    print(f"Enumerating valid N=2 configs (angle_step={angle_step}, pos_step={pos_step})")
    
    valid_configs = []
    
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
                        valid_configs.append((trees, score))
        
        if a0 % 45 == 0:
            print(f"  Progress: a0={a0:.0f}, found {len(valid_configs)} valid configs")
    
    print(f"Total valid configs: {len(valid_configs)}")
    return valid_configs

def find_best_n2():
    """Find the best N=2 configuration from enumeration."""
    configs = enumerate_valid_n2_configs(angle_step=5, pos_step=0.02)
    
    # Sort by score
    configs.sort(key=lambda x: x[1])
    
    best_config, best_score = configs[0]
    print(f"\nBest score: {best_score:.8f}")
    print(f"Best config: {best_config}")
    
    # Refine around the best
    print("\nRefining around best config...")
    base_a0 = best_config[0][2]
    base_a1 = best_config[1][2]
    base_dx = best_config[1][0]
    base_dy = best_config[1][1]
    
    for da0 in np.arange(-5, 5.1, 0.5):
        for da1 in np.arange(-5, 5.1, 0.5):
            a0 = base_a0 + da0
            a1 = base_a1 + da1
            tree0 = (0, 0, a0)
            
            for ddx in np.arange(-0.05, 0.051, 0.005):
                for ddy in np.arange(-0.05, 0.051, 0.005):
                    dx = base_dx + ddx
                    dy = base_dy + ddy
                    tree1 = (dx, dy, a1)
                    
                    if not check_overlap(tree0, tree1):
                        trees = [tree0, tree1]
                        score = calculate_score(trees, 2)
                        
                        if score < best_score - 1e-8:
                            best_score = score
                            best_config = trees
                            print(f"  Refined: {score:.8f}")
    
    return best_config, best_score

def main():
    print("="*60)
    print("CP-SAT enumeration for N=2")
    print("="*60)
    
    start = time.time()
    config, score = find_best_n2()
    elapsed = time.time() - start
    
    print(f"\nFinal best score: {score:.8f}")
    print(f"Current baseline: 0.450779")
    print(f"Improvement: {0.450779 - score:.8f}")
    print(f"Time: {elapsed:.2f}s")
    
    # Save results
    with open('results_cpsat.json', 'w') as f:
        json.dump({
            'best_score': score,
            'baseline': 0.450779,
            'improvement': 0.450779 - score,
            'time': elapsed,
            'config': [[list(t) for t in config]]
        }, f, indent=2)

if __name__ == '__main__':
    main()
