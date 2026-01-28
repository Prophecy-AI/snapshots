"""
Constraint Programming approach for small N values using OR-Tools CP-SAT solver.
Discretize positions and angles, then use CP to find optimal placements.
"""

import numpy as np
from shapely import Polygon
from shapely.affinity import rotate, translate
from scipy.optimize import minimize, differential_evolution, basinhopping
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

# Global objective function for N=2
def objective_n2(params):
    x0, y0, a0, x1, y1, a1 = params
    tree0 = (x0, y0, a0)
    tree1 = (x1, y1, a1)
    
    # Penalty for overlap
    if check_overlap(tree0, tree1):
        return 1000.0
    
    return calculate_score([tree0, tree1], 2)

# Global objective function for N=3
def objective_n3(params):
    trees = [(params[i*3], params[i*3+1], params[i*3+2]) for i in range(3)]
    
    # Penalty for overlap
    if check_all_overlaps(trees):
        return 1000.0
    
    return calculate_score(trees, 3)

# Global objective function for N=4
def objective_n4(params):
    trees = [(params[i*3], params[i*3+1], params[i*3+2]) for i in range(4)]
    
    # Penalty for overlap
    if check_all_overlaps(trees):
        return 1000.0
    
    return calculate_score(trees, 4)

def gradient_descent_n2():
    """
    Gradient-free optimization for N=2 using scipy.optimize.
    """
    print("Gradient-free optimization for N=2")
    
    # Current best as starting point
    x0_init = np.array([0.154097, -0.038541, 203.629378, -0.154097, -0.561459, 23.629378])
    
    # Bounds
    bounds = [
        (-1, 1), (-1, 1), (0, 360),  # Tree 0
        (-1, 1), (-1, 1), (0, 360),  # Tree 1
    ]
    
    best_score = 0.450779
    best_config = None
    
    # Try differential evolution (global optimizer) - single worker
    print("Running differential evolution...")
    result = differential_evolution(
        objective_n2, 
        bounds, 
        seed=42,
        maxiter=500,
        workers=1,  # Single worker to avoid pickling issues
        polish=True,
        tol=1e-10,
        x0=x0_init
    )
    
    print(f"DE result: {result.fun:.8f}")
    if result.fun < best_score:
        best_score = result.fun
        params = result.x
        best_config = [(params[0], params[1], params[2]), (params[3], params[4], params[5])]
        print(f"  IMPROVEMENT found!")
        print(f"  Tree 0: ({params[0]:.6f}, {params[1]:.6f}, {params[2]:.6f})")
        print(f"  Tree 1: ({params[3]:.6f}, {params[4]:.6f}, {params[5]:.6f})")
    
    # Try basin hopping
    print("\nRunning basin hopping...")
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
    result = basinhopping(
        objective_n2,
        x0_init,
        minimizer_kwargs=minimizer_kwargs,
        niter=100,
        seed=42
    )
    
    print(f"Basin hopping result: {result.fun:.8f}")
    if result.fun < best_score:
        best_score = result.fun
        params = result.x
        best_config = [(params[0], params[1], params[2]), (params[3], params[4], params[5])]
        print(f"  IMPROVEMENT found!")
    
    return best_config, best_score

def optimize_n3():
    """Optimize N=3 configuration"""
    print("\nOptimizing N=3...")
    
    # Current best configuration
    x0_init = np.array([
        0.273109, -0.239687, 109.738437,
        0.357805, 0.250391, 66.370622,
        -0.234535, 0.154850, 155.133941
    ])
    
    current_score = calculate_score([
        (x0_init[0], x0_init[1], x0_init[2]),
        (x0_init[3], x0_init[4], x0_init[5]),
        (x0_init[6], x0_init[7], x0_init[8])
    ], 3)
    print(f"Current N=3 score: {current_score:.8f}")
    
    bounds = [(-1.5, 1.5), (-1.5, 1.5), (0, 360)] * 3
    
    best_score = current_score
    best_config = None
    
    # Differential evolution
    print("Running differential evolution for N=3...")
    result = differential_evolution(
        objective_n3,
        bounds,
        seed=42,
        maxiter=500,
        workers=1,
        polish=True,
        tol=1e-10,
        x0=x0_init
    )
    
    print(f"DE result: {result.fun:.8f}")
    if result.fun < best_score:
        best_score = result.fun
        params = result.x
        best_config = [(params[i*3], params[i*3+1], params[i*3+2]) for i in range(3)]
        print(f"  IMPROVEMENT: {current_score - best_score:.8f}")
    
    return best_config, best_score, current_score

def optimize_n4():
    """Optimize N=4 configuration"""
    print("\nOptimizing N=4...")
    
    # Load current best from submission
    import pandas as pd
    df = pd.read_csv('/home/submission/submission.csv')
    df['n'] = df['id'].apply(lambda x: int(x.split('_')[0]))
    
    def parse_value(s):
        if isinstance(s, str) and s.startswith('s'):
            return float(s[1:])
        return float(s)
    
    group = df[df['n'] == 4]
    x0_init = []
    for _, row in group.iterrows():
        x0_init.extend([parse_value(row['x']), parse_value(row['y']), parse_value(row['deg'])])
    x0_init = np.array(x0_init)
    
    current_score = calculate_score([
        (x0_init[i*3], x0_init[i*3+1], x0_init[i*3+2]) for i in range(4)
    ], 4)
    print(f"Current N=4 score: {current_score:.8f}")
    
    bounds = [(-2, 2), (-2, 2), (0, 360)] * 4
    
    best_score = current_score
    best_config = None
    
    # Differential evolution
    print("Running differential evolution for N=4...")
    result = differential_evolution(
        objective_n4,
        bounds,
        seed=42,
        maxiter=500,
        workers=1,
        polish=True,
        tol=1e-10,
        x0=x0_init
    )
    
    print(f"DE result: {result.fun:.8f}")
    if result.fun < best_score:
        best_score = result.fun
        params = result.x
        best_config = [(params[i*3], params[i*3+1], params[i*3+2]) for i in range(4)]
        print(f"  IMPROVEMENT: {current_score - best_score:.8f}")
    
    return best_config, best_score, current_score

def main():
    results = {}
    
    # Optimize N=2
    print("="*60)
    print("Optimizing N=2")
    print("="*60)
    start = time.time()
    config, score = gradient_descent_n2()
    elapsed = time.time() - start
    results['n2'] = {
        'score': score, 
        'time': elapsed, 
        'config': config,
        'baseline': 0.450779,
        'improvement': 0.450779 - score
    }
    print(f"Time: {elapsed:.2f}s, Score: {score:.8f}, Improvement: {0.450779 - score:.8f}")
    
    # Optimize N=3
    print("\n" + "="*60)
    print("Optimizing N=3")
    print("="*60)
    start = time.time()
    config, score, baseline = optimize_n3()
    elapsed = time.time() - start
    results['n3'] = {
        'score': score,
        'time': elapsed,
        'config': config,
        'baseline': baseline,
        'improvement': baseline - score
    }
    print(f"Time: {elapsed:.2f}s, Score: {score:.8f}, Improvement: {baseline - score:.8f}")
    
    # Optimize N=4
    print("\n" + "="*60)
    print("Optimizing N=4")
    print("="*60)
    start = time.time()
    config, score, baseline = optimize_n4()
    elapsed = time.time() - start
    results['n4'] = {
        'score': score,
        'time': elapsed,
        'config': config,
        'baseline': baseline,
        'improvement': baseline - score
    }
    print(f"Time: {elapsed:.2f}s, Score: {score:.8f}, Improvement: {baseline - score:.8f}")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    total_improvement = sum(r['improvement'] for r in results.values())
    print(f"Total improvement: {total_improvement:.8f}")
    for n, data in results.items():
        print(f"  {n}: {data['baseline']:.6f} -> {data['score']:.6f} (improvement: {data['improvement']:.8f})")
    
    # Save results
    with open('results.json', 'w') as f:
        json.dump({
            k: {
                'score': v['score'], 
                'time': v['time'],
                'baseline': v['baseline'],
                'improvement': v['improvement']
            } for k, v in results.items()
        }, f, indent=2)
    
    return results

if __name__ == '__main__':
    main()
