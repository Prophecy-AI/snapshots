"""
Constraint Programming approach for small N values using OR-Tools CP-SAT solver.
Discretize positions and angles, then use CP to find optimal placements.
"""

import numpy as np
from shapely import Polygon
from shapely.affinity import rotate, translate
from ortools.sat.python import cp_model
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
    """Check if two trees overlap (not just touch)"""
    poly1 = get_tree_polygon(*tree1)
    poly2 = get_tree_polygon(*tree2)
    if poly1.intersects(poly2):
        intersection = poly1.intersection(poly2)
        return intersection.area > 1e-10
    return False

def calculate_score(trees, n):
    bbox = get_bbox_size(trees)
    return bbox**2 / n

def exhaustive_search_n2(angle_step=0.5, pos_step=0.01):
    """
    Exhaustive search for N=2 with fine granularity.
    For N=2, we can fix tree 0 at origin and search for tree 1's position.
    """
    print(f"Exhaustive search for N=2 (angle_step={angle_step}, pos_step={pos_step})")
    
    best_score = float('inf')
    best_config = None
    
    # Fix tree 0 at origin with angle 0
    # Search for tree 1's position and angle
    
    # For N=2, the optimal configuration has trees 180 degrees apart
    # Current best: angles 203.63 and 23.63 (180 apart)
    
    # Search around the known optimal
    base_angle1 = 203.63
    base_angle2 = 23.63
    
    tested = 0
    for da1 in np.arange(-10, 10.1, angle_step):
        for da2 in np.arange(-10, 10.1, angle_step):
            angle1 = base_angle1 + da1
            angle2 = base_angle2 + da2
            
            # For each angle pair, find optimal positions
            # Tree 0 at origin
            tree0 = (0, 0, angle1)
            
            # Search for tree 1 position
            for dx in np.arange(-0.5, 0.5, pos_step):
                for dy in np.arange(-0.8, 0.2, pos_step):
                    tree1 = (dx, dy, angle2)
                    
                    if not check_overlap(tree0, tree1):
                        trees = [tree0, tree1]
                        score = calculate_score(trees, 2)
                        tested += 1
                        
                        if score < best_score:
                            best_score = score
                            best_config = trees
                            print(f"  New best: {score:.6f} at angles ({angle1:.2f}, {angle2:.2f}), pos ({dx:.3f}, {dy:.3f})")
    
    print(f"Tested {tested} configurations")
    return best_config, best_score

def cp_search_n2():
    """
    Use CP-SAT solver for N=2 with discretized positions and angles.
    """
    print("CP-SAT search for N=2")
    
    # Discretization parameters
    SCALE = 100  # Position precision: 0.01
    ANGLE_STEPS = 72  # 5 degree steps
    
    # Position bounds (scaled)
    POS_MIN = -100  # -1.0
    POS_MAX = 100   # 1.0
    
    model = cp_model.CpModel()
    
    # Variables for tree 0 (fix at origin for symmetry breaking)
    x0 = model.NewIntVar(0, 0, 'x0')
    y0 = model.NewIntVar(0, 0, 'y0')
    a0 = model.NewIntVar(0, ANGLE_STEPS-1, 'a0')
    
    # Variables for tree 1
    x1 = model.NewIntVar(POS_MIN, POS_MAX, 'x1')
    y1 = model.NewIntVar(POS_MIN, POS_MAX, 'y1')
    a1 = model.NewIntVar(0, ANGLE_STEPS-1, 'a1')
    
    # Bounding box variables
    min_x = model.NewIntVar(POS_MIN - 100, POS_MAX + 100, 'min_x')
    max_x = model.NewIntVar(POS_MIN - 100, POS_MAX + 100, 'max_x')
    min_y = model.NewIntVar(POS_MIN - 100, POS_MAX + 100, 'min_y')
    max_y = model.NewIntVar(POS_MIN - 100, POS_MAX + 100, 'max_y')
    
    # Bounding box size
    width = model.NewIntVar(0, 2 * (POS_MAX - POS_MIN + 200), 'width')
    height = model.NewIntVar(0, 2 * (POS_MAX - POS_MIN + 200), 'height')
    bbox_size = model.NewIntVar(0, 2 * (POS_MAX - POS_MIN + 200), 'bbox_size')
    
    # Constraints: width = max_x - min_x, height = max_y - min_y
    model.Add(width == max_x - min_x)
    model.Add(height == max_y - min_y)
    
    # bbox_size = max(width, height)
    model.AddMaxEquality(bbox_size, [width, height])
    
    # Objective: minimize bbox_size
    model.Minimize(bbox_size)
    
    # Note: The hard part is the no-overlap constraint
    # For CP-SAT, we'd need to precompute which (angle, position) pairs are valid
    # This is complex for non-convex polygons
    
    # For now, let's use a simpler approach: enumerate valid configurations
    print("CP-SAT approach is complex for non-convex polygons.")
    print("Falling back to exhaustive search with finer granularity...")
    
    return None, float('inf')

def ultra_fine_search_n2():
    """
    Ultra-fine exhaustive search for N=2 around the known optimal.
    """
    print("Ultra-fine search for N=2")
    
    # Current best configuration
    base_x0, base_y0, base_a0 = 0.154097, -0.038541, 203.629378
    base_x1, base_y1, base_a1 = -0.154097, -0.561459, 23.629378
    
    best_score = 0.450779  # Current best
    best_config = [(base_x0, base_y0, base_a0), (base_x1, base_y1, base_a1)]
    
    # Search with very fine granularity around the known optimal
    angle_step = 0.01  # 0.01 degree
    pos_step = 0.0001  # 0.0001 position
    
    tested = 0
    improved = 0
    
    for da0 in np.arange(-0.5, 0.51, angle_step):
        for da1 in np.arange(-0.5, 0.51, angle_step):
            a0 = base_a0 + da0
            a1 = base_a1 + da1
            
            # For each angle pair, search positions
            for dx0 in np.arange(-0.01, 0.011, pos_step):
                for dy0 in np.arange(-0.01, 0.011, pos_step):
                    for dx1 in np.arange(-0.01, 0.011, pos_step):
                        for dy1 in np.arange(-0.01, 0.011, pos_step):
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
                                    print(f"  IMPROVEMENT #{improved}: {score:.8f}")
                                    print(f"    Tree 0: ({x0:.6f}, {y0:.6f}, {a0:.6f})")
                                    print(f"    Tree 1: ({x1:.6f}, {y1:.6f}, {a1:.6f})")
    
    print(f"Tested {tested} configurations, found {improved} improvements")
    return best_config, best_score

def gradient_descent_n2():
    """
    Gradient-free optimization for N=2 using scipy.optimize.
    """
    from scipy.optimize import minimize, differential_evolution
    
    print("Gradient-free optimization for N=2")
    
    def objective(params):
        x0, y0, a0, x1, y1, a1 = params
        tree0 = (x0, y0, a0)
        tree1 = (x1, y1, a1)
        
        # Penalty for overlap
        if check_overlap(tree0, tree1):
            return 1000.0
        
        return calculate_score([tree0, tree1], 2)
    
    # Current best as starting point
    x0_init = [0.154097, -0.038541, 203.629378, -0.154097, -0.561459, 23.629378]
    
    # Bounds
    bounds = [
        (-1, 1), (-1, 1), (0, 360),  # Tree 0
        (-1, 1), (-1, 1), (0, 360),  # Tree 1
    ]
    
    # Try differential evolution (global optimizer)
    print("Running differential evolution...")
    result = differential_evolution(
        objective, 
        bounds, 
        seed=42,
        maxiter=1000,
        workers=-1,
        polish=True,
        tol=1e-10
    )
    
    print(f"DE result: {result.fun:.8f}")
    if result.fun < 0.450779:
        print(f"  IMPROVEMENT found!")
        params = result.x
        print(f"  Tree 0: ({params[0]:.6f}, {params[1]:.6f}, {params[2]:.6f})")
        print(f"  Tree 1: ({params[3]:.6f}, {params[4]:.6f}, {params[5]:.6f})")
        return [(params[0], params[1], params[2]), (params[3], params[4], params[5])], result.fun
    
    return None, 0.450779

def main():
    results = {}
    
    # Method 1: Differential Evolution
    print("="*60)
    print("Method 1: Differential Evolution for N=2")
    print("="*60)
    start = time.time()
    config, score = gradient_descent_n2()
    elapsed = time.time() - start
    results['differential_evolution'] = {'score': score, 'time': elapsed, 'config': config}
    print(f"Time: {elapsed:.2f}s, Score: {score:.8f}")
    
    # Method 2: Exhaustive search with coarse granularity
    print("\n" + "="*60)
    print("Method 2: Exhaustive search (coarse)")
    print("="*60)
    start = time.time()
    config, score = exhaustive_search_n2(angle_step=1.0, pos_step=0.02)
    elapsed = time.time() - start
    results['exhaustive_coarse'] = {'score': score, 'time': elapsed}
    print(f"Time: {elapsed:.2f}s, Score: {score:.8f}")
    
    # Save results
    with open('results.json', 'w') as f:
        json.dump({k: {'score': v['score'], 'time': v['time']} for k, v in results.items()}, f, indent=2)
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Current best N=2 score: 0.450779")
    for method, data in results.items():
        improvement = 0.450779 - data['score']
        print(f"{method}: {data['score']:.8f} (improvement: {improvement:.8f})")

if __name__ == '__main__':
    main()
