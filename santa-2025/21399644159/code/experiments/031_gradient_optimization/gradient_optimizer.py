import numpy as np
from scipy.optimize import minimize, differential_evolution
import pandas as pd
import math
import time
from shapely.geometry import Polygon
from shapely import affinity

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def get_tree_polygon(x, y, deg):
    """Get shapely polygon for a tree at (x, y) with rotation deg"""
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = affinity.rotate(poly, deg, origin=(0, 0))
    poly = affinity.translate(poly, x, y)
    return poly

def get_tree_bounds_fast(x, y, deg):
    """Fast bounding box calculation using numpy"""
    rad = math.radians(deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    rx = TX * cos_a - TY * sin_a + x
    ry = TX * sin_a + TY * cos_a + y
    return rx.min(), rx.max(), ry.min(), ry.max()

def compute_bbox_score(params, n):
    """Compute bounding box score from parameters [x0,y0,deg0,x1,y1,deg1,...]"""
    minx = miny = float('inf')
    maxx = maxy = float('-inf')
    
    for i in range(n):
        x, y, deg = params[3*i], params[3*i+1], params[3*i+2]
        x0, x1, y0, y1 = get_tree_bounds_fast(x, y, deg)
        minx = min(minx, x0)
        maxx = max(maxx, x1)
        miny = min(miny, y0)
        maxy = max(maxy, y1)
    
    side = max(maxx - minx, maxy - miny)
    return side**2 / n

def compute_overlap_penalty(params, n, penalty_weight=10000):
    """Penalty for overlapping trees using shapely"""
    trees = []
    for i in range(n):
        x, y, deg = params[3*i], params[3*i+1], params[3*i+2]
        trees.append(get_tree_polygon(x, y, deg))
    
    penalty = 0
    for i in range(n):
        for j in range(i+1, n):
            if trees[i].intersects(trees[j]) and not trees[i].touches(trees[j]):
                try:
                    overlap_area = trees[i].intersection(trees[j]).area
                    penalty += overlap_area * penalty_weight
                except:
                    penalty += 0.1 * penalty_weight  # Small penalty for invalid geometry
    return penalty

def check_overlaps(params, n):
    """Check if any trees overlap"""
    trees = []
    for i in range(n):
        x, y, deg = params[3*i], params[3*i+1], params[3*i+2]
        trees.append(get_tree_polygon(x, y, deg))
    
    for i in range(n):
        for j in range(i+1, n):
            if trees[i].intersects(trees[j]) and not trees[i].touches(trees[j]):
                return True
    return False

def objective(params, n):
    """Combined objective: bbox score + overlap penalty"""
    return compute_bbox_score(params, n) + compute_overlap_penalty(params, n)

def load_baseline_params(df, n):
    """Load baseline parameters for N trees from submission dataframe"""
    cfg = df[df['n'] == n].sort_values('i')
    params = []
    for _, row in cfg.iterrows():
        params.extend([row['x'], row['y'], row['deg']])
    return np.array(params)

def params_to_trees(params, n):
    """Convert params back to list of (x, y, deg) tuples"""
    trees = []
    for i in range(n):
        trees.append((params[3*i], params[3*i+1], params[3*i+2]))
    return trees

# Load baseline submission
print("Loading baseline submission...")
baseline_df = pd.read_csv('/home/submission/submission.csv')

# Test gradient optimization on small N values
test_ns = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
improvements = {}
optimized_configs = {}

print("\n" + "="*60)
print("GRADIENT-BASED OPTIMIZATION (L-BFGS-B)")
print("="*60)

for n in test_ns:
    print(f"\n--- N={n} ---")
    
    # Load baseline
    baseline_params = load_baseline_params(baseline_df, n)
    baseline_score = compute_bbox_score(baseline_params, n)
    print(f"Baseline score: {baseline_score:.8f}")
    
    # Try L-BFGS-B optimization
    start_time = time.time()
    
    # Define bounds (allow some movement around baseline)
    bounds = []
    for i in range(n):
        x, y, deg = baseline_params[3*i], baseline_params[3*i+1], baseline_params[3*i+2]
        bounds.append((x - 0.5, x + 0.5))  # x bounds
        bounds.append((y - 0.5, y + 0.5))  # y bounds
        bounds.append((deg - 45, deg + 45))  # angle bounds
    
    try:
        result = minimize(
            objective, 
            baseline_params, 
            args=(n,), 
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        
        optimized_params = result.x
        optimized_score = compute_bbox_score(optimized_params, n)
        has_overlaps = check_overlaps(optimized_params, n)
        
        elapsed = time.time() - start_time
        
        print(f"Optimized score: {optimized_score:.8f} (overlaps: {has_overlaps})")
        print(f"Improvement: {baseline_score - optimized_score:.10f}")
        print(f"Time: {elapsed:.2f}s, Iterations: {result.nit}")
        
        if optimized_score < baseline_score - 1e-10 and not has_overlaps:
            improvements[n] = baseline_score - optimized_score
            optimized_configs[n] = params_to_trees(optimized_params, n)
            print(f"  ✅ VALID IMPROVEMENT!")
        elif optimized_score < baseline_score - 1e-10 and has_overlaps:
            print(f"  ⚠️ Improvement found but has overlaps")
        else:
            print(f"  ❌ No improvement")
            
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

if improvements:
    print(f"\nFound {len(improvements)} N values with valid improvements:")
    total_improvement = 0
    for n, imp in sorted(improvements.items()):
        print(f"  N={n}: improved by {imp:.10f}")
        total_improvement += imp
    print(f"\nTotal improvement: {total_improvement:.10f}")
else:
    print("\nNo valid improvements found with L-BFGS-B")

# Try SLSQP as alternative
print("\n" + "="*60)
print("TRYING SLSQP OPTIMIZER")
print("="*60)

for n in [5, 10, 15]:
    print(f"\n--- N={n} (SLSQP) ---")
    
    baseline_params = load_baseline_params(baseline_df, n)
    baseline_score = compute_bbox_score(baseline_params, n)
    
    bounds = []
    for i in range(n):
        x, y, deg = baseline_params[3*i], baseline_params[3*i+1], baseline_params[3*i+2]
        bounds.append((x - 0.5, x + 0.5))
        bounds.append((y - 0.5, y + 0.5))
        bounds.append((deg - 45, deg + 45))
    
    try:
        result = minimize(
            objective, 
            baseline_params, 
            args=(n,), 
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        
        optimized_params = result.x
        optimized_score = compute_bbox_score(optimized_params, n)
        has_overlaps = check_overlaps(optimized_params, n)
        
        print(f"Baseline: {baseline_score:.8f}, Optimized: {optimized_score:.8f}")
        print(f"Improvement: {baseline_score - optimized_score:.10f}, Overlaps: {has_overlaps}")
        
        if optimized_score < baseline_score - 1e-10 and not has_overlaps:
            if n not in improvements or (baseline_score - optimized_score) > improvements[n]:
                improvements[n] = baseline_score - optimized_score
                optimized_configs[n] = params_to_trees(optimized_params, n)
                print(f"  ✅ VALID IMPROVEMENT!")
                
    except Exception as e:
        print(f"  Error: {e}")

# Try Nelder-Mead (derivative-free)
print("\n" + "="*60)
print("TRYING NELDER-MEAD OPTIMIZER (derivative-free)")
print("="*60)

for n in [5, 10]:
    print(f"\n--- N={n} (Nelder-Mead) ---")
    
    baseline_params = load_baseline_params(baseline_df, n)
    baseline_score = compute_bbox_score(baseline_params, n)
    
    try:
        result = minimize(
            objective, 
            baseline_params, 
            args=(n,), 
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        
        optimized_params = result.x
        optimized_score = compute_bbox_score(optimized_params, n)
        has_overlaps = check_overlaps(optimized_params, n)
        
        print(f"Baseline: {baseline_score:.8f}, Optimized: {optimized_score:.8f}")
        print(f"Improvement: {baseline_score - optimized_score:.10f}, Overlaps: {has_overlaps}")
        
        if optimized_score < baseline_score - 1e-10 and not has_overlaps:
            if n not in improvements or (baseline_score - optimized_score) > improvements[n]:
                improvements[n] = baseline_score - optimized_score
                optimized_configs[n] = params_to_trees(optimized_params, n)
                print(f"  ✅ VALID IMPROVEMENT!")
                
    except Exception as e:
        print(f"  Error: {e}")

# Final summary
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

if improvements:
    print(f"\nFound {len(improvements)} N values with valid improvements:")
    total_improvement = 0
    for n, imp in sorted(improvements.items()):
        print(f"  N={n}: improved by {imp:.10f}")
        total_improvement += imp
    print(f"\nTotal improvement: {total_improvement:.10f}")
    
    # Save optimized configs
    import json
    with open('improvements.json', 'w') as f:
        json.dump({
            'improvements': {str(k): v for k, v in improvements.items()},
            'configs': {str(k): v for k, v in optimized_configs.items()}
        }, f, indent=2)
else:
    print("\nNo valid improvements found with any gradient-based optimizer")
    print("The current solution is at a local optimum that gradient methods cannot escape")
