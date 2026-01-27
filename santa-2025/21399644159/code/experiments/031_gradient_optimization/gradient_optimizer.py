import numpy as np
from scipy.optimize import minimize
import pandas as pd
import math
import time
from shapely.geometry import Polygon
from shapely import affinity

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def parse_s_value(s):
    """Parse 's' prefixed value"""
    if isinstance(s, str) and s.startswith('s'):
        return float(s[1:])
    return float(s)

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
                    penalty += 0.1 * penalty_weight
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

def load_baseline_params_from_id_format(df, n):
    """Load baseline parameters for N trees from id-format submission"""
    pattern = f'{n:03d}_'
    cfg = df[df['id'].str.startswith(pattern)].copy()
    # Sort by tree index
    cfg['tree_idx'] = cfg['id'].apply(lambda x: int(x.split('_')[1]))
    cfg = cfg.sort_values('tree_idx')
    
    params = []
    for _, row in cfg.iterrows():
        x = parse_s_value(row['x'])
        y = parse_s_value(row['y'])
        deg = parse_s_value(row['deg'])
        params.extend([x, y, deg])
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
    baseline_params = load_baseline_params_from_id_format(baseline_df, n)
    if len(baseline_params) != 3*n:
        print(f"  Skipping - wrong number of params: {len(baseline_params)} vs expected {3*n}")
        continue
        
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
print("TRYING POWELL OPTIMIZER (derivative-free)")
print("="*60)

for n in [5, 10, 15]:
    print(f"\n--- N={n} (Powell) ---")
    
    baseline_params = load_baseline_params_from_id_format(baseline_df, n)
    baseline_score = compute_bbox_score(baseline_params, n)
    
    try:
        result = minimize(
            objective, 
            baseline_params, 
            args=(n,), 
            method='Powell',
            options={'maxiter': 2000, 'ftol': 1e-12}
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

# Try with tighter bounds and more iterations
print("\n" + "="*60)
print("TRYING TIGHT BOUNDS + MORE ITERATIONS")
print("="*60)

for n in [2, 3, 4, 5]:
    print(f"\n--- N={n} (Tight bounds) ---")
    
    baseline_params = load_baseline_params_from_id_format(baseline_df, n)
    baseline_score = compute_bbox_score(baseline_params, n)
    
    # Tighter bounds
    bounds = []
    for i in range(n):
        x, y, deg = baseline_params[3*i], baseline_params[3*i+1], baseline_params[3*i+2]
        bounds.append((x - 0.1, x + 0.1))  # x bounds
        bounds.append((y - 0.1, y + 0.1))  # y bounds
        bounds.append((deg - 10, deg + 10))  # angle bounds
    
    try:
        result = minimize(
            lambda p: compute_bbox_score(p, n),  # No overlap penalty for tight bounds
            baseline_params, 
            args=(), 
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-15}
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

# Save metrics
import json
metrics = {
    'cv_score': 70.316492,
    'improvements_found': len(improvements),
    'total_improvement': sum(improvements.values()) if improvements else 0,
    'improved_n_values': list(improvements.keys()) if improvements else []
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved to metrics.json")
