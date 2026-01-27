import numpy as np
from scipy.optimize import differential_evolution
import pandas as pd
import math
import time
from shapely.geometry import Polygon
from shapely import affinity

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def parse_s_value(s):
    if isinstance(s, str) and s.startswith('s'):
        return float(s[1:])
    return float(s)

def get_tree_polygon(x, y, deg):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = affinity.rotate(poly, deg, origin=(0, 0))
    poly = affinity.translate(poly, x, y)
    return poly

def get_tree_bounds_fast(x, y, deg):
    rad = math.radians(deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    rx = TX * cos_a - TY * sin_a + x
    ry = TX * sin_a + TY * cos_a + y
    return rx.min(), rx.max(), ry.min(), ry.max()

def compute_bbox_score(params, n):
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

def check_overlaps(params, n):
    trees = []
    for i in range(n):
        x, y, deg = params[3*i], params[3*i+1], params[3*i+2]
        trees.append(get_tree_polygon(x, y, deg))
    
    for i in range(n):
        for j in range(i+1, n):
            if trees[i].intersects(trees[j]) and not trees[i].touches(trees[j]):
                return True
    return False

def objective_with_penalty(params, n):
    """Objective with heavy overlap penalty"""
    bbox_score = compute_bbox_score(params, n)
    
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
                    penalty += 1e6 * overlap_area
                except:
                    penalty += 1e3
    
    return bbox_score + penalty

def load_baseline_params_from_id_format(df, n):
    pattern = f'{n:03d}_'
    cfg = df[df['id'].str.startswith(pattern)].copy()
    cfg['tree_idx'] = cfg['id'].apply(lambda x: int(x.split('_')[1]))
    cfg = cfg.sort_values('tree_idx')
    
    params = []
    for _, row in cfg.iterrows():
        x = parse_s_value(row['x'])
        y = parse_s_value(row['y'])
        deg = parse_s_value(row['deg'])
        params.extend([x, y, deg])
    return np.array(params)

# Load baseline
print("Loading baseline submission...")
baseline_df = pd.read_csv('/home/submission/submission.csv')

print("\n" + "="*60)
print("DIFFERENTIAL EVOLUTION (Global Optimizer)")
print("="*60)

improvements = {}
optimized_configs = {}

# Test on small N only (DE is slow)
for n in [2, 3, 4, 5]:
    print(f"\n--- N={n} ---")
    
    baseline_params = load_baseline_params_from_id_format(baseline_df, n)
    baseline_score = compute_bbox_score(baseline_params, n)
    print(f"Baseline score: {baseline_score:.8f}")
    
    # Define bounds around baseline
    bounds = []
    for i in range(n):
        x, y, deg = baseline_params[3*i], baseline_params[3*i+1], baseline_params[3*i+2]
        bounds.append((x - 0.2, x + 0.2))
        bounds.append((y - 0.2, y + 0.2))
        bounds.append((deg - 20, deg + 20))
    
    start_time = time.time()
    
    try:
        result = differential_evolution(
            lambda p: objective_with_penalty(p, n),
            bounds,
            maxiter=200,
            popsize=15,
            tol=1e-10,
            seed=42,
            workers=1,
            updating='deferred'
        )
        
        opt_params = result.x
        opt_score = compute_bbox_score(opt_params, n)
        has_overlaps = check_overlaps(opt_params, n)
        elapsed = time.time() - start_time
        
        print(f"Optimized score: {opt_score:.8f}, overlaps: {has_overlaps}")
        print(f"Time: {elapsed:.1f}s")
        
        if not has_overlaps and opt_score < baseline_score - 1e-10:
            improvements[n] = baseline_score - opt_score
            trees = []
            for i in range(n):
                trees.append((opt_params[3*i], opt_params[3*i+1], opt_params[3*i+2]))
            optimized_configs[n] = trees
            print(f"  ✅ VALID IMPROVEMENT: {baseline_score - opt_score:.10f}")
        else:
            print(f"  ❌ No valid improvement")
            
    except Exception as e:
        print(f"  Error: {e}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

if improvements:
    print(f"\nFound {len(improvements)} N values with valid improvements:")
    total = 0
    for n, imp in sorted(improvements.items()):
        print(f"  N={n}: improved by {imp:.10f}")
        total += imp
    print(f"\nTotal improvement: {total:.10f}")
else:
    print("\nNo valid improvements found with differential evolution")
    print("The current solution appears to be at a global optimum for these N values")

# Update metrics
import json
metrics = {
    'cv_score': 70.316492,
    'improvements_found': len(improvements),
    'total_improvement': sum(improvements.values()) if improvements else 0,
    'improved_n_values': list(improvements.keys()) if improvements else [],
    'methods_tried': ['L-BFGS-B', 'Powell', 'Nelder-Mead', 'barrier_method', 'differential_evolution']
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
