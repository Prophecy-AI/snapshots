"""
Lattice-based construction for specific N values.
Based on egortrushin kernel approach: use two base trees translated in x and y directions.
"""
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely import affinity
import random
import math
import os

TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])
BASE_TREE = Polygon(zip(TX, TY))

def parse_value(val):
    if isinstance(val, str):
        if val.startswith('s'):
            return float(val[1:])
        return float(val)
    return float(val)

def ensure_s_prefix(val):
    return f's{val}'

def create_tree(x, y, deg):
    tree = affinity.rotate(BASE_TREE, deg, origin=(0, 0))
    tree = affinity.translate(tree, x, y)
    return tree

def get_side(trees):
    if not trees:
        return 0
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    for tree in trees:
        bounds = tree.bounds
        min_x = min(min_x, bounds[0])
        min_y = min(min_y, bounds[1])
        max_x = max(max_x, bounds[2])
        max_y = max(max_y, bounds[3])
    return max(max_x - min_x, max_y - min_y)

def check_any_overlap(trees, tolerance=1e-9):
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            if trees[i].intersects(trees[j]):
                intersection = trees[i].intersection(trees[j])
                if intersection.area > tolerance:
                    return True
    return False

def create_lattice_config(nx, ny, dx, dy, angle1, angle2, offset_x=0, offset_y=0):
    """
    Create a lattice configuration with nx * ny trees.
    Two base trees with different angles, translated in x and y directions.
    """
    configs = []
    for i in range(nx):
        for j in range(ny):
            x = i * dx + offset_x
            y = j * dy + offset_y
            # Alternate angles in checkerboard pattern
            if (i + j) % 2 == 0:
                angle = angle1
            else:
                angle = angle2
            configs.append((x, y, angle))
    return configs

def optimize_lattice(nx, ny, n_target, max_iters=5000):
    """
    Optimize lattice parameters using simulated annealing.
    """
    # Initial parameters
    dx = 0.5
    dy = 0.8
    angle1 = 68.0
    angle2 = 248.0
    offset_x = 0.0
    offset_y = 0.0
    
    # Create initial config
    configs = create_lattice_config(nx, ny, dx, dy, angle1, angle2, offset_x, offset_y)
    if len(configs) > n_target:
        configs = configs[:n_target]
    
    trees = [create_tree(x, y, deg) for x, y, deg in configs]
    
    if check_any_overlap(trees):
        # Increase spacing until no overlap
        for scale in [1.1, 1.2, 1.3, 1.5, 2.0]:
            configs = create_lattice_config(nx, ny, dx*scale, dy*scale, angle1, angle2, offset_x, offset_y)
            if len(configs) > n_target:
                configs = configs[:n_target]
            trees = [create_tree(x, y, deg) for x, y, deg in configs]
            if not check_any_overlap(trees):
                dx *= scale
                dy *= scale
                break
    
    best_score = get_side(trees)**2 / n_target
    best_params = (dx, dy, angle1, angle2, offset_x, offset_y)
    
    # Simulated annealing
    T = 1.0
    T_min = 0.001
    alpha = 0.995
    
    current_params = best_params
    current_score = best_score
    
    for iteration in range(max_iters):
        # Perturb parameters
        new_dx = current_params[0] + random.gauss(0, 0.01 * T)
        new_dy = current_params[1] + random.gauss(0, 0.01 * T)
        new_angle1 = (current_params[2] + random.gauss(0, 5 * T)) % 360
        new_angle2 = (current_params[3] + random.gauss(0, 5 * T)) % 360
        new_offset_x = current_params[4] + random.gauss(0, 0.01 * T)
        new_offset_y = current_params[5] + random.gauss(0, 0.01 * T)
        
        new_params = (new_dx, new_dy, new_angle1, new_angle2, new_offset_x, new_offset_y)
        
        # Create new config
        configs = create_lattice_config(nx, ny, new_dx, new_dy, new_angle1, new_angle2, new_offset_x, new_offset_y)
        if len(configs) > n_target:
            configs = configs[:n_target]
        
        trees = [create_tree(x, y, deg) for x, y, deg in configs]
        
        if check_any_overlap(trees):
            continue
        
        new_score = get_side(trees)**2 / n_target
        
        # Accept or reject
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_params = new_params
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_params = new_params
        
        T = max(T * alpha, T_min)
    
    # Return best configuration
    configs = create_lattice_config(nx, ny, best_params[0], best_params[1], 
                                    best_params[2], best_params[3],
                                    best_params[4], best_params[5])
    if len(configs) > n_target:
        configs = configs[:n_target]
    
    return configs, best_score

# Load current best submission
print("Loading current submission...")
df = pd.read_csv('/home/submission/submission.csv')

# Get current scores
current_scores = {}
for n in range(1, 201):
    prefix = f"{n:03d}_"
    n_rows = df[df['id'].str.startswith(prefix)]
    trees = []
    for _, row in n_rows.iterrows():
        x = parse_value(row['x'])
        y = parse_value(row['y'])
        deg = parse_value(row['deg'])
        trees.append(create_tree(x, y, deg))
    side = get_side(trees)
    current_scores[n] = side**2 / n

current_total = sum(current_scores.values())
print(f"Current total score: {current_total:.6f}")

# Try lattice construction for specific N values
# These are the N values from egortrushin kernel
lattice_configs = {
    72: (6, 12),   # 6x12 = 72
    100: (10, 10), # 10x10 = 100
    110: (10, 11), # 10x11 = 110
    144: (12, 12), # 12x12 = 144
    156: (12, 13), # 12x13 = 156
    196: (14, 14), # 14x14 = 196
    200: (10, 20), # 10x20 = 200
}

# Also try for smaller N values
for n in range(4, 50):
    # Find good factorizations
    for nx in range(2, int(math.sqrt(n)) + 1):
        if n % nx == 0:
            ny = n // nx
            if (nx, ny) not in lattice_configs.values():
                lattice_configs[n] = (nx, ny)
                break

print(f"\nTrying lattice construction for {len(lattice_configs)} N values...")

improvements = {}
for n, (nx, ny) in sorted(lattice_configs.items()):
    print(f"\nN={n} ({nx}x{ny}):")
    
    configs, score = optimize_lattice(nx, ny, n, max_iters=3000)
    
    if score < current_scores[n]:
        improvement = current_scores[n] - score
        improvements[n] = (configs, score, improvement)
        print(f"  IMPROVED! {current_scores[n]:.6f} -> {score:.6f} (diff: {improvement:.6f})")
    else:
        print(f"  No improvement: {current_scores[n]:.6f} vs {score:.6f}")

# Calculate total improvement
if improvements:
    total_improvement = sum(imp[2] for imp in improvements.values())
    print(f"\nTotal improvement: {total_improvement:.6f}")
    
    # Save improved submission
    os.makedirs('/home/code/experiments/005_lattice', exist_ok=True)
    
    # Load original configs
    original_configs = {}
    for n in range(1, 201):
        prefix = f"{n:03d}_"
        n_rows = df[df['id'].str.startswith(prefix)]
        original_configs[n] = []
        for _, row in n_rows.iterrows():
            x = parse_value(row['x'])
            y = parse_value(row['y'])
            deg = parse_value(row['deg'])
            original_configs[n].append((x, y, deg))
    
    # Replace improved configs
    for n, (configs, score, improvement) in improvements.items():
        original_configs[n] = configs
    
    # Save
    rows = []
    for n in range(1, 201):
        for i, (x, y, deg) in enumerate(original_configs[n]):
            rows.append({
                'id': f'{n:03d}_{i}',
                'x': ensure_s_prefix(x),
                'y': ensure_s_prefix(y),
                'deg': ensure_s_prefix(deg)
            })
    
    df_out = pd.DataFrame(rows)
    df_out.to_csv('/home/code/experiments/005_lattice/submission.csv', index=False)
    print(f"\nSaved to /home/code/experiments/005_lattice/submission.csv")
else:
    print("\nNo improvements from lattice construction")
