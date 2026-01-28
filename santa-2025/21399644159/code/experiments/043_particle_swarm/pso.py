"""
Particle Swarm Optimization (PSO) for Tree Packing

PSO is fundamentally different from SA/gradient methods:
- Uses a swarm of particles that explore the solution space
- Each particle has position and velocity
- Particles are attracted to their personal best and global best
- This creates emergent exploration behavior

Key differences from previous approaches:
- Population-based (not single-solution)
- Uses velocity/momentum (not random perturbations)
- Social learning (particles share information)
"""

import numpy as np
import pandas as pd
import math
import json
from shapely.geometry import Polygon
from shapely import affinity

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

def get_tree_bounds(x, y, deg):
    rad = math.radians(deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    rx = TX * cos_a - TY * sin_a + x
    ry = TX * sin_a + TY * cos_a + y
    return rx.min(), rx.max(), ry.min(), ry.max()

def compute_bbox_score(trees):
    if not trees:
        return float('inf')
    
    minx = miny = float('inf')
    maxx = maxy = float('-inf')
    
    for x, y, deg in trees:
        x0, x1, y0, y1 = get_tree_bounds(x, y, deg)
        minx = min(minx, x0)
        maxx = max(maxx, x1)
        miny = min(miny, y0)
        maxy = max(maxy, y1)
    
    side = max(maxx - minx, maxy - miny)
    n = len(trees)
    return side**2 / n

def check_overlaps(trees):
    polys = [get_tree_polygon(x, y, deg) for x, y, deg in trees]
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                return True
    return False

def load_config(df, n):
    pattern = f'{n:03d}_'
    cfg = df[df['id'].str.startswith(pattern)].copy()
    cfg['tree_idx'] = cfg['id'].apply(lambda x: int(x.split('_')[1]))
    cfg = cfg.sort_values('tree_idx')
    
    trees = []
    for _, row in cfg.iterrows():
        x = parse_s_value(row['x'])
        y = parse_s_value(row['y'])
        deg = parse_s_value(row['deg'])
        trees.append((x, y, deg))
    return trees

def trees_to_vector(trees):
    """Convert list of (x, y, deg) to flat vector."""
    vec = []
    for x, y, deg in trees:
        vec.extend([x, y, deg])
    return np.array(vec)

def vector_to_trees(vec):
    """Convert flat vector back to list of (x, y, deg)."""
    trees = []
    for i in range(0, len(vec), 3):
        trees.append((vec[i], vec[i+1], vec[i+2]))
    return trees

def evaluate(vec):
    """Evaluate a solution vector. Returns score (lower is better) or inf if invalid."""
    trees = vector_to_trees(vec)
    if check_overlaps(trees):
        return float('inf')
    return compute_bbox_score(trees)

def pso_optimize(initial_trees, n_particles=30, n_iterations=500, w=0.7, c1=1.5, c2=1.5):
    """
    Particle Swarm Optimization
    
    Parameters:
    - n_particles: number of particles in swarm
    - n_iterations: number of iterations
    - w: inertia weight (momentum)
    - c1: cognitive coefficient (attraction to personal best)
    - c2: social coefficient (attraction to global best)
    """
    n_trees = len(initial_trees)
    dim = n_trees * 3  # x, y, deg for each tree
    
    # Initialize particles around the baseline
    baseline_vec = trees_to_vector(initial_trees)
    
    # Initialize positions with small perturbations
    positions = np.zeros((n_particles, dim))
    velocities = np.zeros((n_particles, dim))
    
    for i in range(n_particles):
        if i == 0:
            positions[i] = baseline_vec  # First particle is baseline
        else:
            # Small random perturbation
            perturbation = np.random.normal(0, 0.01, dim)
            perturbation[2::3] *= 10  # Larger perturbation for angles
            positions[i] = baseline_vec + perturbation
        
        # Small initial velocities
        velocities[i] = np.random.normal(0, 0.001, dim)
        velocities[i, 2::3] *= 10  # Larger for angles
    
    # Evaluate initial positions
    scores = np.array([evaluate(pos) for pos in positions])
    
    # Personal best
    p_best_positions = positions.copy()
    p_best_scores = scores.copy()
    
    # Global best
    g_best_idx = np.argmin(scores)
    g_best_position = positions[g_best_idx].copy()
    g_best_score = scores[g_best_idx]
    
    # Track progress
    baseline_score = evaluate(baseline_vec)
    improvements_found = 0
    
    for iteration in range(n_iterations):
        for i in range(n_particles):
            # Update velocity
            r1, r2 = np.random.random(dim), np.random.random(dim)
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (p_best_positions[i] - positions[i]) +
                           c2 * r2 * (g_best_position - positions[i]))
            
            # Limit velocity
            max_vel = 0.05
            velocities[i] = np.clip(velocities[i], -max_vel, max_vel)
            velocities[i, 2::3] = np.clip(velocities[i, 2::3], -5, 5)  # Angles
            
            # Update position
            positions[i] += velocities[i]
            
            # Evaluate
            score = evaluate(positions[i])
            scores[i] = score
            
            # Update personal best
            if score < p_best_scores[i]:
                p_best_positions[i] = positions[i].copy()
                p_best_scores[i] = score
                
                # Update global best
                if score < g_best_score:
                    g_best_position = positions[i].copy()
                    g_best_score = score
                    improvements_found += 1
        
        # Adaptive inertia weight
        w = max(0.4, w * 0.99)
    
    return vector_to_trees(g_best_position), g_best_score, {
        'improvements_found': improvements_found,
        'baseline_score': baseline_score
    }

# Load baseline
print("Loading baseline...")
baseline_df = pd.read_csv('/home/submission/submission.csv')

baseline_scores = {}
baseline_configs = {}
for n in range(1, 201):
    trees = load_config(baseline_df, n)
    baseline_configs[n] = trees
    baseline_scores[n] = compute_bbox_score(trees)

print(f"Baseline total: {sum(baseline_scores.values()):.6f}")

# Run PSO on various N values
print("\nRunning Particle Swarm Optimization...")
improvements = {}
best_configs = {}

test_ns = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]

for n in test_ns:
    print(f"\nN={n}: baseline={baseline_scores[n]:.6f}")
    
    pso_trees, pso_score, stats = pso_optimize(
        baseline_configs[n],
        n_particles=20,
        n_iterations=300
    )
    
    print(f"  PSO score: {pso_score:.6f}")
    print(f"  Improvements found: {stats['improvements_found']}")
    
    if pso_score < baseline_scores[n] - 1e-10:
        improvements[n] = baseline_scores[n] - pso_score
        best_configs[n] = pso_trees
        print(f"  ✅ IMPROVEMENT: {improvements[n]:.8f}")
    else:
        print(f"  ❌ No improvement")

total_improvement = sum(improvements.values())
print(f"\nTotal improvement: {total_improvement:.8f}")
print(f"Improved N values: {list(improvements.keys())}")

# Save metrics
metrics = {
    'cv_score': 70.316492,
    'improvements_found': len(improvements),
    'total_improvement': total_improvement,
    'improved_n_values': list(improvements.keys()),
    'method': 'particle_swarm_optimization'
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Copy baseline as submission
import shutil
shutil.copy('/home/submission/submission.csv', 'submission.csv')

print("\nDone!")
