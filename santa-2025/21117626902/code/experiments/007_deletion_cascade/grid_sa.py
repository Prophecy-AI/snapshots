"""
Grid-based SA optimization from jiweiliu kernel.
Generates novel solutions by starting from grid configurations.
"""
import math
import os
import time
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List as NumbaList

print("Grid-based SA optimization from jiweiliu kernel")

# Tree shape constants
TRUNK_W = 0.15
TRUNK_H = 0.2
BASE_W = 0.7
MID_W = 0.4
TOP_W = 0.25
TIP_Y = 0.8
TIER_1_Y = 0.5
TIER_2_Y = 0.25
BASE_Y = 0.0
TRUNK_BOTTOM_Y = -TRUNK_H

MAX_OVERLAP_DIST = 1.8
MAX_OVERLAP_DIST_SQ = MAX_OVERLAP_DIST * MAX_OVERLAP_DIST

@njit
def rotate_point(x, y, cos_a, sin_a):
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a

@njit
def get_tree_vertices(cx, cy, angle_deg):
    """Get 15 vertices of tree polygon at given position and angle."""
    angle_rad = angle_deg * math.pi / 180.0
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    vertices = np.empty((15, 2), dtype=np.float64)
    pts = np.array([
        [0.0, TIP_Y],
        [TOP_W / 2.0, TIER_1_Y],
        [TOP_W / 4.0, TIER_1_Y],
        [MID_W / 2.0, TIER_2_Y],
        [MID_W / 4.0, TIER_2_Y],
        [BASE_W / 2.0, BASE_Y],
        [TRUNK_W / 2.0, BASE_Y],
        [TRUNK_W / 2.0, TRUNK_BOTTOM_Y],
        [-TRUNK_W / 2.0, TRUNK_BOTTOM_Y],
        [-TRUNK_W / 2.0, BASE_Y],
        [-BASE_W / 2.0, BASE_Y],
        [-MID_W / 4.0, TIER_2_Y],
        [-MID_W / 2.0, TIER_2_Y],
        [-TOP_W / 4.0, TIER_1_Y],
        [-TOP_W / 2.0, TIER_1_Y],
    ], dtype=np.float64)
    for i in range(15):
        rx, ry = rotate_point(pts[i, 0], pts[i, 1], cos_a, sin_a)
        vertices[i, 0] = rx + cx
        vertices[i, 1] = ry + cy
    return vertices

@njit
def polygon_bounds(vertices):
    """Get bounding box of polygon vertices."""
    min_x = vertices[0, 0]
    min_y = vertices[0, 1]
    max_x = vertices[0, 0]
    max_y = vertices[0, 1]
    for i in range(1, vertices.shape[0]):
        x = vertices[i, 0]
        y = vertices[i, 1]
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y
    return min_x, min_y, max_x, max_y

@njit
def point_in_polygon(px, py, vertices):
    """Check if point is inside polygon using ray casting."""
    n = vertices.shape[0]
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i, 0], vertices[i, 1]
        xj, yj = vertices[j, 0], vertices[j, 1]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

@njit
def segments_intersect(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
    """Check if two line segments intersect using cross-product method."""
    dax = p2x - p1x
    day = p2y - p1y
    dbx = p4x - p3x
    dby = p4y - p3y
    d1x = p1x - p3x
    d1y = p1y - p3y
    d2x = p2x - p3x
    d2y = p2y - p3y
    cross_b1 = dbx * d1y - dby * d1x
    cross_b2 = dbx * d2y - dby * d2x
    if cross_b1 * cross_b2 > 0:
        return False
    d3x = p3x - p1x
    d3y = p3y - p1y
    d4x = p4x - p1x
    d4y = p4y - p1y
    cross_a1 = dax * d3y - day * d3x
    cross_a2 = dax * d4y - day * d4x
    if cross_a1 * cross_a2 > 0:
        return False
    return True

@njit
def polygons_overlap(v1, v2):
    """Check if two polygons overlap (edges intersect or one contains the other)."""
    n1 = v1.shape[0]
    n2 = v2.shape[0]
    for i in range(n1):
        i_next = (i + 1) % n1
        for j in range(n2):
            j_next = (j + 1) % n2
            if segments_intersect(
                v1[i, 0], v1[i, 1], v1[i_next, 0], v1[i_next, 1],
                v2[j, 0], v2[j, 1], v2[j_next, 0], v2[j_next, 1]
            ):
                return True
    if point_in_polygon(v1[0, 0], v1[0, 1], v2):
        return True
    if point_in_polygon(v2[0, 0], v2[0, 1], v1):
        return True
    return False

@njit
def trees_overlap(x1, y1, deg1, x2, y2, deg2):
    """Check if two trees overlap, with fast center distance check."""
    dx = x1 - x2
    dy = y1 - y2
    if dx * dx + dy * dy > MAX_OVERLAP_DIST_SQ:
        return False
    v1 = get_tree_vertices(x1, y1, deg1)
    v2 = get_tree_vertices(x2, y2, deg2)
    return polygons_overlap(v1, v2)

@njit
def any_overlap(xs, ys, degs, n):
    """Check if any trees in configuration overlap."""
    for i in range(n):
        for j in range(i + 1, n):
            if trees_overlap(xs[i], ys[i], degs[i], xs[j], ys[j], degs[j]):
                return True
    return False

@njit
def calculate_score_from_arrays(xs, ys, degs, n):
    """Calculate bounding box side length for n trees."""
    min_x = np.inf
    min_y = np.inf
    max_x = -np.inf
    max_y = -np.inf
    for i in range(n):
        vertices = get_tree_vertices(xs[i], ys[i], degs[i])
        for j in range(15):
            if vertices[j, 0] < min_x:
                min_x = vertices[j, 0]
            if vertices[j, 0] > max_x:
                max_x = vertices[j, 0]
            if vertices[j, 1] < min_y:
                min_y = vertices[j, 1]
            if vertices[j, 1] > max_y:
                max_y = vertices[j, 1]
    return max(max_x - min_x, max_y - min_y)

@njit
def sa_optimize_improved(xs, ys, degs, a, b, ncols, nrows, append_x, append_y,
                         Tmax, Tmin, nsteps, nsteps_per_T,
                         position_delta, angle_delta, angle_delta2, delta_t, seed):
    """Simulated annealing with translations and rotations."""
    np.random.seed(seed)
    n = len(xs)
    
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_degs = degs.copy()
    best_score = calculate_score_from_arrays(xs, ys, degs, n)
    
    curr_xs = xs.copy()
    curr_ys = ys.copy()
    curr_degs = degs.copy()
    curr_score = best_score
    
    T = Tmax
    T_decay = (Tmin / Tmax) ** (1.0 / nsteps)
    
    for step in range(nsteps):
        for _ in range(nsteps_per_T):
            # Choose move type
            move_type = np.random.randint(0, 5)
            
            if move_type == 0:
                # Move single tree
                idx = np.random.randint(0, n)
                old_x, old_y = curr_xs[idx], curr_ys[idx]
                curr_xs[idx] += np.random.uniform(-position_delta, position_delta)
                curr_ys[idx] += np.random.uniform(-position_delta, position_delta)
                
                if any_overlap(curr_xs, curr_ys, curr_degs, n):
                    curr_xs[idx], curr_ys[idx] = old_x, old_y
                    continue
                    
            elif move_type == 1:
                # Rotate single tree
                idx = np.random.randint(0, n)
                old_deg = curr_degs[idx]
                curr_degs[idx] += np.random.uniform(-angle_delta, angle_delta)
                
                if any_overlap(curr_xs, curr_ys, curr_degs, n):
                    curr_degs[idx] = old_deg
                    continue
                    
            elif move_type == 2:
                # Translate all trees
                dx = np.random.uniform(-delta_t, delta_t)
                dy = np.random.uniform(-delta_t, delta_t)
                for i in range(n):
                    curr_xs[i] += dx
                    curr_ys[i] += dy
                    
            elif move_type == 3:
                # Rotate all trees by same angle
                d_angle = np.random.uniform(-angle_delta2, angle_delta2)
                for i in range(n):
                    curr_degs[i] += d_angle
                    
            else:
                # Adjust grid spacing
                da = np.random.uniform(-delta_t, delta_t)
                db = np.random.uniform(-delta_t, delta_t)
                new_a = a + da
                new_b = b + db
                if new_a > 0.5 and new_b > 0.5:
                    # Regenerate positions with new spacing
                    old_xs = curr_xs.copy()
                    old_ys = curr_ys.copy()
                    idx = 0
                    for row in range(nrows):
                        for col in range(ncols):
                            curr_xs[idx] = col * new_a
                            curr_ys[idx] = row * new_b
                            idx += 1
                            curr_xs[idx] = col * new_a + new_a / 2
                            curr_ys[idx] = row * new_b + new_b / 2
                            idx += 1
                    if append_x:
                        for row in range(nrows):
                            curr_xs[idx] = ncols * new_a
                            curr_ys[idx] = row * new_b + new_b / 2
                            idx += 1
                    if append_y:
                        for col in range(ncols):
                            curr_xs[idx] = col * new_a + new_a / 2
                            curr_ys[idx] = nrows * new_b
                            idx += 1
                    
                    if any_overlap(curr_xs, curr_ys, curr_degs, n):
                        curr_xs = old_xs
                        curr_ys = old_ys
                        continue
                    a, b = new_a, new_b
            
            new_score = calculate_score_from_arrays(curr_xs, curr_ys, curr_degs, n)
            delta = new_score - curr_score
            
            if delta < 0 or np.random.random() < math.exp(-delta / T):
                curr_score = new_score
                if curr_score < best_score:
                    best_score = curr_score
                    best_xs = curr_xs.copy()
                    best_ys = curr_ys.copy()
                    best_degs = curr_degs.copy()
            else:
                # Reject move - restore
                if move_type == 2:
                    for i in range(n):
                        curr_xs[i] -= dx
                        curr_ys[i] -= dy
                elif move_type == 3:
                    for i in range(n):
                        curr_degs[i] -= d_angle
        
        T *= T_decay
    
    return best_xs, best_ys, best_degs, best_score

def generate_grid_config(ncols, nrows, append_x, append_y, a, b, initial_seeds):
    """Generate initial grid configuration."""
    n_base = 2 * ncols * nrows
    n_append_x = nrows if append_x else 0
    n_append_y = ncols if append_y else 0
    n_trees = n_base + n_append_x + n_append_y
    
    xs = np.zeros(n_trees, dtype=np.float64)
    ys = np.zeros(n_trees, dtype=np.float64)
    degs = np.zeros(n_trees, dtype=np.float64)
    
    idx = 0
    for row in range(nrows):
        for col in range(ncols):
            xs[idx] = col * a
            ys[idx] = row * b
            degs[idx] = initial_seeds[0]
            idx += 1
            xs[idx] = col * a + a / 2
            ys[idx] = row * b + b / 2
            degs[idx] = initial_seeds[1]
            idx += 1
    
    if append_x:
        for row in range(nrows):
            xs[idx] = ncols * a
            ys[idx] = row * b + b / 2
            degs[idx] = initial_seeds[1]
            idx += 1
    
    if append_y:
        for col in range(ncols):
            xs[idx] = col * a + a / 2
            ys[idx] = nrows * b
            degs[idx] = initial_seeds[1]
            idx += 1
    
    return xs, ys, degs, n_trees

def optimize_grid_config(args):
    """Optimize a single grid configuration."""
    ncols, nrows, append_x, append_y, initial_seeds, a_init, b_init, sa_params, seed = args
    
    xs, ys, degs, n_trees = generate_grid_config(ncols, nrows, append_x, append_y, a_init, b_init, initial_seeds)
    
    # Check for initial overlaps
    if any_overlap(xs, ys, degs, n_trees):
        return n_trees, float('inf'), None
    
    best_xs, best_ys, best_degs, best_score = sa_optimize_improved(
        xs, ys, degs, a_init, b_init, ncols, nrows, append_x, append_y,
        sa_params['Tmax'], sa_params['Tmin'], sa_params['nsteps'], sa_params['nsteps_per_T'],
        sa_params['position_delta'], sa_params['angle_delta'], sa_params['angle_delta2'],
        sa_params['delta_t'], seed
    )
    
    tree_data = [(best_xs[i], best_ys[i], best_degs[i]) for i in range(n_trees)]
    return n_trees, best_score, tree_data

def strip(val):
    return float(str(val).replace('s', ''))

# Load current best submission
df = pd.read_csv('/home/submission/submission.csv')
print(f'Loaded {len(df)} rows')

# Parse into flat arrays
total_trees = 200 * 201 // 2
baseline_xs = np.zeros(total_trees, dtype=np.float64)
baseline_ys = np.zeros(total_trees, dtype=np.float64)
baseline_degs = np.zeros(total_trees, dtype=np.float64)

for n in range(1, 201):
    group = df[df['id'].str.startswith(f'{n:03d}_')]
    idx = n * (n - 1) // 2
    for i, (_, row) in enumerate(group.iterrows()):
        baseline_xs[idx + i] = strip(row['x'])
        baseline_ys[idx + i] = strip(row['y'])
        baseline_degs[idx + i] = strip(row['deg'])

# Calculate baseline score
baseline_total = 0.0
for n in range(1, 201):
    idx = n * (n - 1) // 2
    side = calculate_score_from_arrays(baseline_xs[idx:idx+n], baseline_ys[idx:idx+n], baseline_degs[idx:idx+n], n)
    baseline_total += side * side / n

print(f'Baseline score: {baseline_total:.6f}')

# Grid configuration parameters
initial_seeds = [0.0, 180.0]  # Alternating orientations
a_init = 0.75  # Initial horizontal spacing
b_init = 0.75  # Initial vertical spacing

# Generate all viable grid configurations
grid_configs = []
for ncols in range(1, 15):
    for nrows in range(1, 15):
        for append_x in [False, True]:
            for append_y in [False, True]:
                n_base = 2 * ncols * nrows
                n_append_x = nrows if append_x else 0
                n_append_y = ncols if append_y else 0
                n_trees = n_base + n_append_x + n_append_y
                if 2 <= n_trees <= 200:
                    grid_configs.append((ncols, nrows, append_x, append_y))

# Sort by tree count
grid_configs.sort(key=lambda x: (2 * x[0] * x[1] + (x[1] if x[2] else 0) + (x[0] if x[3] else 0)))
print(f"Generated {len(grid_configs)} grid configurations")

# SA parameters
sa_params = {
    "Tmax": 0.001,
    "Tmin": 0.000001,
    "nsteps": 10,
    "nsteps_per_T": 10000,
    "position_delta": 0.002,
    "angle_delta": 1.0,
    "angle_delta2": 1.0,
    "delta_t": 0.002,
}

# Warm up numba
print("Compiling numba functions...")
t0 = time.time()
dummy_xs = np.array([0.0, 1.0], dtype=np.float64)
dummy_ys = np.array([0.0, 0.0], dtype=np.float64)
dummy_degs = np.array([0.0, 180.0], dtype=np.float64)
_ = sa_optimize_improved(
    dummy_xs, dummy_ys, dummy_degs,
    1.0, 1.0, 1, 1, False, False,
    0.001, 0.0001, 2, 10,
    0.01, 10.0, 10.0, 0.01, 42
)
print(f"Compilation done in {time.time() - t0:.1f}s")

# Prepare tasks
tasks = []
tree_counts = []
for i, (ncols, nrows, append_x, append_y) in enumerate(grid_configs):
    n_base = 2 * ncols * nrows
    n_append_x = nrows if append_x else 0
    n_append_y = ncols if append_y else 0
    n_trees = n_base + n_append_x + n_append_y
    
    if n_trees > 200:
        continue
    
    seed = 42 + i * 1000
    tasks.append((ncols, nrows, append_x, append_y, initial_seeds, a_init, b_init, sa_params, seed))
    tree_counts.append(n_trees)

print(f"Tasks: {len(tasks)}, tree counts: {min(tree_counts)} to {max(tree_counts)}")

# Run SA optimization in parallel
print(f"Running SA optimization on {len(tasks)} configurations...")
num_workers = min(cpu_count(), len(tasks))
print(f"Using {num_workers} workers")

t0 = time.time()
with Pool(num_workers) as pool:
    results = pool.map(optimize_grid_config, tasks)
elapsed = time.time() - t0
print(f"SA optimization completed in {elapsed:.1f}s")

# Collect results and compare with baseline
new_trees = {}
improved_count = 0
for n_trees, score, tree_data in results:
    if tree_data is None:
        continue
    
    # Get baseline score for this n
    idx = n_trees * (n_trees - 1) // 2
    baseline_score = calculate_score_from_arrays(
        baseline_xs[idx:idx+n_trees], 
        baseline_ys[idx:idx+n_trees], 
        baseline_degs[idx:idx+n_trees], 
        n_trees
    )
    
    if score < baseline_score:
        new_trees[n_trees] = tree_data
        improvement = baseline_score - score
        if improvement > 1e-6:
            improved_count += 1
            print(f"  n={n_trees}: {score:.6f} (baseline: {baseline_score:.6f}, improved by {improvement:.6f})")

print(f"SA improved {improved_count} configurations")

# Merge with baseline
print("Merging with baseline...")
merged_xs = baseline_xs.copy()
merged_ys = baseline_ys.copy()
merged_degs = baseline_degs.copy()

for n_trees, tree_data in new_trees.items():
    idx = n_trees * (n_trees - 1) // 2
    for i in range(n_trees):
        merged_xs[idx + i] = tree_data[i][0]
        merged_ys[idx + i] = tree_data[i][1]
        merged_degs[idx + i] = tree_data[i][2]

# Calculate merged score
merged_score = 0.0
for n in range(1, 201):
    idx = n * (n - 1) // 2
    side = calculate_score_from_arrays(merged_xs[idx:idx+n], merged_ys[idx:idx+n], merged_degs[idx:idx+n], n)
    merged_score += side * side / n

print(f"Score after SA merge: {merged_score:.6f}")
print(f"Improvement: {baseline_total - merged_score:.6f}")

# Save if improved
if merged_score < baseline_total - 1e-9:
    # Build new submission preserving precision for unchanged configs
    rows = []
    for n in range(1, 201):
        if n in new_trees:
            # Use new configuration
            for i in range(n):
                rows.append({
                    'id': f'{n:03d}_{i}',
                    'x': f's{new_trees[n][i][0]}',
                    'y': f's{new_trees[n][i][1]}',
                    'deg': f's{new_trees[n][i][2]}'
                })
        else:
            # Keep original configuration (preserve precision)
            group = df[df['id'].str.startswith(f'{n:03d}_')]
            for _, row in group.iterrows():
                rows.append({
                    'id': row['id'],
                    'x': row['x'],
                    'y': row['y'],
                    'deg': row['deg']
                })
    
    out_df = pd.DataFrame(rows)
    out_df.to_csv('/home/submission/submission.csv', index=False)
    print(f'Saved improved submission')
else:
    print('No improvement - keeping original submission')
