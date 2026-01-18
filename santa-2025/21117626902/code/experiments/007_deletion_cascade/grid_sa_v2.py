"""
Grid-based SA optimization from jiweiliu kernel - v2.
Uses pre-optimized 2-tree seed configuration.
"""
import math
import os
import time
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List as NumbaList

print("Grid-based SA optimization from jiweiliu kernel - v2")

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
def create_grid_extended(seed_xs, seed_ys, seed_degs, a, b, ncols, nrows, append_x, append_y):
    """Create grid of trees by translation with optional append."""
    n_seeds = len(seed_xs)
    n_base = n_seeds * ncols * nrows
    n_append_x = nrows if append_x else 0
    n_append_y = ncols if append_y else 0
    n_total = n_base + n_append_x + n_append_y
    
    xs = np.empty(n_total, dtype=np.float64)
    ys = np.empty(n_total, dtype=np.float64)
    degs = np.empty(n_total, dtype=np.float64)
    
    idx = 0
    # Base grid
    for s in range(n_seeds):
        for col in range(ncols):
            for row in range(nrows):
                xs[idx] = seed_xs[s] + col * a
                ys[idx] = seed_ys[s] + row * b
                degs[idx] = seed_degs[s]
                idx += 1
    
    # Append in x direction
    if append_x and n_seeds > 1:
        for row in range(nrows):
            xs[idx] = seed_xs[1] + ncols * a
            ys[idx] = seed_ys[1] + row * b
            degs[idx] = seed_degs[1]
            idx += 1
    
    # Append in y direction
    if append_y and n_seeds > 1:
        for col in range(ncols):
            xs[idx] = seed_xs[1] + col * a
            ys[idx] = seed_ys[1] + nrows * b
            degs[idx] = seed_degs[1]
            idx += 1
    
    return xs, ys, degs, n_total

@njit
def has_any_overlap(xs, ys, degs, n):
    """Check if any trees overlap."""
    for i in range(n):
        for j in range(i + 1, n):
            if trees_overlap(xs[i], ys[i], degs[i], xs[j], ys[j], degs[j]):
                return True
    return False

@njit
def sa_optimize_grid(seed_xs, seed_ys, seed_degs, a_init, b_init, ncols, nrows, append_x, append_y,
                     Tmax, Tmin, nsteps, nsteps_per_T, position_delta, angle_delta, angle_delta2, delta_t, seed):
    """Simulated annealing for grid configuration."""
    np.random.seed(seed)
    n_seeds = len(seed_xs)
    
    # Copy initial seeds
    curr_xs = seed_xs.copy()
    curr_ys = seed_ys.copy()
    curr_degs = seed_degs.copy()
    a = a_init
    b = b_init
    
    # Create initial grid
    grid_xs, grid_ys, grid_degs, n_total = create_grid_extended(curr_xs, curr_ys, curr_degs, a, b, ncols, nrows, append_x, append_y)
    
    # Check for initial overlaps and adjust spacing if needed
    if has_any_overlap(grid_xs, grid_ys, grid_degs, n_total):
        # Increase spacing until no overlap
        for mult in [1.5, 2.0, 2.5, 3.0]:
            a_test = a_init * mult
            b_test = b_init * mult
            grid_xs, grid_ys, grid_degs, n_total = create_grid_extended(curr_xs, curr_ys, curr_degs, a_test, b_test, ncols, nrows, append_x, append_y)
            if not has_any_overlap(grid_xs, grid_ys, grid_degs, n_total):
                a, b = a_test, b_test
                break
        else:
            # Still overlapping, return infinity
            return np.inf, curr_xs, curr_ys, curr_degs, a, b
    
    curr_score = calculate_score_from_arrays(grid_xs, grid_ys, grid_degs, n_total)
    
    best_score = curr_score
    best_xs = curr_xs.copy()
    best_ys = curr_ys.copy()
    best_degs = curr_degs.copy()
    best_a = a
    best_b = b
    
    T = Tmax
    T_decay = (Tmin / Tmax) ** (1.0 / nsteps)
    
    n_move_types = n_seeds + 2
    
    for step in range(nsteps):
        for _ in range(nsteps_per_T):
            move_type = np.random.randint(0, n_move_types)
            
            if move_type < n_seeds:
                # Move single seed
                i = move_type
                old_x, old_y, old_deg = curr_xs[i], curr_ys[i], curr_degs[i]
                curr_xs[i] += np.random.uniform(-position_delta, position_delta)
                curr_ys[i] += np.random.uniform(-position_delta, position_delta)
                curr_degs[i] = (curr_degs[i] + np.random.uniform(-angle_delta, angle_delta)) % 360.0
                
            elif move_type == n_seeds:
                # Adjust translation lengths
                old_a, old_b = a, b
                a = a * (1.0 + np.random.uniform(-delta_t, delta_t))
                b = b * (1.0 + np.random.uniform(-delta_t, delta_t))
                
            else:
                # Rotate all seeds by same angle
                old_degs = curr_degs.copy()
                ddeg = np.random.uniform(-angle_delta2, angle_delta2)
                for i in range(n_seeds):
                    curr_degs[i] = (curr_degs[i] + ddeg) % 360.0
            
            # Create new grid and check
            grid_xs, grid_ys, grid_degs, n_total = create_grid_extended(curr_xs, curr_ys, curr_degs, a, b, ncols, nrows, append_x, append_y)
            
            if has_any_overlap(grid_xs, grid_ys, grid_degs, n_total):
                # Reject move
                if move_type < n_seeds:
                    curr_xs[move_type], curr_ys[move_type], curr_degs[move_type] = old_x, old_y, old_deg
                elif move_type == n_seeds:
                    a, b = old_a, old_b
                else:
                    curr_degs = old_degs
                continue
            
            new_score = calculate_score_from_arrays(grid_xs, grid_ys, grid_degs, n_total)
            delta = new_score - curr_score
            
            if delta < 0 or np.random.random() < math.exp(-delta / T):
                curr_score = new_score
                if curr_score < best_score:
                    best_score = curr_score
                    best_xs = curr_xs.copy()
                    best_ys = curr_ys.copy()
                    best_degs = curr_degs.copy()
                    best_a = a
                    best_b = b
            else:
                # Reject move
                if move_type < n_seeds:
                    curr_xs[move_type], curr_ys[move_type], curr_degs[move_type] = old_x, old_y, old_deg
                elif move_type == n_seeds:
                    a, b = old_a, old_b
                else:
                    curr_degs = old_degs
        
        T *= T_decay
    
    return best_score, best_xs, best_ys, best_degs, best_a, best_b

def optimize_grid_config(args):
    """Optimize a single grid configuration."""
    ncols, nrows, append_x, append_y, seed_xs, seed_ys, seed_degs, a_init, b_init, sa_params, seed = args
    
    best_score, best_xs, best_ys, best_degs, best_a, best_b = sa_optimize_grid(
        seed_xs, seed_ys, seed_degs, a_init, b_init, ncols, nrows, append_x, append_y,
        sa_params['Tmax'], sa_params['Tmin'], sa_params['nsteps'], sa_params['nsteps_per_T'],
        sa_params['position_delta'], sa_params['angle_delta'], sa_params['angle_delta2'],
        sa_params['delta_t'], seed
    )
    
    if best_score == np.inf:
        return None, np.inf, None
    
    # Get final grid positions
    grid_xs, grid_ys, grid_degs, n_total = create_grid_extended(best_xs, best_ys, best_degs, best_a, best_b, ncols, nrows, append_x, append_y)
    
    tree_data = [(grid_xs[i], grid_ys[i], grid_degs[i]) for i in range(n_total)]
    return n_total, best_score, tree_data

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

# Pre-optimized 2-tree seed configuration from jiweiliu kernel
# These are relative positions that form a valid 2-tree unit cell
seed_xs = np.array([-4.191683864412409, -4.92202045352307], dtype=np.float64)
seed_ys = np.array([-4.498489528496051, -4.727639556649786], dtype=np.float64)
seed_degs = np.array([74.54421568660419, 254.5401905706735], dtype=np.float64)

# Initial translation lengths
a_init = 0.8744896974945239
b_init = 0.7499641699190263

# Generate all viable grid configurations
grid_configs = []
for ncols in range(1, 15):
    for nrows in range(1, 15):
        for append_x in [False, True]:
            for append_y in [False, True]:
                n_seeds = 2
                n_base = n_seeds * ncols * nrows
                n_append_x = nrows if append_x else 0
                n_append_y = ncols if append_y else 0
                n_trees = n_base + n_append_x + n_append_y
                if 2 <= n_trees <= 200:
                    grid_configs.append((ncols, nrows, append_x, append_y, n_trees))

# Sort by tree count
grid_configs.sort(key=lambda x: x[4])
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
_ = sa_optimize_grid(
    dummy_xs, dummy_ys, dummy_degs,
    1.0, 1.0, 1, 1, False, False,
    0.001, 0.0001, 2, 10,
    0.01, 10.0, 10.0, 0.01, 42
)
print(f"Compilation done in {time.time() - t0:.1f}s")

# Prepare tasks
tasks = []
for i, (ncols, nrows, append_x, append_y, n_trees) in enumerate(grid_configs):
    seed = 42 + i * 1000
    tasks.append((ncols, nrows, append_x, append_y, seed_xs, seed_ys, seed_degs, a_init, b_init, sa_params, seed))

print(f"Tasks: {len(tasks)}")

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
valid_count = 0
for n_trees, score, tree_data in results:
    if tree_data is None:
        continue
    valid_count += 1
    
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

print(f"Valid configurations: {valid_count}")
print(f"SA improved {improved_count} configurations")

# Merge with baseline
if improved_count > 0:
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
else:
    print('No improvements found - keeping original submission')
