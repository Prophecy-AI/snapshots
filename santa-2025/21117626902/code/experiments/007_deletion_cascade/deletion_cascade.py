import numpy as np
import pandas as pd
from numba import njit
import math
import time

print("Implementing jiweiliu deletion cascade technique...")

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
        [0.0, 0.8],  # TIP_Y
        [0.125, 0.5],  # TOP_W/2, TIER_1_Y
        [0.0625, 0.5],  # TOP_W/4, TIER_1_Y
        [0.2, 0.25],  # MID_W/2, TIER_2_Y
        [0.1, 0.25],  # MID_W/4, TIER_2_Y
        [0.35, 0.0],  # BASE_W/2, BASE_Y
        [0.075, 0.0],  # TRUNK_W/2, BASE_Y
        [0.075, -0.2],  # TRUNK_W/2, TRUNK_BOTTOM_Y
        [-0.075, -0.2],  # -TRUNK_W/2, TRUNK_BOTTOM_Y
        [-0.075, 0.0],  # -TRUNK_W/2, BASE_Y
        [-0.35, 0.0],  # -BASE_W/2, BASE_Y
        [-0.1, 0.25],  # -MID_W/4, TIER_2_Y
        [-0.2, 0.25],  # -MID_W/2, TIER_2_Y
        [-0.0625, 0.5],  # -TOP_W/4, TIER_1_Y
        [-0.125, 0.5],  # -TOP_W/2, TIER_1_Y
    ], dtype=np.float64)
    for i in range(15):
        rx, ry = rotate_point(pts[i, 0], pts[i, 1], cos_a, sin_a)
        vertices[i, 0] = rx + cx
        vertices[i, 1] = ry + cy
    return vertices

@njit
def calculate_bounding_box(xs, ys, degs, n):
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
def deletion_cascade_numba(xs, ys, degs):
    """
    For each N from 200 down to 2, try removing each tree and keep the deletion
    that minimizes the bounding box. This propagates good large configs to smaller sizes.
    """
    max_n = 200
    total_trees = max_n * (max_n + 1) // 2
    
    # Output arrays
    out_xs = np.zeros(total_trees, dtype=np.float64)
    out_ys = np.zeros(total_trees, dtype=np.float64)
    out_degs = np.zeros(total_trees, dtype=np.float64)
    side_lengths = np.zeros(max_n, dtype=np.float64)
    
    # Copy input to output
    for i in range(total_trees):
        out_xs[i] = xs[i]
        out_ys[i] = ys[i]
        out_degs[i] = degs[i]
    
    # Calculate initial side lengths
    for n in range(1, max_n + 1):
        idx = n * (n - 1) // 2
        side_lengths[n - 1] = calculate_bounding_box(out_xs[idx:idx+n], out_ys[idx:idx+n], out_degs[idx:idx+n], n)
    
    # Working arrays for current configuration
    work_xs = np.zeros(max_n, dtype=np.float64)
    work_ys = np.zeros(max_n, dtype=np.float64)
    work_degs = np.zeros(max_n, dtype=np.float64)
    
    # Process from largest to smallest
    for n in range(max_n, 1, -1):
        # Get current n configuration
        idx = n * (n - 1) // 2
        for i in range(n):
            work_xs[i] = out_xs[idx + i]
            work_ys[i] = out_ys[idx + i]
            work_degs[i] = out_degs[idx + i]
        
        # Try removing each tree and find the one that minimizes bounding box
        best_side = np.inf
        best_remove_idx = -1
        
        for remove_idx in range(n):
            # Calculate bounding box without this tree
            temp_xs = np.zeros(n - 1, dtype=np.float64)
            temp_ys = np.zeros(n - 1, dtype=np.float64)
            temp_degs = np.zeros(n - 1, dtype=np.float64)
            
            j = 0
            for i in range(n):
                if i != remove_idx:
                    temp_xs[j] = work_xs[i]
                    temp_ys[j] = work_ys[i]
                    temp_degs[j] = work_degs[i]
                    j += 1
            
            side = calculate_bounding_box(temp_xs, temp_ys, temp_degs, n - 1)
            if side < best_side:
                best_side = side
                best_remove_idx = remove_idx
        
        # Check if this is better than current best for n-1
        target_n = n - 1
        if best_side < side_lengths[target_n - 1]:
            # Update the n-1 configuration
            idx_target = target_n * (target_n - 1) // 2
            j = 0
            for i in range(n):
                if i != best_remove_idx:
                    out_xs[idx_target + j] = work_xs[i]
                    out_ys[idx_target + j] = work_ys[i]
                    out_degs[idx_target + j] = work_degs[i]
                    j += 1
            side_lengths[target_n - 1] = best_side
    
    return out_xs, out_ys, out_degs, side_lengths

def strip(val):
    return float(str(val).replace('s', ''))

# Load current best submission
df = pd.read_csv('/home/submission/submission.csv')
print(f'Loaded {len(df)} rows')

# Parse into flat arrays
total_trees = 200 * 201 // 2
xs = np.zeros(total_trees, dtype=np.float64)
ys = np.zeros(total_trees, dtype=np.float64)
degs = np.zeros(total_trees, dtype=np.float64)

for n in range(1, 201):
    group = df[df['id'].str.startswith(f'{n:03d}_')]
    idx = n * (n - 1) // 2
    for i, (_, row) in enumerate(group.iterrows()):
        xs[idx + i] = strip(row['x'])
        ys[idx + i] = strip(row['y'])
        degs[idx + i] = strip(row['deg'])

# Calculate initial score
initial_score = 0.0
for n in range(1, 201):
    idx = n * (n - 1) // 2
    side = calculate_bounding_box(xs[idx:idx+n], ys[idx:idx+n], degs[idx:idx+n], n)
    initial_score += side * side / n

print(f'Initial score: {initial_score:.6f}')

# Run deletion cascade
print('Running deletion cascade...')
t0 = time.time()
out_xs, out_ys, out_degs, side_lengths = deletion_cascade_numba(xs, ys, degs)
print(f'Cascade completed in {time.time() - t0:.1f}s')

# Calculate final score
final_score = 0.0
for n in range(1, 201):
    final_score += side_lengths[n - 1] ** 2 / n

print(f'Final score: {final_score:.6f}')
print(f'Improvement: {initial_score - final_score:.6f}')

# Check for improvements
improvements = 0
improved_ns = []
for n in range(1, 201):
    idx = n * (n - 1) // 2
    old_side = calculate_bounding_box(xs[idx:idx+n], ys[idx:idx+n], degs[idx:idx+n], n)
    new_side = side_lengths[n - 1]
    if new_side < old_side - 1e-9:
        improvements += 1
        improved_ns.append(n)
        if improvements <= 10:
            print(f'  N={n}: {old_side:.6f} -> {new_side:.6f}')

print(f'Total improvements: {improvements}')
if len(improved_ns) > 10:
    print(f'  ... and {len(improved_ns) - 10} more')

# Save if improved
if final_score < initial_score - 1e-9:
    # Build new submission preserving precision for unchanged configs
    rows = []
    for n in range(1, 201):
        idx = n * (n - 1) // 2
        old_side = calculate_bounding_box(xs[idx:idx+n], ys[idx:idx+n], degs[idx:idx+n], n)
        new_side = side_lengths[n - 1]
        
        if new_side < old_side - 1e-9:
            # Use new configuration
            for i in range(n):
                rows.append({
                    'id': f'{n:03d}_{i}',
                    'x': f's{out_xs[idx + i]}',
                    'y': f's{out_ys[idx + i]}',
                    'deg': f's{out_degs[idx + i]}'
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
