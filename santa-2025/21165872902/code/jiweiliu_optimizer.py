import math
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List as NumbaList

# Tree shape constants (must match official spec)
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

# Maximum distance between tree centers for possible overlap
# Tree spans from y=-0.2 to y=0.8 (height=1.0) and x=-0.35 to x=0.35 (width=0.7)
# Max radius when rotated: sqrt(0.5^2 + 0.8^2) ≈ 0.94
# Two trees can overlap only if centers < 2 * 0.94 ≈ 1.88
# Use 1.8 as conservative threshold
MAX_OVERLAP_DIST = 1.8
MAX_OVERLAP_DIST_SQ = MAX_OVERLAP_DIST * MAX_OVERLAP_DIST

@njit(cache=True)
def rotate_point(x, y, cos_a, sin_a):
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a


@njit(cache=True)
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

@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
def segments_intersect(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
    """
    Check if two line segments intersect using cross-product method.

    Segment A: (p1x, p1y) to (p2x, p2y)
    Segment B: (p3x, p3y) to (p4x, p4y)

    Two segments intersect iff:
    - Endpoints of A are on opposite sides of line B
    - Endpoints of B are on opposite sides of line A

    This is checked using the sign of cross products (avoids divisions).
    """
    # Direction vectors
    dax = p2x - p1x
    day = p2y - p1y
    dbx = p4x - p3x
    dby = p4y - p3y

    # Vectors from B's start to A's endpoints
    d1x = p1x - p3x
    d1y = p1y - p3y
    d2x = p2x - p3x
    d2y = p2y - p3y

    # Cross products: check if A's endpoints are on opposite sides of line B
    cross_b1 = dbx * d1y - dby * d1x
    cross_b2 = dbx * d2y - dby * d2x

    # If both have same sign, A's endpoints are on same side of line B
    if cross_b1 * cross_b2 > 0:
        return False

    # Vectors from A's start to B's endpoints
    d3x = p3x - p1x
    d3y = p3y - p1y
    d4x = p4x - p1x
    d4y = p4y - p1y

    # Cross products: check if B's endpoints are on opposite sides of line A
    cross_a1 = dax * d3y - day * d3x
    cross_a2 = dax * d4y - day * d4x

    if cross_a1 * cross_a2 > 0:
        return False

    return True

@njit(cache=True)
def polygons_overlap(verts1, verts2, cx1, cy1, cx2, cy2):
    """Check if two polygons overlap (not just touch)."""
    # Quick center distance check
    dx = cx2 - cx1
    dy = cy2 - cy1
    dist_sq = dx * dx + dy * dy
    if dist_sq > MAX_OVERLAP_DIST_SQ:
        return False

    # Bounding box check
    min_x1, min_y1, max_x1, max_y1 = polygon_bounds(verts1)
    min_x2, min_y2, max_x2, max_y2 = polygon_bounds(verts2)
    if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
        return False

    # Check if any vertex of poly1 is inside poly2
    for i in range(verts1.shape[0]):
        if point_in_polygon(verts1[i, 0], verts1[i, 1], verts2):
            return True

    # Check if any vertex of poly2 is inside poly1
    for i in range(verts2.shape[0]):
        if point_in_polygon(verts2[i, 0], verts2[i, 1], verts1):
            return True

    # Check edge intersections
    n1 = verts1.shape[0]
    n2 = verts2.shape[0]
    for i in range(n1):
        j = (i + 1) % n1
        p1x, p1y = verts1[i, 0], verts1[i, 1]
        p2x, p2y = verts1[j, 0], verts1[j, 1]
        for k in range(n2):
            m = (k + 1) % n2
            p3x, p3y = verts2[k, 0], verts2[k, 1]
            p4x, p4y = verts2[m, 0], verts2[m, 1]
            if segments_intersect(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
                return True
    return False


@njit(cache=True)
def has_any_overlap(all_vertices, centers_x, centers_y):
    """Check if any pair of polygons overlap."""
    n = len(all_vertices)
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap(all_vertices[i], all_vertices[j],
                              centers_x[i], centers_y[i], centers_x[j], centers_y[j]):
                return True
    return False

@njit(cache=True)
def compute_bounding_box(all_vertices):
    """Compute overall bounding box of all polygons."""
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    for verts in all_vertices:
        x1, y1, x2, y2 = polygon_bounds(verts)
        if x1 < min_x:
            min_x = x1
        if y1 < min_y:
            min_y = y1
        if x2 > max_x:
            max_x = x2
        if y2 > max_y:
            max_y = y2
    return min_x, min_y, max_x, max_y


@njit(cache=True)
def get_side_length(all_vertices):
    """Get side length of bounding square."""
    min_x, min_y, max_x, max_y = compute_bounding_box(all_vertices)
    return max(max_x - min_x, max_y - min_y)


@njit(cache=True)
def calculate_score_numba(all_vertices):
    """Calculate score = max(width, height)^2 / n"""
    side = get_side_length(all_vertices)
    return side * side / len(all_vertices)

@njit(cache=True)
def create_grid_vertices_extended(seed_xs, seed_ys, seed_degs, a, b, ncols, nrows, append_x, append_y):
    """
    Create grid of tree vertices by translation with optional append.
    Returns both vertices and center coordinates.

    append_x: if True, add one extra tree (seed index 1) at the end of each row
    append_y: if True, add one extra tree (seed index 1) at the end of each column
    """
    n_seeds = len(seed_xs)

    # Calculate total number of trees
    n_base = n_seeds * ncols * nrows
    n_append_x = nrows if append_x else 0
    n_append_y = ncols if append_y else 0
    n_total = n_base + n_append_x + n_append_y

    all_vertices = []
    centers_x = np.empty(n_total, dtype=np.float64)
    centers_y = np.empty(n_total, dtype=np.float64)

    idx = 0
    # Base grid
    for s in range(n_seeds):
        for col in range(ncols):
            for row in range(nrows):
                cx = seed_xs[s] + col * a
                cy = seed_ys[s] + row * b
                all_vertices.append(get_tree_vertices(cx, cy, seed_degs[s]))
                centers_x[idx] = cx
                centers_y[idx] = cy
                idx += 1

    # Append in x direction
    if append_x and n_seeds > 1:
        for row in range(nrows):
            cx = seed_xs[1] + ncols * a
            cy = seed_ys[1] + row * b
            all_vertices.append(get_tree_vertices(cx, cy, seed_degs[1]))
            centers_x[idx] = cx
            centers_y[idx] = cy
            idx += 1

    # Append in y direction
    if append_y and n_seeds > 1:
        for col in range(ncols):
            cx = seed_xs[1] + col * a
            cy = seed_ys[1] + nrows * b
            all_vertices.append(get_tree_vertices(cx, cy, seed_degs[1]))
            centers_x[idx] = cx
            centers_y[idx] = cy
            idx += 1

    return all_vertices, centers_x, centers_y


@njit(cache=True)
def get_initial_translations(seed_xs, seed_ys, seed_degs):
    """Get initial translation lengths from seed bounding box."""
    seed_vertices = [get_tree_vertices(seed_xs[i], seed_ys[i], seed_degs[i]) for i in range(len(seed_xs))]
    min_x, min_y, max_x, max_y = compute_bounding_box(seed_vertices)
    return max_x - min_x, max_y - min_y

@njit(cache=True)
def sa_optimize_improved(
    seed_xs_init,
    seed_ys_init,
    seed_degs_init,
    a_init,
    b_init,
    ncols,
    nrows,
    append_x,
    append_y,
    Tmax,
    Tmin,
    nsteps,
    nsteps_per_T,
    position_delta,
    angle_delta,
    angle_delta2,
    delta_t,
    random_seed,
):
    """
    Improved simulated annealing with:
    1. Translation lengths optimized via SA
    2. rotate_all move type
    3. append_x/append_y support
    """
    np.random.seed(random_seed)
    n_seeds = len(seed_xs_init)

    # Copy initial seeds
    seed_xs = seed_xs_init.copy()
    seed_ys = seed_ys_init.copy()
    seed_degs = seed_degs_init.copy()

    # Initial translations
    a = a_init
    b = b_init

    # Create initial grid and check validity
    all_vertices, centers_x, centers_y = create_grid_vertices_extended(seed_xs, seed_ys, seed_degs, a, b, ncols, nrows, append_x, append_y)
    if has_any_overlap(all_vertices, centers_x, centers_y):
        # Try to find valid initial translations
        a_test, b_test = get_initial_translations(seed_xs, seed_ys, seed_degs)
        a = max(a, a_test * 1.5)
        b = max(b, b_test * 1.5)
        all_vertices, centers_x, centers_y = create_grid_vertices_extended(seed_xs, seed_ys, seed_degs, a, b, ncols, nrows, append_x, append_y)

    current_score = calculate_score_numba(all_vertices)

    best_score = current_score
    best_xs = seed_xs.copy()
    best_ys = seed_ys.copy()
    best_degs = seed_degs.copy()
    best_a = a
    best_b = b

    T = Tmax
    Tfactor = -math.log(Tmax / Tmin)

    n_move_types = n_seeds + 2

    for step in range(nsteps):
        for _ in range(nsteps_per_T):
            # Choose move type
            move_type = np.random.randint(0, n_move_types)

            if move_type < n_seeds:
                i = move_type
                old_x = seed_xs[i]
                old_y = seed_ys[i]
                old_deg = seed_degs[i]

                dx = (np.random.random() * 2.0 - 1.0) * position_delta
                dy = (np.random.random() * 2.0 - 1.0) * position_delta
                ddeg = (np.random.random() * 2.0 - 1.0) * angle_delta

                seed_xs[i] = old_x + dx
                seed_ys[i] = old_y + dy
                seed_degs[i] = (old_deg + ddeg) % 360.0

            elif move_type == n_seeds:
                old_a = a
                old_b = b
                da = (np.random.random() * 2.0 - 1.0) * delta_t
                db = (np.random.random() * 2.0 - 1.0) * delta_t
                a = old_a + old_a * da
                b = old_b + old_b * db

            else:
                # Rotate all trees by same angle
                old_degs = seed_degs.copy()
                ddeg = (np.random.random() * 2.0 - 1.0) * angle_delta2
                for i in range(n_seeds):
                    seed_degs[i] = (seed_degs[i] + ddeg) % 360.0

            # Check for collisions in a small test grid (2x2)
            test_vertices, test_cx, test_cy = create_grid_vertices_extended(seed_xs, seed_ys, seed_degs, a, b, 2, 2, False, False)
            if has_any_overlap(test_vertices, test_cx, test_cy):
                # Revert
                if move_type < n_seeds:
                    seed_xs[move_type] = old_x
                    seed_ys[move_type] = old_y
                    seed_degs[move_type] = old_deg
                elif move_type == n_seeds:
                    a = old_a
                    b = old_b
                else:
                    for i in range(n_seeds):
                        seed_degs[i] = old_degs[i]
                continue

            # Create full grid and calculate score
            new_vertices, new_cx, new_cy = create_grid_vertices_extended(seed_xs, seed_ys, seed_degs, a, b, ncols, nrows, append_x, append_y)

            # Additional overlap check for full grid
            if has_any_overlap(new_vertices, new_cx, new_cy):
                if move_type < n_seeds:
                    seed_xs[move_type] = old_x
                    seed_ys[move_type] = old_y
                    seed_degs[move_type] = old_deg
                elif move_type == n_seeds:
                    a = old_a
                    b = old_b
                else:
                    for i in range(n_seeds):
                        seed_degs[i] = old_degs[i]
                continue

            new_score = calculate_score_numba(new_vertices)
            delta = new_score - current_score

            # Metropolis criterion
            accept = False
            if delta < 0:
                accept = True
            elif T > 1e-10:
                if np.random.random() < math.exp(-delta / T):
                    accept = True

            if accept:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_xs = seed_xs.copy()
                    best_ys = seed_ys.copy()
                    best_degs = seed_degs.copy()
                    best_a = a
                    best_b = b
            else:
                # Revert
                if move_type < n_seeds:
                    seed_xs[move_type] = old_x
                    seed_ys[move_type] = old_y
                    seed_degs[move_type] = old_deg
                elif move_type == n_seeds:
                    a = old_a
                    b = old_b
                else:
                    for i in range(n_seeds):
                        seed_degs[i] = old_degs[i]

        # Exponential cooling
        T = Tmax * math.exp(Tfactor * (step + 1) / nsteps)

    return best_score, best_xs, best_ys, best_degs, best_a, best_b

@njit(cache=True)
def get_final_grid_positions_extended(seed_xs, seed_ys, seed_degs, a, b, ncols, nrows, append_x, append_y):
    """Get final tree positions for the optimized grid with append support."""
    n_seeds = len(seed_xs)

    # Calculate total trees
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

    # Append x
    if append_x and n_seeds > 1:
        for row in range(nrows):
            xs[idx] = seed_xs[1] + ncols * a
            ys[idx] = seed_ys[1] + row * b
            degs[idx] = seed_degs[1]
            idx += 1

    # Append y
    if append_y and n_seeds > 1:
        for col in range(ncols):
            xs[idx] = seed_xs[1] + col * a
            ys[idx] = seed_ys[1] + nrows * b
            degs[idx] = seed_degs[1]
            idx += 1

    return xs, ys, degs

def optimize_grid_config(args):
    """Optimize a single grid configuration (worker function for multiprocessing)."""
    ncols, nrows, append_x, append_y, initial_seeds, a_init, b_init, params, seed = args

    seed_xs = np.array([s[0] for s in initial_seeds], dtype=np.float64)
    seed_ys = np.array([s[1] for s in initial_seeds], dtype=np.float64)
    seed_degs = np.array([s[2] for s in initial_seeds], dtype=np.float64)

    n_seeds = len(initial_seeds)
    n_base = n_seeds * ncols * nrows
    n_append_x = nrows if append_x else 0
    n_append_y = ncols if append_y else 0
    n_trees = n_base + n_append_x + n_append_y

    best_score, best_xs, best_ys, best_degs, best_a, best_b = sa_optimize_improved(
        seed_xs, seed_ys, seed_degs,
        a_init, b_init,
        ncols, nrows,
        append_x, append_y,
        params["Tmax"],
        params["Tmin"],
        params["nsteps"],
        params["nsteps_per_T"],
        params["position_delta"],
        params["angle_delta"],
        params["angle_delta2"],
        params["delta_t"],
        seed,
    )

    # Get final grid positions
    final_xs, final_ys, final_degs = get_final_grid_positions_extended(
        best_xs, best_ys, best_degs, best_a, best_b, ncols, nrows, append_x, append_y
    )

    tree_data = [(final_xs[i], final_ys[i], final_degs[i]) for i in range(len(final_xs))]

    return n_trees, best_score, tree_data

@njit(cache=True)
def deletion_cascade_numba(all_xs, all_ys, all_degs, group_sizes):
    """
    Apply tree deletion cascade using numba.
    """
    # Build index mapping
    group_start = np.zeros(201, dtype=np.int64)
    for n in range(1, 201):
        group_start[n] = group_start[n-1] + (n - 1) if n > 1 else 0

    # Copy arrays
    new_xs = all_xs.copy()
    new_ys = all_ys.copy()
    new_degs = all_degs.copy()

    # Calculate initial side lengths
    side_lengths = np.zeros(201, dtype=np.float64)
    for n in range(1, 201):
        start = group_start[n]
        end = start + n
        vertices = [get_tree_vertices(new_xs[i], new_ys[i], new_degs[i]) for i in range(start, end)]
        side_lengths[n] = get_side_length(vertices)

    # Cascade from n=200 down to n=2
    for n in range(200, 1, -1):
        start_n = group_start[n]
        end_n = start_n + n
        start_prev = group_start[n - 1]

        best_side = side_lengths[n - 1]
        best_delete_idx = -1

        for del_idx in range(n):
            vertices = []
            for i in range(n):
                if i != del_idx:
                    idx = start_n + i
                    vertices.append(get_tree_vertices(new_xs[idx], new_ys[idx], new_degs[idx]))

            candidate_side = get_side_length(vertices)
            if candidate_side < best_side:
                best_side = candidate_side
                best_delete_idx = del_idx

        if best_delete_idx >= 0:
            out_idx = start_prev
            for i in range(n):
                if i != best_delete_idx:
                    in_idx = start_n + i
                    new_xs[out_idx] = new_xs[in_idx]
                    new_ys[out_idx] = new_ys[in_idx]
                    new_degs[out_idx] = new_degs[in_idx]
                    out_idx += 1
            side_lengths[n - 1] = best_side

    return new_xs, new_ys, new_degs, side_lengths

def load_submission_data(filepath):
    """Load submission and return flattened arrays."""
    df = pd.read_csv(filepath)

    all_xs = []
    all_ys = []
    all_degs = []

    for n in range(1, 201):
        prefix = f"{n:03d}_"
        group = df[df["id"].str.startswith(prefix)].sort_values("id")
        for _, row in group.iterrows():
            x = float(row["x"][1:]) if isinstance(row["x"], str) else float(row["x"])
            y = float(row["y"][1:]) if isinstance(row["y"], str) else float(row["y"])
            deg = float(row["deg"][1:]) if isinstance(row["deg"], str) else float(row["deg"])
            all_xs.append(x)
            all_ys.append(y)
            all_degs.append(deg)

    return np.array(all_xs), np.array(all_ys), np.array(all_degs)


def save_submission(filepath, all_xs, all_ys, all_degs):
    """Save submission from flattened arrays."""
    rows = []
    idx = 0
    for n in range(1, 201):
        for t in range(n):
            rows.append({
                "id": f"{n:03d}_{t}",
                "x": f"s{all_xs[idx]}",
                "y": f"s{all_ys[idx]}",
                "deg": f"s{all_degs[idx]}",
            })
            idx += 1

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)


def calculate_total_score(all_xs, all_ys, all_degs):
    """Calculate total score across all groups."""
    total = 0.0
    idx = 0
    for n in range(1, 201):
        vertices = [get_tree_vertices(all_xs[idx + i], all_ys[idx + i], all_degs[idx + i]) for i in range(n)]
        score = calculate_score_numba(vertices)
        total += score
        idx += n
    return total

print("=" * 80)
print("Improved SA Translation Optimizer (Numba-accelerated)")
print("Features: SA-optimized translations, rotate_all, append_x/y support")
print("=" * 80)


# Find baseline submission
#baseline_path = '/kaggle/input/santa-submission/submission.csv'
baseline_path = '/kaggle/input/santa2025-solutions-guided-refinement/submission.csv'
print(f"\nBaseline: {baseline_path}")

# Load baseline
baseline_xs, baseline_ys, baseline_degs = load_submission_data(baseline_path)
baseline_total = calculate_total_score(baseline_xs, baseline_ys, baseline_degs)
print(f"Baseline total score: {baseline_total:.6f}")

# Initial 2-tree seed configuration
initial_seeds = [
    (-4.191683864412409, -4.498489528496051, 74.54421568660419),
    (-4.92202045352307, -4.727639556649786, 254.5401905706735),
]

# Initial translation lengths
a_init = 0.8744896974945239
b_init = 0.7499641699190263

# Grid configurations: (ncols, nrows, append_x, append_y)
grid_configs = [
    (3, 5, False, False),   # 30 trees
    (4, 5, False, False),   # 40 trees
    (4, 6, False, False),   # 48 trees
    (4, 7, False, False),   # 56 trees
    (5, 7, False, True),    # 75 trees
    (5, 8, False, False),   # 80 trees
    (6, 7, False, False),   # 84 trees
    (7, 11, False, True),   # 161 trees
    (8, 12, False, True),   # 200 trees
]

# Generate more configurations for better coverage
for ncols in range(2, 11):
    for nrows in range(ncols, 15):
        n_trees = 2 * ncols * nrows
        if 20 <= n_trees <= 200:
            if (ncols, nrows, False, False) not in grid_configs:
                grid_configs.append((ncols, nrows, False, False))
            n_with_append_y = n_trees + ncols
            if n_with_append_y <= 200:
                if (ncols, nrows, False, True) not in grid_configs:
                    grid_configs.append((ncols, nrows, False, True))
            n_with_append_x = n_trees + nrows
            if n_with_append_x <= 200:
                if (ncols, nrows, True, False) not in grid_configs:
                    grid_configs.append((ncols, nrows, True, False))

# Remove duplicates and sort
grid_configs = list(set(grid_configs))
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
    1.0, 1.0, 2, 2, False, False,
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
    # Get baseline score for this n
    idx = sum(range(1, n_trees))
    baseline_vertices = [get_tree_vertices(baseline_xs[idx + i], baseline_ys[idx + i], baseline_degs[idx + i]) for i in range(n_trees)]
    baseline_score = calculate_score_numba(baseline_vertices)

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
    idx = sum(range(1, n_trees))
    for i in range(n_trees):
        merged_xs[idx + i] = tree_data[i][0]
        merged_ys[idx + i] = tree_data[i][1]
        merged_degs[idx + i] = tree_data[i][2]

pre_cascade_score = calculate_total_score(merged_xs, merged_ys, merged_degs)
print(f"Score after SA merge: {pre_cascade_score:.6f}")

# Apply tree deletion cascade
print("Applying tree deletion cascade...")
t0 = time.time()
final_xs, final_ys, final_degs, side_lengths = deletion_cascade_numba(
    merged_xs, merged_ys, merged_degs,
    np.arange(1, 201, dtype=np.int64)
)
print(f"Cascade completed in {time.time() - t0:.1f}s")

final_score = calculate_total_score(final_xs, final_ys, final_degs)

print("=" * 80)
print("Summary:")
print(f"  Baseline total:      {baseline_total:.6f}")
print(f"  After SA:            {pre_cascade_score:.6f}")
print(f"  After cascade:       {final_score:.6f}")
print(f"  Total improvement:   {baseline_total - final_score:+.6f}")
print("=" * 80)

# Save if improved
if final_score < baseline_total:
    output_path = "submission.csv"
    save_submission(output_path, final_xs, final_ys, final_degs)
    print(f"Saved to {output_path}")
else:
    print("No improvement - keeping baseline")

print(f"\nNew total score: {final_score:.9f}")

if os.path.exists('submission.csv'):
    cmd = f'python /kaggle/input/santa-2025-helpers/fix_overlap.py {baseline_path} submission.csv'
    os.system(cmd)