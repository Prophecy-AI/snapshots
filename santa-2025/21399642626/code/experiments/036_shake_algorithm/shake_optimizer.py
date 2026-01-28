import numpy as np
import pandas as pd
from numba import jit
import math
import time

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

@jit(nopython=True)
def get_tree_vertices_numba(x, y, deg):
    """Get the vertices for a tree at position (x, y) with rotation deg."""
    rad = deg * np.pi / 180.0
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    vertices_x = np.zeros(15)
    vertices_y = np.zeros(15)
    for i in range(15):
        vertices_x[i] = TX[i] * cos_a - TY[i] * sin_a + x
        vertices_y[i] = TX[i] * sin_a + TY[i] * cos_a + y
    return vertices_x, vertices_y

@jit(nopython=True)
def calculate_bounding_box(trees_x, trees_y, trees_deg, n):
    """Calculate the bounding box side length."""
    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf
    
    for i in range(n):
        vx, vy = get_tree_vertices_numba(trees_x[i], trees_y[i], trees_deg[i])
        for j in range(15):
            if vx[j] < min_x:
                min_x = vx[j]
            if vx[j] > max_x:
                max_x = vx[j]
            if vy[j] < min_y:
                min_y = vy[j]
            if vy[j] > max_y:
                max_y = vy[j]
    
    return max(max_x - min_x, max_y - min_y)

@jit(nopython=True)
def point_in_polygon(px, py, poly_x, poly_y, n_vertices):
    """Check if a point is inside a polygon using ray casting."""
    inside = False
    j = n_vertices - 1
    for i in range(n_vertices):
        if ((poly_y[i] > py) != (poly_y[j] > py)) and \
           (px < (poly_x[j] - poly_x[i]) * (py - poly_y[i]) / (poly_y[j] - poly_y[i]) + poly_x[i]):
            inside = not inside
        j = i
    return inside

@jit(nopython=True)
def segments_intersect(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    """Check if two line segments intersect."""
    def ccw(ax, ay, bx, by, cx, cy):
        return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)
    
    return ccw(ax1, ay1, bx1, by1, bx2, by2) != ccw(ax2, ay2, bx1, by1, bx2, by2) and \
           ccw(ax1, ay1, ax2, ay2, bx1, by1) != ccw(ax1, ay1, ax2, ay2, bx2, by2)

@jit(nopython=True)
def polygons_overlap(vx1, vy1, vx2, vy2):
    """Check if two polygons overlap."""
    for i in range(15):
        if point_in_polygon(vx1[i], vy1[i], vx2, vy2, 15):
            return True
    for i in range(15):
        if point_in_polygon(vx2[i], vy2[i], vx1, vy1, 15):
            return True
    for i in range(15):
        i_next = (i + 1) % 15
        for j in range(15):
            j_next = (j + 1) % 15
            if segments_intersect(vx1[i], vy1[i], vx1[i_next], vy1[i_next],
                                  vx2[j], vy2[j], vx2[j_next], vy2[j_next]):
                return True
    return False

@jit(nopython=True)
def check_any_overlap(trees_x, trees_y, trees_deg, n):
    """Check if any trees overlap."""
    for i in range(n):
        vx1, vy1 = get_tree_vertices_numba(trees_x[i], trees_y[i], trees_deg[i])
        for j in range(i+1, n):
            vx2, vy2 = get_tree_vertices_numba(trees_x[j], trees_y[j], trees_deg[j])
            if polygons_overlap(vx1, vy1, vx2, vy2):
                return True
    return False

@jit(nopython=True)
def shake_optimizer(trees_x, trees_y, trees_deg, n, iterations=50000, T0=0.5, Tf=0.0001):
    """
    Shake optimizer - different from standard SA.
    
    Key differences:
    1. Perturb ALL trees slightly each iteration (not just one)
    2. Apply "compression" force toward center
    3. Use different move types: jiggle, compress, rotate-all
    """
    best_x = trees_x.copy()
    best_y = trees_y.copy()
    best_deg = trees_deg.copy()
    best_side = calculate_bounding_box(best_x, best_y, best_deg, n)
    
    current_x = trees_x.copy()
    current_y = trees_y.copy()
    current_deg = trees_deg.copy()
    current_side = best_side
    
    alpha = (Tf / T0) ** (1.0 / iterations)
    T = T0
    
    for it in range(iterations):
        # Save current state
        saved_x = current_x.copy()
        saved_y = current_y.copy()
        saved_deg = current_deg.copy()
        
        # Choose move type
        move_type = np.random.randint(0, 5)
        
        if move_type == 0:
            # JIGGLE ALL: Small perturbation to all trees
            for i in range(n):
                current_x[i] += np.random.normal(0, T * 0.02)
                current_y[i] += np.random.normal(0, T * 0.02)
                current_deg[i] += np.random.normal(0, T * 2.0)
        
        elif move_type == 1:
            # COMPRESS: Move all trees toward center
            cx = np.mean(current_x)
            cy = np.mean(current_y)
            compress_factor = 0.01 * T
            for i in range(n):
                dx = cx - current_x[i]
                dy = cy - current_y[i]
                current_x[i] += dx * compress_factor
                current_y[i] += dy * compress_factor
        
        elif move_type == 2:
            # ROTATE ALL: Rotate all trees by small amount
            angle_delta = np.random.normal(0, T * 5.0)
            for i in range(n):
                current_deg[i] += angle_delta
        
        elif move_type == 3:
            # SINGLE TREE: Standard SA move on one tree
            idx = np.random.randint(0, n)
            current_x[idx] += np.random.normal(0, T * 0.1)
            current_y[idx] += np.random.normal(0, T * 0.1)
            current_deg[idx] += np.random.normal(0, T * 10.0)
        
        else:
            # SWAP: Swap positions of two random trees
            if n >= 2:
                i = np.random.randint(0, n)
                j = np.random.randint(0, n)
                if i != j:
                    current_x[i], current_x[j] = current_x[j], current_x[i]
                    current_y[i], current_y[j] = current_y[j], current_y[i]
        
        # Check for overlaps
        if check_any_overlap(current_x, current_y, current_deg, n):
            # Revert
            current_x = saved_x
            current_y = saved_y
            current_deg = saved_deg
        else:
            # Calculate new score
            new_side = calculate_bounding_box(current_x, current_y, current_deg, n)
            
            # Accept or reject (Metropolis criterion)
            delta = new_side - current_side
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_side = new_side
                if new_side < best_side:
                    best_side = new_side
                    best_x = current_x.copy()
                    best_y = current_y.copy()
                    best_deg = current_deg.copy()
            else:
                # Revert
                current_x = saved_x
                current_y = saved_y
                current_deg = saved_deg
        
        T *= alpha
    
    return best_x, best_y, best_deg, best_side

def parse_coord(val):
    if isinstance(val, str) and val.startswith('s'):
        return float(val[1:])
    return float(val)

def parse_id(id_str):
    parts = str(id_str).split('_')
    n = int(parts[0])
    i = int(parts[1])
    return n, i

# Load exp_029 baseline
baseline_df = pd.read_csv('/home/code/experiments/029_final_ensemble_v2/submission.csv')
baseline_df['n'] = baseline_df['id'].apply(lambda x: parse_id(x)[0])
baseline_df['i'] = baseline_df['id'].apply(lambda x: parse_id(x)[1])
for col in ['x', 'y', 'deg']:
    baseline_df[col] = baseline_df[col].apply(parse_coord)

# Test shake optimizer on small N values
print("Testing SHAKE optimizer on small N values:")
print("(Starting from baseline solution)")

total_improvement = 0
for n in [10, 20, 30, 50]:
    n_df = baseline_df[baseline_df['n'] == n].sort_values('i')
    trees_x = np.array(n_df['x'].values)
    trees_y = np.array(n_df['y'].values)
    trees_deg = np.array(n_df['deg'].values)
    
    baseline_side = calculate_bounding_box(trees_x, trees_y, trees_deg, n)
    baseline_score = baseline_side**2 / n
    
    print(f"\nN={n}: baseline score = {baseline_score:.6f}")
    
    # Run shake optimizer
    start_time = time.time()
    best_x, best_y, best_deg, best_side = shake_optimizer(
        trees_x, trees_y, trees_deg, n, iterations=100000, T0=0.3, Tf=0.0001
    )
    elapsed = time.time() - start_time
    
    new_score = best_side**2 / n
    improvement = baseline_score - new_score
    total_improvement += improvement
    status = "✅ BETTER" if improvement > 1e-9 else "❌ NO IMPROVEMENT"
    print(f"  Shake score = {new_score:.6f} ({status}, diff={improvement:.9f})")
    print(f"  Time: {elapsed:.2f}s")

print(f"\nTotal improvement across tested N values: {total_improvement:.9f}")
