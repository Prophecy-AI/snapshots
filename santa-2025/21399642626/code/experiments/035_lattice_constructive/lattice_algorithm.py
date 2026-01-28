import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely import affinity
import math

# Tree polygon vertices
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]

def get_tree_polygon(x, y, deg):
    """Get the polygon for a tree at position (x, y) with rotation deg."""
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = affinity.rotate(poly, deg, origin=(0, 0))
    poly = affinity.translate(poly, x, y)
    return poly

def get_tree_vertices(x, y, deg):
    """Get the vertices for a tree at position (x, y) with rotation deg."""
    rad = np.radians(deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    vertices = []
    for tx, ty in zip(TX, TY):
        rx = tx * cos_a - ty * sin_a + x
        ry = tx * sin_a + ty * cos_a + y
        vertices.append((rx, ry))
    return vertices

def check_overlap(tree1, tree2):
    """Check if two trees overlap."""
    poly1 = get_tree_polygon(tree1['x'], tree1['y'], tree1['deg'])
    poly2 = get_tree_polygon(tree2['x'], tree2['y'], tree2['deg'])
    if poly1.intersects(poly2) and not poly1.touches(poly2):
        intersection = poly1.intersection(poly2)
        if intersection.area > 1e-12:
            return True
    return False

def check_all_overlaps(trees):
    """Check if any trees overlap."""
    for i in range(len(trees)):
        for j in range(i+1, len(trees)):
            if check_overlap(trees[i], trees[j]):
                return True
    return False

def calculate_score(trees, n):
    """Calculate the score for a configuration."""
    all_x = []
    all_y = []
    for tree in trees:
        vertices = get_tree_vertices(tree['x'], tree['y'], tree['deg'])
        for vx, vy in vertices:
            all_x.append(vx)
            all_y.append(vy)
    side = max(max(all_x) - min(all_x), max(all_y) - min(all_y))
    return side**2 / n

def generate_lattice_solution(n, h_spacing=0.55, v_spacing=0.65, angle1=45.0, angle2=225.0):
    """
    Generate a lattice-based solution for N trees.
    
    Key insight: Trees can interlock when rotated 180° apart.
    The optimal pattern is a 2D lattice where:
    - Even rows: trees at angle1
    - Odd rows: trees at angle2
    - Offset positions to allow interlocking
    """
    trees = []
    
    # Calculate grid dimensions
    cols = int(math.ceil(math.sqrt(n * 1.2)))
    rows = int(math.ceil(n / cols))
    
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            x = col * h_spacing
            y = row * v_spacing
            # Alternate angles for interlocking
            if (row + col) % 2 == 0:
                deg = angle1
            else:
                deg = angle2
            trees.append({'x': x, 'y': y, 'deg': deg})
            idx += 1
    
    return trees

def optimize_lattice_spacing(n, angle1=45.0, angle2=225.0):
    """Find optimal spacing for the lattice."""
    best_score = float('inf')
    best_config = None
    best_params = None
    
    # Search over spacing parameters
    for h_spacing in np.arange(0.4, 0.8, 0.02):
        for v_spacing in np.arange(0.4, 0.8, 0.02):
            trees = generate_lattice_solution(n, h_spacing, v_spacing, angle1, angle2)
            
            # Check for overlaps
            if check_all_overlaps(trees):
                continue
            
            score = calculate_score(trees, n)
            if score < best_score:
                best_score = score
                best_config = trees
                best_params = (h_spacing, v_spacing)
    
    return best_config, best_score, best_params

# Load baseline scores
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

# Calculate baseline scores
baseline_scores = {}
for n in range(1, 201):
    n_df = baseline_df[baseline_df['n'] == n]
    trees = n_df[['x', 'y', 'deg']].to_dict('records')
    baseline_scores[n] = calculate_score(trees, n)

print(f"Baseline total score: {sum(baseline_scores.values()):.6f}")

# Test lattice algorithm on small N values
print("\nTesting lattice algorithm on small N values:")
for n in [5, 10, 15, 20, 25, 30]:
    config, score, params = optimize_lattice_spacing(n)
    if config is not None:
        baseline = baseline_scores[n]
        diff = baseline - score
        status = "✅ BETTER" if diff > 0 else "❌ WORSE"
        print(f"N={n}: lattice={score:.6f} vs baseline={baseline:.6f} ({status}, diff={diff:.6f})")
        if params:
            print(f"       Best params: h_spacing={params[0]:.2f}, v_spacing={params[1]:.2f}")
    else:
        print(f"N={n}: No valid configuration found (all have overlaps)")
