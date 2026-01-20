"""
Lattice-based construction for tree packing.
Try different lattice patterns and orientations.
"""
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import pandas as pd
import math

TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])
BASE_TREE = Polygon(zip(TX, TY))

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

def hexagonal_lattice(n, spacing=0.6):
    """Generate hexagonal lattice positions."""
    positions = []
    row = 0
    while len(positions) < n:
        cols_in_row = int(math.ceil(math.sqrt(n)))
        for col in range(cols_in_row):
            if len(positions) >= n:
                break
            x = col * spacing
            if row % 2 == 1:
                x += spacing / 2
            y = row * spacing * 0.866  # sqrt(3)/2
            positions.append((x, y))
        row += 1
    return positions[:n]

def square_lattice(n, spacing=0.6):
    """Generate square lattice positions."""
    positions = []
    side = int(math.ceil(math.sqrt(n)))
    for row in range(side):
        for col in range(side):
            if len(positions) >= n:
                break
            positions.append((col * spacing, row * spacing))
    return positions[:n]

def try_lattice_config(n, positions, angles, spacing_mult=1.0):
    """Try a lattice configuration with given positions and angles."""
    # Center the positions
    xs = [p[0] * spacing_mult for p in positions]
    ys = [p[1] * spacing_mult for p in positions]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    xs = [x - cx for x in xs]
    ys = [y - cy for y in ys]
    
    trees = [create_tree(xs[i], ys[i], angles[i % len(angles)]) for i in range(n)]
    
    if check_any_overlap(trees):
        return None, float('inf')
    
    side = get_side(trees)
    score = side**2 / n
    return (xs, ys, [angles[i % len(angles)] for i in range(n)]), score

def optimize_lattice(n, lattice_type='hexagonal'):
    """Try different lattice configurations and find the best."""
    best_config = None
    best_score = float('inf')
    
    # Generate base positions
    if lattice_type == 'hexagonal':
        base_positions = hexagonal_lattice(n, spacing=1.0)
    else:
        base_positions = square_lattice(n, spacing=1.0)
    
    # Try different angle patterns
    angle_patterns = [
        [0, 180],
        [45, 225],
        [90, 270],
        [0, 90, 180, 270],
        [45, 135, 225, 315],
        [68, 248],  # Current baseline pattern
        [0],
        [45],
        [90],
    ]
    
    # Try different spacings
    for spacing in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
        for angles in angle_patterns:
            config, score = try_lattice_config(n, base_positions, angles, spacing)
            if config is not None and score < best_score:
                best_config = config
                best_score = score
    
    return best_config, best_score

def main():
    print("Lattice Construction Optimization")
    print("=" * 50)
    
    # Load baseline
    baseline_df = pd.read_csv('/home/submission/submission.csv')
    
    for n in range(2, 16):
        # Get baseline score
        prefix = f"{n:03d}_"
        n_rows = baseline_df[baseline_df['id'].str.startswith(prefix)]
        baseline_trees = []
        for _, row in n_rows.iterrows():
            x = float(str(row['x']).replace('s', ''))
            y = float(str(row['y']).replace('s', ''))
            deg = float(str(row['deg']).replace('s', ''))
            baseline_trees.append(create_tree(x, y, deg))
        baseline_score = get_side(baseline_trees)**2 / n
        
        # Try lattice
        hex_config, hex_score = optimize_lattice(n, 'hexagonal')
        sq_config, sq_score = optimize_lattice(n, 'square')
        
        best_lattice_score = min(hex_score, sq_score)
        lattice_type = 'hex' if hex_score < sq_score else 'sq'
        
        improvement = baseline_score - best_lattice_score
        status = "✓ BETTER" if improvement > 0.0001 else "= same" if abs(improvement) < 0.0001 else "✗ worse"
        
        print(f"N={n}: baseline={baseline_score:.6f}, lattice({lattice_type})={best_lattice_score:.6f}, diff={improvement:+.6f} {status}")

if __name__ == "__main__":
    main()
