"""
Translation-based Grid Construction for tree packing.
Based on egortrushin kernel approach - place trees in a grid pattern
with alternating angles.
"""
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import pandas as pd
import math
import itertools

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

def get_factors(n):
    """Get all factor pairs (nx, ny) where nx * ny = n."""
    factors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factors.append((i, n // i))
            if i != n // i:
                factors.append((n // i, i))
    return factors

def create_grid_config(nx, ny, dx, dy, angle1, angle2, offset_x=0, offset_y=0):
    """Create a grid of nx*ny trees with alternating angles."""
    configs = []
    for i in range(nx):
        for j in range(ny):
            x = i * dx + offset_x
            y = j * dy + offset_y
            angle = angle1 if (i + j) % 2 == 0 else angle2
            configs.append((x, y, angle))
    return configs

def try_grid_config(configs):
    """Try a grid configuration and return score if valid."""
    trees = [create_tree(x, y, deg) for x, y, deg in configs]
    
    if check_any_overlap(trees):
        return None, float('inf')
    
    # Center the configuration
    min_x = min(t.bounds[0] for t in trees)
    min_y = min(t.bounds[1] for t in trees)
    max_x = max(t.bounds[2] for t in trees)
    max_y = max(t.bounds[3] for t in trees)
    
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    
    centered_configs = [(x - cx, y - cy, deg) for x, y, deg in configs]
    
    side = get_side(trees)
    n = len(configs)
    score = side**2 / n
    
    return centered_configs, score

def optimize_grid_for_n(n, verbose=False):
    """Find the best grid configuration for N trees."""
    factors = get_factors(n)
    
    # Angle patterns to try
    angle_patterns = [
        (68, 248),   # Current baseline pattern
        (45, 225),   # Diagonal
        (0, 180),    # Vertical
        (90, 270),   # Horizontal
        (22.5, 202.5),
        (67.5, 247.5),
        (112.5, 292.5),
    ]
    
    # Spacing values to try
    spacings = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    
    best_config = None
    best_score = float('inf')
    
    for nx, ny in factors:
        for dx in spacings:
            for dy in spacings:
                for angle1, angle2 in angle_patterns:
                    configs = create_grid_config(nx, ny, dx, dy, angle1, angle2)
                    result, score = try_grid_config(configs)
                    
                    if result is not None and score < best_score:
                        best_config = result
                        best_score = score
                        if verbose:
                            print(f"    New best: {nx}x{ny}, dx={dx}, dy={dy}, angles=({angle1},{angle2}), score={score:.6f}")
    
    return best_config, best_score

def parse_value(val):
    if isinstance(val, str):
        if val.startswith('s'):
            return float(val[1:])
        return float(val)
    return float(val)

def ensure_s_prefix(val):
    if isinstance(val, str):
        if val.startswith('s'):
            return val
        return f's{val}'
    return f's{val}'

def main():
    print("Grid Construction Optimization")
    print("=" * 60)
    
    # Load baseline
    baseline_df = pd.read_csv('/home/submission/submission.csv')
    
    improvements = {}
    total_improvement = 0
    
    # Focus on N values with nice factor decompositions
    # and small N values (worst efficiency)
    test_ns = list(range(2, 21)) + [25, 36, 49, 64, 72, 81, 100, 110, 121, 144, 156, 169, 196, 200]
    test_ns = sorted(set(test_ns))
    
    for n in test_ns:
        # Get baseline score
        prefix = f"{n:03d}_"
        n_rows = baseline_df[baseline_df['id'].str.startswith(prefix)]
        baseline_trees = []
        for _, row in n_rows.iterrows():
            x = parse_value(row['x'])
            y = parse_value(row['y'])
            deg = parse_value(row['deg'])
            baseline_trees.append(create_tree(x, y, deg))
        baseline_score = get_side(baseline_trees)**2 / n
        
        # Try grid construction
        grid_config, grid_score = optimize_grid_for_n(n, verbose=False)
        
        if grid_config is not None:
            improvement = baseline_score - grid_score
            
            status = "✓ BETTER" if improvement > 0.0001 else "= same" if abs(improvement) < 0.0001 else "✗ worse"
            print(f"N={n:3d}: baseline={baseline_score:.6f}, grid={grid_score:.6f}, diff={improvement:+.6f} {status}")
            
            if improvement > 0.0001:
                improvements[n] = {
                    'config': grid_config,
                    'score': grid_score,
                    'improvement': improvement
                }
                total_improvement += improvement
        else:
            print(f"N={n:3d}: baseline={baseline_score:.6f}, grid=NO VALID CONFIG")
    
    print(f"\n{'=' * 60}")
    print(f"Total improvements found: {len(improvements)}")
    print(f"Total improvement: {total_improvement:+.6f}")
    
    if improvements:
        print("\nImproved N values:")
        for n, data in sorted(improvements.items()):
            print(f"  N={n}: improvement={data['improvement']:.6f}")
    
    return improvements

if __name__ == "__main__":
    improvements = main()
