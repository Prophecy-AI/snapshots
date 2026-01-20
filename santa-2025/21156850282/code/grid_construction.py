"""
Grid-based construction approach - based on zaburo kernel.
Try different grid configurations to find better solutions.
"""
import numpy as np
import pandas as pd
import math
from decimal import Decimal, getcontext
from shapely.geometry import Polygon
from shapely import affinity

getcontext().prec = 25
scale_factor = Decimal('1e15')

TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])
BASE_TREE = Polygon(zip(TX, TY))

def parse_value(val):
    if isinstance(val, str):
        if val.startswith('s'):
            return float(val[1:])
        return float(val)
    return float(val)

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

def grid_construction(n, x_spacing=0.7, y_spacing=1.0, angle_offset=0):
    """
    Construct N trees in a grid pattern with alternating orientations.
    """
    trees = []
    positions = []
    
    # Calculate grid dimensions
    cols = int(math.ceil(math.sqrt(n * y_spacing / x_spacing)))
    rows = int(math.ceil(n / cols))
    
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= n:
                break
            
            # Alternating pattern
            if row % 2 == 0:
                angle = angle_offset
                x = col * x_spacing
            else:
                angle = (angle_offset + 180) % 360
                x = col * x_spacing + x_spacing / 2
            
            y = row * y_spacing
            
            tree = create_tree(x, y, angle)
            trees.append(tree)
            positions.append((x, y, angle))
            count += 1
        
        if count >= n:
            break
    
    return trees, positions

def find_best_grid(n, verbose=False):
    """Try different grid configurations and return the best one."""
    best_score = float('inf')
    best_positions = None
    
    # Try different spacings and angles
    for x_spacing in [0.65, 0.68, 0.7, 0.72, 0.75]:
        for y_spacing in [0.9, 0.95, 1.0, 1.05, 1.1]:
            for angle in [0, 45, 68, 90, 135, 180, 248]:
                trees, positions = grid_construction(n, x_spacing, y_spacing, angle)
                
                if check_any_overlap(trees):
                    continue
                
                side = get_side(trees)
                score = side**2 / n
                
                if score < best_score:
                    best_score = score
                    best_positions = positions
                    if verbose:
                        print(f"  N={n}: x={x_spacing}, y={y_spacing}, angle={angle} -> score={score:.6f}")
    
    return best_score, best_positions

# Load current submission for comparison
current_df = pd.read_csv('/home/submission/submission.csv')

print("Grid Construction Approach")
print("=" * 70)

improvements = []
total_improvement = 0

for n in range(1, 201):
    # Get current score
    prefix = f"{n:03d}_"
    n_rows = current_df[current_df['id'].str.startswith(prefix)]
    current_trees = []
    for _, row in n_rows.iterrows():
        x = parse_value(row['x'])
        y = parse_value(row['y'])
        deg = parse_value(row['deg'])
        current_trees.append(create_tree(x, y, deg))
    current_side = get_side(current_trees)
    current_score = current_side**2 / n
    
    # Try grid construction
    grid_score, grid_positions = find_best_grid(n)
    
    if grid_score < current_score - 0.0001:
        improvement = current_score - grid_score
        improvements.append((n, improvement, current_score, grid_score))
        total_improvement += improvement
        print(f"N={n}: IMPROVED! {current_score:.6f} -> {grid_score:.6f} (diff: {improvement:.6f})")

print(f"\nTotal improvements: {len(improvements)}")
print(f"Total improvement: {total_improvement:.6f}")

if improvements:
    print("\nTop 10 improvements:")
    improvements.sort(key=lambda x: -x[1])
    for n, diff, old, new in improvements[:10]:
        print(f"  N={n}: {old:.6f} -> {new:.6f} (diff: {diff:.6f})")
