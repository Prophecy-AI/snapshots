"""
Bottom-Left Beam Search for tree packing.
Constructive approach: build configurations from scratch using beam search.
"""
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import pandas as pd
from copy import deepcopy
import time

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])
BASE_TREE = Polygon(zip(TX, TY))

# Pre-compute rotated trees for common angles
ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
ROTATED_TREES = {}
for angle in ANGLES:
    ROTATED_TREES[angle] = affinity.rotate(BASE_TREE, angle, origin=(0, 0))

def create_tree(x, y, deg):
    """Create a tree polygon at position (x, y) with rotation deg."""
    if deg in ROTATED_TREES:
        tree = affinity.translate(ROTATED_TREES[deg], x, y)
    else:
        tree = affinity.rotate(BASE_TREE, deg, origin=(0, 0))
        tree = affinity.translate(tree, x, y)
    return tree

def get_bounding_box(trees):
    """Get bounding box of all trees."""
    if not trees:
        return 0, 0, 0, 0
    
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    
    for tree in trees:
        bounds = tree.bounds
        min_x = min(min_x, bounds[0])
        min_y = min(min_y, bounds[1])
        max_x = max(max_x, bounds[2])
        max_y = max(max_y, bounds[3])
    
    return min_x, min_y, max_x, max_y

def get_side(trees):
    """Get the side length of the bounding box."""
    if not trees:
        return 0
    min_x, min_y, max_x, max_y = get_bounding_box(trees)
    return max(max_x - min_x, max_y - min_y)

def check_overlap(new_tree, existing_trees, tolerance=1e-9):
    """Check if new tree overlaps with any existing tree."""
    for tree in existing_trees:
        if new_tree.intersects(tree):
            intersection = new_tree.intersection(tree)
            if intersection.area > tolerance:
                return True
    return False

def find_bottom_left_position(existing_trees, angle, grid_step=0.05, max_range=10):
    """Find the bottom-left valid position for a tree with given angle."""
    rotated = ROTATED_TREES.get(angle, affinity.rotate(BASE_TREE, angle, origin=(0, 0)))
    
    if not existing_trees:
        # First tree - place at origin
        return 0, 0
    
    # Get current bounding box
    min_x, min_y, max_x, max_y = get_bounding_box(existing_trees)
    
    # Search for bottom-left position
    # Start from bottom-left corner and scan
    search_min_x = min_x - 1.5
    search_min_y = min_y - 1.5
    search_max_x = max_x + 1.5
    search_max_y = max_y + 1.5
    
    best_pos = None
    best_score = float('inf')
    
    # Scan from bottom to top, left to right
    y = search_min_y
    while y <= search_max_y:
        x = search_min_x
        while x <= search_max_x:
            tree = affinity.translate(rotated, x, y)
            if not check_overlap(tree, existing_trees):
                # Calculate score (side of bounding box with this tree)
                test_trees = existing_trees + [tree]
                side = get_side(test_trees)
                score = side
                
                if score < best_score:
                    best_score = score
                    best_pos = (x, y)
            x += grid_step
        y += grid_step
    
    return best_pos

def bottom_left_beam_search(n, beam_width=10, angles=None):
    """Build n-tree configuration using bottom-left placement with beam search."""
    if angles is None:
        angles = ANGLES
    
    if n == 1:
        # Optimal single tree at 45 degrees
        return [(0, 0, 45)], get_side([create_tree(0, 0, 45)])
    
    # Start with empty configuration
    # State: (trees_list, polygons_list, score)
    states = [([], [], 0)]
    
    for tree_idx in range(n):
        candidates = []
        
        for trees, polys, _ in states:
            for angle in angles:
                pos = find_bottom_left_position(polys, angle)
                if pos is not None:
                    x, y = pos
                    new_tree = create_tree(x, y, angle)
                    new_trees = trees + [(x, y, angle)]
                    new_polys = polys + [new_tree]
                    new_score = get_side(new_polys)
                    candidates.append((new_trees, new_polys, new_score))
        
        if not candidates:
            print(f"  No valid candidates at tree {tree_idx}")
            break
        
        # Keep top beam_width by score
        candidates.sort(key=lambda x: x[2])
        states = candidates[:beam_width]
        
        if tree_idx < 5 or tree_idx % 10 == 0:
            print(f"  Tree {tree_idx+1}/{n}: best side = {states[0][2]:.6f}")
    
    if states:
        best_trees, best_polys, best_score = states[0]
        return best_trees, best_score
    return None, float('inf')

def optimize_small_n(max_n=20, beam_width=15):
    """Optimize small N values using beam search."""
    results = {}
    
    for n in range(1, max_n + 1):
        print(f"\nOptimizing N={n}...")
        start = time.time()
        trees, side = bottom_left_beam_search(n, beam_width=beam_width)
        elapsed = time.time() - start
        
        if trees:
            score = side**2 / n
            results[n] = {
                'trees': trees,
                'side': side,
                'score': score
            }
            print(f"  N={n}: side={side:.6f}, score={score:.6f}, time={elapsed:.1f}s")
    
    return results

def main():
    print("Bottom-Left Beam Search Optimization")
    print("=" * 50)
    
    # Start with small N values (most expensive per-N)
    results = optimize_small_n(max_n=15, beam_width=10)
    
    # Compare with baseline
    print("\n" + "=" * 50)
    print("Results vs Baseline:")
    
    # Load baseline for comparison
    baseline_df = pd.read_csv('/home/submission/submission.csv')
    
    total_improvement = 0
    for n, data in results.items():
        # Get baseline score for this N
        prefix = f"{n:03d}_"
        n_rows = baseline_df[baseline_df['id'].str.startswith(prefix)]
        
        baseline_trees = []
        for _, row in n_rows.iterrows():
            x = float(str(row['x']).replace('s', ''))
            y = float(str(row['y']).replace('s', ''))
            deg = float(str(row['deg']).replace('s', ''))
            baseline_trees.append(create_tree(x, y, deg))
        
        baseline_side = get_side(baseline_trees)
        baseline_score = baseline_side**2 / n
        
        improvement = baseline_score - data['score']
        total_improvement += improvement
        
        status = "✓ BETTER" if improvement > 0.0001 else "= same" if abs(improvement) < 0.0001 else "✗ worse"
        print(f"  N={n}: baseline={baseline_score:.6f}, new={data['score']:.6f}, diff={improvement:+.6f} {status}")
    
    print(f"\nTotal improvement for N=1-15: {total_improvement:+.6f}")
    
    return results

if __name__ == "__main__":
    results = main()
