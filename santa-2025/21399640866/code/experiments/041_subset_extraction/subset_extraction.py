"""
Subset Extraction (balabaskar technique)
Extract smaller N configurations from larger N configurations by selecting trees closest to corners.
This could improve small N values which have the worst packing efficiency.
"""
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
from numba import njit
import math
import json
import time

# Tree vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def get_tree_polygon(x, y, angle):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = rotate(poly, angle, origin=(0, 0), use_radians=False)
    poly = translate(poly, x, y)
    return poly

def check_overlaps(trees):
    """Check if any trees overlap."""
    n = len(trees)
    if n <= 1:
        return False
    polygons = [get_tree_polygon(x, y, a) for x, y, a in trees]
    for i in range(n):
        for j in range(i+1, n):
            if polygons[i].intersects(polygons[j]):
                if not polygons[i].touches(polygons[j]):
                    area = polygons[i].intersection(polygons[j]).area
                    if area > 1e-12:
                        return True
    return False

def compute_bbox_side(trees):
    """Compute bounding box side length for a list of (x, y, angle) tuples"""
    polygons = [get_tree_polygon(x, y, a) for x, y, a in trees]
    union = unary_union(polygons)
    bounds = union.bounds
    return max(bounds[2] - bounds[0], bounds[3] - bounds[1])

def compute_score(trees):
    """Compute S²/N score"""
    side = compute_bbox_side(trees)
    return side * side / len(trees)

def strip(v):
    return float(str(v).replace("s", ""))

def main():
    print("=" * 70)
    print("Subset Extraction (balabaskar technique)")
    print("=" * 70)
    
    # Load current best solution
    df = pd.read_csv('/home/submission/submission.csv')
    df['N'] = df['id'].str.split('_').str[0].astype(int)
    
    # Calculate baseline per-N scores
    baseline_scores = {}
    baseline_configs = {}
    for n in range(1, 201):
        g = df[df['N'] == n]
        trees = [(strip(row['x']), strip(row['y']), strip(row['deg'])) for _, row in g.iterrows()]
        baseline_configs[n] = trees
        baseline_scores[n] = compute_score(trees)
    
    baseline_total = sum(baseline_scores.values())
    print(f"Baseline total: {baseline_total:.6f}")
    
    # SUBSET EXTRACTION: For each large N, extract subsets for smaller N
    improvements = []
    best_improvements = {}  # Track best improvement for each target_n
    
    start_time = time.time()
    
    for source_n in range(20, 201):  # Start from N=20
        source_trees = baseline_configs[source_n]
        source_polygons = [get_tree_polygon(x, y, a) for x, y, a in source_trees]
        
        # Get bounding box corners
        union = unary_union(source_polygons)
        bounds = union.bounds
        corners = [
            (bounds[0], bounds[1]),  # bottom-left
            (bounds[0], bounds[3]),  # top-left
            (bounds[2], bounds[1]),  # bottom-right
            (bounds[2], bounds[3]),  # top-right
        ]
        
        for corner_x, corner_y in corners:
            # Calculate distance from each tree to this corner
            distances = []
            for i, poly in enumerate(source_polygons):
                tree_bounds = poly.bounds
                dist = max(
                    abs(tree_bounds[0] - corner_x),
                    abs(tree_bounds[2] - corner_x),
                    abs(tree_bounds[1] - corner_y),
                    abs(tree_bounds[3] - corner_y),
                )
                distances.append((dist, i))
            
            # Sort by distance (closest first)
            distances.sort()
            
            # Extract subsets of increasing size (focus on small N)
            for target_n in range(2, min(source_n, 30)):  # Focus on N=2-29
                # Take the target_n closest trees
                subset_indices = [idx for _, idx in distances[:target_n]]
                subset_trees = [source_trees[i] for i in subset_indices]
                
                # Check for overlaps
                if check_overlaps(subset_trees):
                    continue
                
                subset_score = compute_score(subset_trees)
                baseline_score = baseline_scores[target_n]
                
                if subset_score < baseline_score - 0.0001:  # Improvement threshold
                    improvement = baseline_score - subset_score
                    
                    # Track best improvement for this target_n
                    if target_n not in best_improvements or improvement > best_improvements[target_n][4]:
                        best_improvements[target_n] = (target_n, source_n, corner_x, corner_y, improvement, subset_trees)
                        print(f"✅ N={target_n} from N={source_n} corner ({corner_x:.2f},{corner_y:.2f}): "
                              f"{baseline_score:.6f} -> {subset_score:.6f} (+{improvement:.6f})")
        
        # Progress update
        if source_n % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: source_n={source_n}, elapsed={elapsed:.1f}s, improvements={len(best_improvements)}")
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"Subset Extraction Complete")
    print(f"  Elapsed time: {elapsed:.1f}s")
    print(f"  Improvements found: {len(best_improvements)}")
    
    if best_improvements:
        total_improvement = sum(imp[4] for imp in best_improvements.values())
        print(f"  Total improvement: {total_improvement:.6f}")
        
        # Update best solutions
        for target_n, (_, source_n, cx, cy, imp, trees) in best_improvements.items():
            baseline_configs[target_n] = trees
            baseline_scores[target_n] = compute_score(trees)
        
        new_total = sum(baseline_scores.values())
        print(f"  New total: {new_total:.6f}")
        print(f"  Net improvement: {baseline_total - new_total:.6f}")
        
        # Save updated submission
        rows = []
        for n in range(1, 201):
            trees = baseline_configs[n]
            for i, (x, y, angle) in enumerate(trees):
                rows.append({
                    'id': f'{n:03d}_{i}',
                    'x': f's{x:.20f}',
                    'y': f's{y:.20f}',
                    'deg': f's{angle:.20f}'
                })
        
        result_df = pd.DataFrame(rows)
        result_df.to_csv('submission.csv', index=False)
        result_df.to_csv('/home/submission/submission.csv', index=False)
        print(f"\nSubmission saved with {len(best_improvements)} improvements")
    else:
        print("  No improvements found")
    
    print(f"{'='*70}")
    
    # Save results
    results = {
        'improvements': [(n, imp[1], imp[4]) for n, imp in best_improvements.items()],
        'total_improvement': sum(imp[4] for imp in best_improvements.values()) if best_improvements else 0,
        'elapsed_time': elapsed
    }
    
    with open('subset_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return best_improvements

if __name__ == "__main__":
    improvements = main()
