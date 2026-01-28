"""
Subset extraction from large N solutions.
For each large N solution (N=50-200), extract subsets of trees that might form 
better solutions for smaller N values.
"""

import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def parse_coord(val):
    if isinstance(val, str):
        if val.startswith('s'):
            return float(val[1:])
        return float(val)
    return float(val)

def get_tree_polygon(x, y, angle_deg):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = affinity.rotate(poly, angle_deg, origin=(0, 0))
    poly = affinity.translate(poly, x, y)
    return poly

def get_tree_vertices(x, y, angle_deg):
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rx = TX * cos_a - TY * sin_a
    ry = TX * sin_a + TY * cos_a
    return rx + x, ry + y

def compute_bbox_size(trees):
    """Compute bounding box size using vertex coordinates."""
    all_x, all_y = [], []
    for x, y, angle in trees:
        vx, vy = get_tree_vertices(x, y, angle)
        all_x.extend(vx)
        all_y.extend(vy)
    if not all_x:
        return float('inf')
    return max(max(all_x) - min(all_x), max(all_y) - min(all_y))

def compute_score(trees, n):
    """Compute S^2/N for a list of (x, y, angle) trees."""
    if not trees or len(trees) != n:
        return float('inf')
    size = compute_bbox_size(trees)
    return (size ** 2) / n

def check_overlap(trees, threshold=1e-10):
    """Check for overlaps between trees."""
    polygons = [get_tree_polygon(x, y, a) for x, y, a in trees]
    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > threshold:
                    return True
    return False

def load_baseline(path):
    """Load baseline solution."""
    df = pd.read_csv(path)
    df['n'] = df['id'].apply(lambda x: int(x.split('_')[0]))
    df['x'] = df['x'].apply(parse_coord)
    df['y'] = df['y'].apply(parse_coord)
    df['deg'] = df['deg'].apply(parse_coord)
    
    result = {}
    for n in range(1, 201):
        n_df = df[df['n'] == n]
        if len(n_df) == n:
            result[n] = [(row['x'], row['y'], row['deg']) for _, row in n_df.iterrows()]
    return result

# Load baseline
print("Loading baseline (exp_039)...")
baseline_path = "/home/code/experiments/039_per_n_analysis/safe_ensemble.csv"
baseline = load_baseline(baseline_path)
baseline_scores = {n: compute_score(baseline[n], n) for n in range(1, 201)}
total_baseline = sum(baseline_scores.values())
print(f"Baseline total: {total_baseline:.6f}")

# SUBSET EXTRACTION
print("\n" + "="*60)
print("SUBSET EXTRACTION FROM LARGE N SOLUTIONS")
print("="*60)

improvements = {}
MIN_IMPROVEMENT = 0.001  # Safety threshold to avoid precision issues

for large_n in range(50, 201):
    large_trees = baseline[large_n]
    
    # Get bounding box of large N solution
    all_x, all_y = [], []
    for x, y, angle in large_trees:
        vx, vy = get_tree_vertices(x, y, angle)
        all_x.extend(vx)
        all_y.extend(vy)
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    # Four corners
    corners = [
        (min_x, min_y),  # bottom-left
        (min_x, max_y),  # top-left
        (max_x, min_y),  # bottom-right
        (max_x, max_y),  # top-right
    ]
    
    for corner_x, corner_y in corners:
        # Sort trees by max distance from corner (closest first)
        def tree_distance(tree):
            x, y, a = tree
            vx, vy = get_tree_vertices(x, y, a)
            return max(
                max(abs(vx - corner_x)),
                max(abs(vy - corner_y))
            )
        
        sorted_trees = sorted(large_trees, key=tree_distance)
        
        # Check subsets for smaller N
        for small_n in range(2, min(large_n, 50)):
            subset = sorted_trees[:small_n]
            subset_score = compute_score(subset, small_n)
            
            # Check if this is an improvement
            if subset_score < baseline_scores[small_n] - MIN_IMPROVEMENT:
                # Verify no overlaps
                if check_overlap(subset):
                    continue
                
                improvement = baseline_scores[small_n] - subset_score
                if small_n not in improvements or improvement > improvements[small_n][1]:
                    improvements[small_n] = (subset, improvement, large_n, (corner_x, corner_y))
                    print(f"✅ N={small_n}: {baseline_scores[small_n]:.6f} -> {subset_score:.6f} (improvement: {improvement:.6f}, from N={large_n})")

# Also try centroid-based extraction
print("\n" + "="*60)
print("CENTROID-BASED EXTRACTION")
print("="*60)

for large_n in range(50, 201):
    large_trees = baseline[large_n]
    
    # Compute centroid of all trees
    centroids = []
    for x, y, angle in large_trees:
        vx, vy = get_tree_vertices(x, y, angle)
        centroids.append((np.mean(vx), np.mean(vy)))
    
    overall_cx = np.mean([c[0] for c in centroids])
    overall_cy = np.mean([c[1] for c in centroids])
    
    # Sort trees by distance from centroid (closest first)
    def centroid_distance(tree):
        x, y, a = tree
        vx, vy = get_tree_vertices(x, y, a)
        cx, cy = np.mean(vx), np.mean(vy)
        return np.sqrt((cx - overall_cx)**2 + (cy - overall_cy)**2)
    
    sorted_trees = sorted(large_trees, key=centroid_distance)
    
    # Check subsets for smaller N
    for small_n in range(2, min(large_n, 50)):
        subset = sorted_trees[:small_n]
        subset_score = compute_score(subset, small_n)
        
        # Check if this is an improvement
        if subset_score < baseline_scores[small_n] - MIN_IMPROVEMENT:
            # Verify no overlaps
            if check_overlap(subset):
                continue
            
            improvement = baseline_scores[small_n] - subset_score
            if small_n not in improvements or improvement > improvements[small_n][1]:
                improvements[small_n] = (subset, improvement, large_n, "centroid")
                print(f"✅ N={small_n}: {baseline_scores[small_n]:.6f} -> {subset_score:.6f} (improvement: {improvement:.6f}, from N={large_n}, centroid)")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

if improvements:
    total_improvement = sum(imp[1] for imp in improvements.values())
    print(f"Total improvements found: {len(improvements)}")
    print(f"Total score improvement: {total_improvement:.6f}")
    
    print("\nTop 10 improvements:")
    for n, (trees, imp, source_n, corner) in sorted(improvements.items(), key=lambda x: -x[1][1])[:10]:
        print(f"  N={n}: improvement={imp:.6f} (from N={source_n})")
else:
    print("No improvements found from subset extraction")
    print("The baseline solutions are already optimized for each N independently")
