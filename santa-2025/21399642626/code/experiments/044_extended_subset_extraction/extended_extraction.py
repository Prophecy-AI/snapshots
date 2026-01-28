"""
Extended subset extraction: N+k -> N for k=2,3,4,5
Try all combinations of removing k trees from N+k solution to create N solution.
"""

import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
from itertools import combinations
import time
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

def get_tree_vertices(x, y, angle_deg):
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rx = TX * cos_a - TY * sin_a
    ry = TX * sin_a + TY * cos_a
    return rx + x, ry + y

def get_tree_polygon(x, y, angle_deg):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = affinity.rotate(poly, angle_deg, origin=(0, 0))
    poly = affinity.translate(poly, x, y)
    return poly

def compute_bbox_size(trees):
    all_x, all_y = [], []
    for x, y, angle in trees:
        vx, vy = get_tree_vertices(x, y, angle)
        all_x.extend(vx)
        all_y.extend(vy)
    if not all_x:
        return float('inf')
    return max(max(all_x) - min(all_x), max(all_y) - min(all_y))

def compute_score(trees, n):
    if not trees or len(trees) != n:
        return float('inf')
    size = compute_bbox_size(trees)
    return (size ** 2) / n

def check_overlap(trees, threshold=1e-20):
    polygons = [get_tree_polygon(x, y, a) for x, y, a in trees]
    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > threshold:
                    return True
    return False

def load_baseline(path):
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

def extract_n_from_n_plus_k(source_trees, target_n, k, baseline_score, min_improvement=0.0001):
    """
    Extract target_n trees from source_trees by removing k trees.
    Returns best subset and score if improvement found, else None.
    """
    source_n = len(source_trees)
    if source_n != target_n + k:
        return None, float('inf')
    
    best_score = baseline_score
    best_subset = None
    
    # Try all combinations of removing k trees
    for remove_indices in combinations(range(source_n), k):
        subset = [t for i, t in enumerate(source_trees) if i not in remove_indices]
        
        # Quick score check before expensive overlap check
        score = compute_score(subset, target_n)
        if score >= best_score - min_improvement:
            continue
        
        # Check overlaps only for promising candidates
        if check_overlap(subset):
            continue
        
        best_score = score
        best_subset = subset
    
    return best_subset, best_score

# Load baseline (exp_043 which has the improvements from k=1)
print("Loading baseline (exp_043)...")
baseline_path = "/home/code/experiments/043_subset_extraction/ensemble_043.csv"
baseline = load_baseline(baseline_path)
baseline_scores = {n: compute_score(baseline[n], n) for n in range(1, 201)}
total_baseline = sum(baseline_scores.values())
print(f"Baseline total: {total_baseline:.6f}")

# Extended extraction for k=2,3,4,5
print("\n" + "="*60)
print("EXTENDED SUBSET EXTRACTION (k=2,3,4,5)")
print("="*60)

all_improvements = {}
MIN_IMPROVEMENT = 0.0001

# k=2: Try for all N from 1 to 198
print("\n--- k=2 (remove 2 trees from N+2) ---")
start_time = time.time()
for target_n in range(1, 199):
    source_n = target_n + 2
    if source_n > 200:
        continue
    
    source_trees = baseline[source_n]
    subset, score = extract_n_from_n_plus_k(source_trees, target_n, 2, baseline_scores[target_n], MIN_IMPROVEMENT)
    
    if subset is not None and score < baseline_scores[target_n] - MIN_IMPROVEMENT:
        improvement = baseline_scores[target_n] - score
        if target_n not in all_improvements or improvement > all_improvements[target_n][1]:
            all_improvements[target_n] = (subset, improvement, source_n, 2)
            print(f"✅ N={target_n}: {baseline_scores[target_n]:.6f} -> {score:.6f} (improvement: {improvement:.6f}, from N={source_n}, k=2)")

print(f"k=2 completed in {time.time() - start_time:.1f}s")

# k=3: Try for N <= 100 (expensive for large N)
print("\n--- k=3 (remove 3 trees from N+3) ---")
start_time = time.time()
for target_n in range(1, 101):  # Limit to N<=100 due to computational cost
    source_n = target_n + 3
    if source_n > 200:
        continue
    
    source_trees = baseline[source_n]
    subset, score = extract_n_from_n_plus_k(source_trees, target_n, 3, baseline_scores[target_n], MIN_IMPROVEMENT)
    
    if subset is not None and score < baseline_scores[target_n] - MIN_IMPROVEMENT:
        improvement = baseline_scores[target_n] - score
        if target_n not in all_improvements or improvement > all_improvements[target_n][1]:
            all_improvements[target_n] = (subset, improvement, source_n, 3)
            print(f"✅ N={target_n}: {baseline_scores[target_n]:.6f} -> {score:.6f} (improvement: {improvement:.6f}, from N={source_n}, k=3)")

print(f"k=3 completed in {time.time() - start_time:.1f}s")

# k=4: Try for N <= 50 (very expensive)
print("\n--- k=4 (remove 4 trees from N+4) ---")
start_time = time.time()
for target_n in range(1, 51):  # Limit to N<=50
    source_n = target_n + 4
    if source_n > 200:
        continue
    
    source_trees = baseline[source_n]
    subset, score = extract_n_from_n_plus_k(source_trees, target_n, 4, baseline_scores[target_n], MIN_IMPROVEMENT)
    
    if subset is not None and score < baseline_scores[target_n] - MIN_IMPROVEMENT:
        improvement = baseline_scores[target_n] - score
        if target_n not in all_improvements or improvement > all_improvements[target_n][1]:
            all_improvements[target_n] = (subset, improvement, source_n, 4)
            print(f"✅ N={target_n}: {baseline_scores[target_n]:.6f} -> {score:.6f} (improvement: {improvement:.6f}, from N={source_n}, k=4)")

print(f"k=4 completed in {time.time() - start_time:.1f}s")

# k=5: Try for N <= 30 (extremely expensive)
print("\n--- k=5 (remove 5 trees from N+5) ---")
start_time = time.time()
for target_n in range(1, 31):  # Limit to N<=30
    source_n = target_n + 5
    if source_n > 200:
        continue
    
    source_trees = baseline[source_n]
    subset, score = extract_n_from_n_plus_k(source_trees, target_n, 5, baseline_scores[target_n], MIN_IMPROVEMENT)
    
    if subset is not None and score < baseline_scores[target_n] - MIN_IMPROVEMENT:
        improvement = baseline_scores[target_n] - score
        if target_n not in all_improvements or improvement > all_improvements[target_n][1]:
            all_improvements[target_n] = (subset, improvement, source_n, 5)
            print(f"✅ N={target_n}: {baseline_scores[target_n]:.6f} -> {score:.6f} (improvement: {improvement:.6f}, from N={source_n}, k=5)")

print(f"k=5 completed in {time.time() - start_time:.1f}s")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

if all_improvements:
    total_improvement = sum(imp[1] for imp in all_improvements.values())
    print(f"Total improvements found: {len(all_improvements)}")
    print(f"Total score improvement: {total_improvement:.6f}")
    
    print("\nAll improvements:")
    for n, (subset, imp, source_n, k) in sorted(all_improvements.items(), key=lambda x: -x[1][1]):
        print(f"  N={n}: improvement={imp:.6f} (from N={source_n}, k={k})")
    
    # Create ensemble with all improvements
    ensemble = {n: baseline[n] for n in range(1, 201)}
    for n, (subset, imp, source_n, k) in all_improvements.items():
        ensemble[n] = subset
    
    new_total = sum(compute_score(ensemble[n], n) for n in range(1, 201))
    print(f"\nBaseline total: {total_baseline:.6f}")
    print(f"New total: {new_total:.6f}")
    print(f"Actual improvement: {total_baseline - new_total:.6f}")
    
    # Save improvements dict for later use
    import json
    with open('improvements.json', 'w') as f:
        json.dump({str(n): {'improvement': imp, 'source_n': source_n, 'k': k} 
                   for n, (_, imp, source_n, k) in all_improvements.items()}, f, indent=2)
else:
    print("No improvements found from extended subset extraction")
