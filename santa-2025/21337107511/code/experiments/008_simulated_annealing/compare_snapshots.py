"""
Compare all snapshots to find the best solution for each N value.

The key insight: different optimizers find different local optima.
By combining the best from each, we might beat the baseline.
"""

import numpy as np
import pandas as pd
import math
import os
from shapely import Polygon
from shapely.affinity import rotate, translate
from decimal import Decimal, getcontext
import time
import json

getcontext().prec = 30

# Tree polygon vertices
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]
TREE_COORDS = list(zip(TX, TY))

# Bad snapshots to exclude (known format issues)
BAD_SNAPSHOTS = {'21145963314', '21337107511', '21145965159', '21336527339'}

def get_tree_polygon(x, y, angle):
    poly = Polygon(TREE_COORDS)
    poly = rotate(poly, angle, origin=(0, 0))
    poly = translate(poly, x, y)
    return poly

def calculate_bounding_box(trees):
    all_coords = []
    for x, y, angle in trees:
        poly = get_tree_polygon(x, y, angle)
        all_coords.extend(poly.exterior.coords)
    
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    
    return max(max(xs) - min(xs), max(ys) - min(ys))

def has_overlap(trees):
    """Check if any trees overlap."""
    n = len(trees)
    if n <= 1:
        return False
    
    polygons = [get_tree_polygon(x, y, angle) for x, y, angle in trees]
    
    for i in range(n):
        for j in range(i + 1, n):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > 1e-15:
                    return True
    return False

def load_snapshot(snapshot_path):
    """Load a snapshot and return solutions by N."""
    try:
        df = pd.read_csv(snapshot_path)
        
        # Check format
        if 'id' not in df.columns:
            return None
        
        # Check for bad format (wrong ID format or missing 's' prefix)
        sample_id = str(df['id'].iloc[0])
        if '_' in sample_id:
            parts = sample_id.split('_')
            if len(parts[1]) > 2:  # Wrong format like '013_000'
                return None
        
        sample_x = str(df['x'].iloc[0])
        if not sample_x.startswith('s'):
            return None
        
        solutions = {}
        for n in range(1, 201):
            n_df = df[df['id'].str.startswith(f'{n:03d}_')]
            if len(n_df) != n:
                continue
            
            trees = []
            for _, row in n_df.iterrows():
                x = float(str(row['x']).replace('s', ''))
                y = float(str(row['y']).replace('s', ''))
                angle = float(str(row['deg']).replace('s', ''))
                trees.append([x, y, angle])
            solutions[n] = trees
        
        return solutions
    except Exception as e:
        return None

def compare_all_snapshots():
    """Compare all snapshots and find best per-N solutions."""
    snapshots_dir = "/home/nonroot/snapshots/santa-2025/"
    snapshot_ids = [d for d in os.listdir(snapshots_dir) if d not in BAD_SNAPSHOTS]
    
    print(f"Found {len(snapshot_ids)} snapshots (excluding {len(BAD_SNAPSHOTS)} bad ones)")
    
    # Load baseline
    baseline_path = os.path.join(snapshots_dir, "21337353543/submission/submission.csv")
    baseline_solutions = load_snapshot(baseline_path)
    
    if baseline_solutions is None:
        print("ERROR: Could not load baseline!")
        return
    
    baseline_scores = {}
    for n in range(1, 201):
        if n in baseline_solutions:
            side = calculate_bounding_box(baseline_solutions[n])
            baseline_scores[n] = (side ** 2) / n
    
    baseline_total = sum(baseline_scores.values())
    print(f"Baseline total score: {baseline_total:.6f}")
    
    # Track best per-N
    best_per_n = {n: {'score': baseline_scores[n], 'source': '21337353543', 'trees': baseline_solutions[n]} 
                  for n in range(1, 201)}
    
    # Check each snapshot
    improvements_found = 0
    
    for i, snapshot_id in enumerate(snapshot_ids):
        if snapshot_id == '21337353543':
            continue
        
        snapshot_path = os.path.join(snapshots_dir, snapshot_id, "submission/submission.csv")
        if not os.path.exists(snapshot_path):
            continue
        
        solutions = load_snapshot(snapshot_path)
        if solutions is None:
            continue
        
        for n in range(1, 201):
            if n not in solutions:
                continue
            
            trees = solutions[n]
            
            # Check for overlaps
            if has_overlap(trees):
                continue
            
            side = calculate_bounding_box(trees)
            score = (side ** 2) / n
            
            if score < best_per_n[n]['score'] - 1e-10:  # Significant improvement
                improvement = best_per_n[n]['score'] - score
                print(f"N={n}: Found better in {snapshot_id}: {score:.8f} vs {best_per_n[n]['score']:.8f} (improvement: {improvement:.8f})")
                best_per_n[n] = {'score': score, 'source': snapshot_id, 'trees': trees}
                improvements_found += 1
        
        if (i + 1) % 20 == 0:
            print(f"Checked {i + 1}/{len(snapshot_ids)} snapshots, {improvements_found} improvements found")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    new_total = sum(best_per_n[n]['score'] for n in range(1, 201))
    total_improvement = baseline_total - new_total
    
    print(f"Baseline total: {baseline_total:.6f}")
    print(f"Best ensemble total: {new_total:.6f}")
    print(f"Total improvement: {total_improvement:.6f}")
    print(f"Improvements found: {improvements_found}")
    
    # Show which N values improved
    improved_ns = [n for n in range(1, 201) if best_per_n[n]['source'] != '21337353543']
    print(f"N values improved: {improved_ns}")
    
    return best_per_n, new_total

if __name__ == "__main__":
    best_per_n, new_total = compare_all_snapshots()
    
    # Save results
    results = {
        'total_score': new_total,
        'improvements': {n: {'score': best_per_n[n]['score'], 'source': best_per_n[n]['source']} 
                        for n in range(1, 201) if best_per_n[n]['source'] != '21337353543'}
    }
    
    with open('/home/code/experiments/008_simulated_annealing/snapshot_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to snapshot_comparison.json")
