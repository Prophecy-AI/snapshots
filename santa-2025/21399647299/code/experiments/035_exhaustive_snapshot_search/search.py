"""
Exhaustive Snapshot Search with Strict Validation

Key insight: 62.5% of submissions failed due to overlaps.
This experiment uses VERY strict validation (tolerance=1e-12) and
falls back to exp_022 for any N value with potential overlap issues.
"""

import glob
import pandas as pd
from shapely import Polygon
import math
import numpy as np
import json
import os

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def get_tree_vertices(x, y, angle_deg):
    """Get the vertices of a tree at position (x, y) with rotation angle_deg."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    vertices = []
    for tx, ty in zip(TX, TY):
        rx = tx * cos_a - ty * sin_a
        ry = tx * sin_a + ty * cos_a
        vertices.append((rx + x, ry + y))
    
    return vertices

def get_tree_polygon(x, y, angle_deg):
    """Get Shapely polygon for a tree."""
    vertices = get_tree_vertices(x, y, angle_deg)
    return Polygon(vertices)

def compute_bbox(trees):
    """Compute bounding box size for all trees."""
    all_x = []
    all_y = []
    
    for x, y, angle in trees:
        vertices = get_tree_vertices(x, y, angle)
        for vx, vy in vertices:
            all_x.append(vx)
            all_y.append(vy)
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    return max(max_x - min_x, max_y - min_y)

def compute_score(trees, n):
    """Compute score for N trees."""
    bbox = compute_bbox(trees)
    return bbox ** 2 / n

def validate_strict(trees, tolerance=1e-12):
    """
    VERY strict overlap validation.
    Returns (is_valid, message)
    """
    n = len(trees)
    polygons = [get_tree_polygon(x, y, a) for x, y, a in trees]
    
    for i in range(n):
        for j in range(i + 1, n):
            if polygons[i].intersects(polygons[j]):
                if not polygons[i].touches(polygons[j]):
                    intersection = polygons[i].intersection(polygons[j])
                    if intersection.area > tolerance:
                        return False, f"Trees {i} and {j} overlap (area={intersection.area:.2e})"
    
    return True, "OK"

def parse_trees(df, n):
    """Parse trees for a specific N from a dataframe."""
    trees = []
    
    for _, row in df.iterrows():
        try:
            id_parts = str(row['id']).split('_')
            if len(id_parts) != 2:
                continue
            
            row_n = int(id_parts[0])
            if row_n != n:
                continue
            
            x = float(str(row['x']).replace('s', ''))
            y = float(str(row['y']).replace('s', ''))
            deg = float(str(row['deg']).replace('s', ''))
            
            # Check for NaN
            if math.isnan(x) or math.isnan(y) or math.isnan(deg):
                return None
            
            # Normalize angle
            deg = deg % 360.0
            if deg < 0:
                deg += 360.0
            
            trees.append((x, y, deg))
        except:
            continue
    
    if len(trees) != n:
        return None
    
    return trees

def load_baseline():
    """Load exp_022 as the safe baseline."""
    baseline_path = "/home/code/experiments/022_extended_cpp_optimization/optimized.csv"
    df = pd.read_csv(baseline_path)
    
    solutions = {}
    for n in range(1, 201):
        trees = parse_trees(df, n)
        if trees is not None:
            solutions[n] = trees
    
    return solutions

def main():
    print("=" * 60)
    print("EXHAUSTIVE SNAPSHOT SEARCH WITH STRICT VALIDATION")
    print("=" * 60)
    
    # Load baseline (exp_022)
    print("\nLoading exp_022 baseline...")
    baseline = load_baseline()
    
    # Compute baseline scores
    baseline_scores = {}
    total_baseline = 0
    for n in range(1, 201):
        if n in baseline:
            score = compute_score(baseline[n], n)
            baseline_scores[n] = score
            total_baseline += score
    
    print(f"Baseline total score: {total_baseline:.6f}")
    
    # Find all snapshot CSV files
    snapshot_dir = "/home/nonroot/snapshots/santa-2025"
    snapshot_files = []
    
    for root, dirs, files in os.walk(snapshot_dir):
        for f in files:
            if f.endswith('.csv'):
                snapshot_files.append(os.path.join(root, f))
    
    print(f"\nFound {len(snapshot_files)} snapshot files")
    
    # Also check external data
    external_dir = "/home/code/data/external"
    for root, dirs, files in os.walk(external_dir):
        for f in files:
            if f.endswith('.csv'):
                snapshot_files.append(os.path.join(root, f))
    
    print(f"Total files to scan: {len(snapshot_files)}")
    
    # Track best per-N solutions
    best_per_n = {}
    for n in range(1, 201):
        if n in baseline:
            best_per_n[n] = {
                'score': baseline_scores[n],
                'trees': baseline[n],
                'source': 'baseline'
            }
    
    # Scan all files
    print("\nScanning files...")
    files_processed = 0
    improvements_found = 0
    
    for filepath in snapshot_files:
        try:
            df = pd.read_csv(filepath)
            
            if 'id' not in df.columns:
                continue
            
            files_processed += 1
            if files_processed % 500 == 0:
                print(f"  Processed {files_processed} files, found {improvements_found} improvements...")
            
            for n in range(1, 201):
                trees = parse_trees(df, n)
                if trees is None:
                    continue
                
                # Compute score
                score = compute_score(trees, n)
                
                # Only consider if significantly better
                if n not in best_per_n or score < best_per_n[n]['score'] - 0.0001:
                    # STRICT validation
                    is_valid, msg = validate_strict(trees, tolerance=1e-12)
                    
                    if is_valid:
                        best_per_n[n] = {
                            'score': score,
                            'trees': trees,
                            'source': filepath
                        }
                        improvements_found += 1
                        print(f"  N={n}: Found better solution {score:.6f} from {os.path.basename(filepath)}")
        except Exception as e:
            continue
    
    print(f"\nProcessed {files_processed} files")
    print(f"Found {improvements_found} improvements")
    
    # Create final solution
    print("\n" + "=" * 60)
    print("CREATING FINAL SOLUTION")
    print("=" * 60)
    
    final_solution = {}
    total_final = 0
    improvements_applied = 0
    
    for n in range(1, 201):
        if n not in best_per_n:
            continue
        
        trees = best_per_n[n]['trees']
        source = best_per_n[n]['source']
        
        # Double-check validation
        is_valid, msg = validate_strict(trees, tolerance=1e-12)
        
        if is_valid:
            final_solution[n] = trees
            score = compute_score(trees, n)
            total_final += score
            
            if source != 'baseline':
                improvements_applied += 1
        else:
            # Fall back to baseline
            print(f"  N={n}: Falling back to baseline ({msg})")
            final_solution[n] = baseline[n]
            total_final += baseline_scores[n]
    
    print(f"\nFinal total score: {total_final:.6f}")
    print(f"Baseline score: {total_baseline:.6f}")
    print(f"Improvement: {total_baseline - total_final:.6f}")
    print(f"Improvements applied: {improvements_applied}")
    
    # Save submission
    print("\n" + "=" * 60)
    print("SAVING SUBMISSION")
    print("=" * 60)
    
    rows = []
    for n in range(1, 201):
        if n not in final_solution:
            continue
        
        trees = final_solution[n]
        for i, (x, y, deg) in enumerate(trees):
            rows.append({
                'id': f"{n:03d}_{i}",
                'x': f"s{x:.20f}",
                'y': f"s{y:.20f}",
                'deg': f"s{deg:.20f}"
            })
    
    df = pd.DataFrame(rows)
    
    # Save to experiment folder
    output_path = "/home/code/experiments/035_exhaustive_snapshot_search/submission.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    # Save to submission folder
    submission_path = "/home/submission/submission.csv"
    df.to_csv(submission_path, index=False)
    print(f"Saved to {submission_path}")
    
    # Save metrics
    metrics = {
        'cv_score': total_final,
        'baseline_score': total_baseline,
        'improvement': total_baseline - total_final,
        'improvements_applied': improvements_applied,
        'files_processed': files_processed
    }
    
    with open('/home/code/experiments/035_exhaustive_snapshot_search/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nMetrics saved")
    
    return total_final, improvements_applied

if __name__ == "__main__":
    total_final, improvements_applied = main()
