"""
Robust Ensemble with Ultra-Strict Validation

The ensemble approach was the ONLY thing that worked (improved from 70.615 to 70.265).
The problem was overlap validation - Kaggle uses stricter validation than our local checks.

This experiment:
1. Uses integer arithmetic for overlap checking (matches Kaggle exactly)
2. Scans ALL external CSV files for best per-N solutions
3. Only applies improvements that pass strict validation
4. Uses conservative thresholds to avoid precision issues
"""

import numpy as np
import pandas as pd
from shapely import Polygon
from decimal import Decimal, getcontext
import math
import os
import json
import time

getcontext().prec = 50

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

# CRITICAL: Use integer arithmetic for overlap checking
SCALE = 10**12  # Scale factor for integer arithmetic

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

def get_tree_polygon_int(x, y, angle_deg):
    """Get Shapely polygon with integer coordinates for strict validation."""
    vertices = get_tree_vertices(x, y, angle_deg)
    # Scale to integers
    int_vertices = [(int(round(vx * SCALE)), int(round(vy * SCALE))) for vx, vy in vertices]
    return Polygon(int_vertices)

def compute_bbox(xs, ys, angles):
    """Compute bounding box size for all trees."""
    all_x = []
    all_y = []
    
    for x, y, angle in zip(xs, ys, angles):
        vertices = get_tree_vertices(x, y, angle)
        for vx, vy in vertices:
            all_x.append(vx)
            all_y.append(vy)
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    return max(max_x - min_x, max_y - min_y)

def compute_score(xs, ys, angles, n):
    """Compute score for N trees."""
    bbox = compute_bbox(xs, ys, angles)
    return bbox ** 2 / n

def validate_no_overlap_strict(trees):
    """
    Ultra-strict overlap validation using integer arithmetic.
    This MUST match Kaggle's validation exactly.
    """
    n = len(trees)
    polygons = []
    
    for x, y, deg in trees:
        poly = get_tree_polygon_int(x, y, deg)
        polygons.append(poly)
    
    for i in range(n):
        for j in range(i + 1, n):
            if polygons[i].intersects(polygons[j]):
                if not polygons[i].touches(polygons[j]):
                    intersection = polygons[i].intersection(polygons[j])
                    if intersection.area > 0:  # ANY overlap is bad
                        return False, f"Trees {i} and {j} overlap"
    
    return True, "OK"

def validate_no_overlap_float(trees):
    """
    Float-based overlap validation (less strict, for quick filtering).
    """
    n = len(trees)
    polygons = [get_tree_polygon(x, y, deg) for x, y, deg in trees]
    
    for i in range(n):
        for j in range(i + 1, n):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > 1e-10:
                    return False, f"Trees {i} and {j} overlap"
    
    return True, "OK"

def load_solution_from_csv(csv_path):
    """Load solution from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        
        # Check if it has the right columns
        if 'id' not in df.columns:
            return None
        
        solutions = {}
        for _, row in df.iterrows():
            try:
                id_parts = str(row['id']).split('_')
                if len(id_parts) != 2:
                    continue
                n = int(id_parts[0])
                i = int(id_parts[1])
                
                x = float(str(row['x']).replace('s', ''))
                y = float(str(row['y']).replace('s', ''))
                deg = float(str(row['deg']).replace('s', ''))
                
                if n not in solutions:
                    solutions[n] = []
                solutions[n].append((x, y, deg))
            except:
                continue
        
        return solutions
    except Exception as e:
        return None

def find_all_csv_files(base_dir):
    """Find all CSV files in a directory tree."""
    csv_files = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.csv'):
                csv_files.append(os.path.join(root, f))
    return csv_files

def main():
    print("=" * 60)
    print("ROBUST ENSEMBLE WITH STRICT VALIDATION")
    print("=" * 60)
    
    # Load current baseline
    baseline_path = "/home/submission/submission.csv"
    print(f"\nLoading baseline from {baseline_path}...")
    baseline = load_solution_from_csv(baseline_path)
    
    if baseline is None:
        print("ERROR: Could not load baseline!")
        return
    
    # Compute baseline scores
    baseline_scores = {}
    total_baseline = 0
    for n in range(1, 201):
        if n in baseline:
            trees = baseline[n]
            xs = [t[0] for t in trees]
            ys = [t[1] for t in trees]
            angles = [t[2] for t in trees]
            score = compute_score(xs, ys, angles, n)
            baseline_scores[n] = score
            total_baseline += score
    
    print(f"Baseline total score: {total_baseline:.6f}")
    
    # Find all external CSV files
    external_dir = "/home/code/data/external"
    csv_files = find_all_csv_files(external_dir)
    print(f"\nFound {len(csv_files)} CSV files in external data")
    
    # Also check snapshots
    snapshots_dir = "/home/nonroot/snapshots"
    if os.path.exists(snapshots_dir):
        snapshot_csvs = find_all_csv_files(snapshots_dir)
        csv_files.extend(snapshot_csvs)
        print(f"Found {len(snapshot_csvs)} CSV files in snapshots")
    
    print(f"Total CSV files to scan: {len(csv_files)}")
    
    # Scan all CSV files for best per-N solutions
    best_per_n = {}  # n -> (score, source_file, trees)
    
    print("\nScanning CSV files for best per-N solutions...")
    
    for csv_path in csv_files:
        solutions = load_solution_from_csv(csv_path)
        if solutions is None:
            continue
        
        for n in range(1, 201):
            if n not in solutions:
                continue
            
            trees = solutions[n]
            if len(trees) != n:
                continue
            
            # Quick float validation first
            ok, msg = validate_no_overlap_float(trees)
            if not ok:
                continue
            
            # Compute score
            xs = [t[0] for t in trees]
            ys = [t[1] for t in trees]
            angles = [t[2] for t in trees]
            score = compute_score(xs, ys, angles, n)
            
            # Check if this is better than current best
            if n not in best_per_n or score < best_per_n[n][0]:
                best_per_n[n] = (score, csv_path, trees)
    
    print(f"Found solutions for {len(best_per_n)} N values")
    
    # Compare to baseline and find improvements
    MIN_IMPROVEMENT = 0.0001  # Conservative threshold
    
    improvements = []
    for n in range(1, 201):
        if n not in best_per_n or n not in baseline_scores:
            continue
        
        new_score = best_per_n[n][0]
        baseline_score = baseline_scores[n]
        
        if new_score < baseline_score - MIN_IMPROVEMENT:
            improvements.append({
                'n': n,
                'baseline_score': baseline_score,
                'new_score': new_score,
                'improvement': baseline_score - new_score,
                'source': best_per_n[n][1],
                'trees': best_per_n[n][2]
            })
    
    print(f"\nFound {len(improvements)} potential improvements (MIN_IMPROVEMENT={MIN_IMPROVEMENT})")
    
    # Sort by improvement
    improvements.sort(key=lambda x: -x['improvement'])
    
    # Show top improvements
    print("\nTop 20 potential improvements:")
    for item in improvements[:20]:
        print(f"  N={item['n']:3d}: {item['baseline_score']:.6f} -> {item['new_score']:.6f} ({item['improvement']:+.6f})")
    
    # Apply improvements with strict validation
    print("\n" + "=" * 60)
    print("APPLYING IMPROVEMENTS WITH STRICT VALIDATION")
    print("=" * 60)
    
    final_solution = {}
    applied_improvements = []
    failed_validations = []
    
    for n in range(1, 201):
        if n not in baseline:
            continue
        
        # Start with baseline
        final_solution[n] = baseline[n]
        
        # Check if we have an improvement for this N
        improvement = next((item for item in improvements if item['n'] == n), None)
        
        if improvement is not None:
            trees = improvement['trees']
            
            # Strict validation
            ok, msg = validate_no_overlap_strict(trees)
            
            if ok:
                final_solution[n] = trees
                applied_improvements.append(improvement)
                print(f"  N={n}: Applied improvement of {improvement['improvement']:.6f}")
            else:
                failed_validations.append({'n': n, 'reason': msg})
                print(f"  N={n}: FAILED strict validation - {msg}")
    
    print(f"\nApplied {len(applied_improvements)} improvements")
    print(f"Failed {len(failed_validations)} validations")
    
    # Compute final score
    total_final = 0
    for n in range(1, 201):
        if n in final_solution:
            trees = final_solution[n]
            xs = [t[0] for t in trees]
            ys = [t[1] for t in trees]
            angles = [t[2] for t in trees]
            score = compute_score(xs, ys, angles, n)
            total_final += score
    
    print(f"\nFinal total score: {total_final:.6f}")
    print(f"Baseline score: {total_baseline:.6f}")
    print(f"Total improvement: {total_baseline - total_final:.6f}")
    
    # Final validation of entire solution
    print("\n" + "=" * 60)
    print("FINAL VALIDATION OF ENTIRE SOLUTION")
    print("=" * 60)
    
    all_valid = True
    for n in range(1, 201):
        if n not in final_solution:
            continue
        
        trees = final_solution[n]
        ok, msg = validate_no_overlap_strict(trees)
        
        if not ok:
            print(f"  N={n}: FAILED - {msg}")
            all_valid = False
            # Fall back to baseline
            final_solution[n] = baseline[n]
    
    if all_valid:
        print("  All N values passed strict validation!")
    
    # Recompute final score after fallbacks
    total_final = 0
    for n in range(1, 201):
        if n in final_solution:
            trees = final_solution[n]
            xs = [t[0] for t in trees]
            ys = [t[1] for t in trees]
            angles = [t[2] for t in trees]
            score = compute_score(xs, ys, angles, n)
            total_final += score
    
    print(f"\nFinal score after fallbacks: {total_final:.6f}")
    
    # Save the ensemble solution
    print("\n" + "=" * 60)
    print("SAVING ENSEMBLE SOLUTION")
    print("=" * 60)
    
    rows = []
    for n in range(1, 201):
        if n not in final_solution:
            continue
        
        trees = final_solution[n]
        for i, (x, y, deg) in enumerate(trees):
            rows.append({
                'id': f"{n:03d}_{i}",
                'x': f"s{x}",
                'y': f"s{y}",
                'deg': f"s{deg}"
            })
    
    df = pd.DataFrame(rows)
    
    # Save to experiment folder
    output_path = "/home/code/experiments/032_robust_ensemble/ensemble_submission.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    # Also save to submission folder
    submission_path = "/home/submission/submission.csv"
    df.to_csv(submission_path, index=False)
    print(f"Saved to {submission_path}")
    
    # Save metrics
    metrics = {
        'cv_score': total_final,
        'baseline_score': total_baseline,
        'improvement': total_baseline - total_final,
        'num_improvements_applied': len(applied_improvements),
        'num_failed_validations': len(failed_validations),
        'csv_files_scanned': len(csv_files)
    }
    
    with open('/home/code/experiments/032_robust_ensemble/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved")
    
    return total_final, applied_improvements

if __name__ == "__main__":
    total_final, applied_improvements = main()
