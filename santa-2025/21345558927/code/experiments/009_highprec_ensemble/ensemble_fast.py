"""
High-precision ensemble with strict overlap validation - FAST VERSION.
Only accepts improvements that are significant AND pass strict overlap validation.
"""

import sys
import os
sys.path.insert(0, '/home/code')

import numpy as np
import pandas as pd
import json
import glob
from decimal import Decimal, getcontext
from shapely.geometry import Polygon
import time

from code.tree_geometry import TX, TY, calculate_score, get_tree_vertices_numba
from code.utils import parse_submission, save_submission

# Set high precision for Decimal operations
getcontext().prec = 50
SCALE = 10**18

# Minimum improvement threshold - ignore improvements smaller than this
MIN_IMPROVEMENT = 1e-5  # 0.00001 - ignore floating-point noise

def get_tree_polygon_highprec(x, y, angle_deg):
    """Get tree polygon with high-precision integer coordinates."""
    rx, ry = get_tree_vertices_numba(x, y, angle_deg)
    
    # Scale to integers for exact arithmetic
    coords = []
    for xi, yi in zip(rx, ry):
        xi_int = int(Decimal(str(xi)) * SCALE)
        yi_int = int(Decimal(str(yi)) * SCALE)
        coords.append((xi_int, yi_int))
    
    return Polygon(coords)

def validate_no_overlap_strict(trees):
    """Validate no overlaps using integer arithmetic for precision."""
    if len(trees) <= 1:
        return True, "OK"
    
    polygons = []
    for x, y, angle in trees:
        poly = get_tree_polygon_highprec(x, y, angle)
        if not poly.is_valid:
            return False, "Invalid polygon"
        polygons.append(poly)
    
    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            if polygons[i].intersects(polygons[j]):
                if not polygons[i].touches(polygons[j]):
                    inter = polygons[i].intersection(polygons[j])
                    if inter.area > 0:  # Any overlap at integer scale = real overlap
                        return False, f"Trees {i} and {j} overlap (area={inter.area / (SCALE**2):.2e})"
    return True, "OK"

def check_no_nan(trees):
    """Check for NaN values in tree configuration."""
    for i, (x, y, angle) in enumerate(trees):
        if np.isnan(x) or np.isnan(y) or np.isnan(angle):
            return False, f"NaN value in tree {i}"
    return True, "OK"

def load_submission_fast(csv_file):
    """Load submission without NaN check (faster)."""
    try:
        df = pd.read_csv(csv_file)
        if 'id' not in df.columns or 'x' not in df.columns or len(df) != 20100:
            return None
        return parse_submission(df)
    except:
        return None

def main():
    print("=" * 70)
    print("HIGH-PRECISION ENSEMBLE WITH STRICT VALIDATION (FAST)")
    print("=" * 70)
    
    start_time = time.time()
    
    # Load baseline
    baseline_df = pd.read_csv('/home/code/experiments/001_valid_baseline/submission.csv')
    baseline_configs = parse_submission(baseline_df)
    
    # Calculate baseline scores per N
    baseline_scores = {}
    for n in range(1, 201):
        baseline_scores[n] = calculate_score(baseline_configs[n])
    
    baseline_total = sum(baseline_scores.values())
    print(f"Baseline total score: {baseline_total:.6f}")
    
    # Load all valid submissions from snapshots
    print("\nLoading submissions from snapshots...")
    snapshot_dirs = glob.glob('/home/nonroot/snapshots/santa-2025/*')
    print(f"Found {len(snapshot_dirs)} snapshot directories")
    
    all_solutions = {}
    loaded = 0
    for snap_dir in snapshot_dirs:
        csv_files = glob.glob(f'{snap_dir}/**/*.csv', recursive=True)
        for csv_file in csv_files:
            configs = load_submission_fast(csv_file)
            if configs is not None:
                all_solutions[csv_file] = configs
                loaded += 1
                if loaded % 500 == 0:
                    print(f"  Loaded {loaded} submissions...")
    
    print(f"Loaded {len(all_solutions)} valid submissions in {time.time() - start_time:.1f}s")
    
    # Find best per-N solutions with strict validation
    print("\nFinding best per-N solutions with strict validation...")
    print(f"Minimum improvement threshold: {MIN_IMPROVEMENT}")
    
    best_per_n = {}
    best_source_per_n = {}
    improvements = []
    rejected_overlap = []
    rejected_small = []
    rejected_nan = []
    
    for n in range(1, 201):
        best_score = baseline_scores[n]
        best_config = baseline_configs[n]
        best_source = "baseline"
        
        for source, configs in all_solutions.items():
            if n not in configs:
                continue
            
            config = configs[n]
            
            # Skip if wrong number of trees
            if len(config) != n:
                continue
            
            # Check for NaN values
            valid, msg = check_no_nan(config)
            if not valid:
                rejected_nan.append((n, source.split('/')[-1]))
                continue
            
            # Calculate score
            score = calculate_score(config)
            improvement = best_score - score
            
            # Only consider if improvement is significant
            if improvement < MIN_IMPROVEMENT:
                if improvement > 0:
                    rejected_small.append((n, improvement, source.split('/')[-1]))
                continue
            
            # Strict overlap validation
            valid, msg = validate_no_overlap_strict(config)
            if not valid:
                rejected_overlap.append((n, improvement, source.split('/')[-1], msg))
                continue
            
            # Accept this improvement
            best_score = score
            best_config = config
            best_source = source
        
        best_per_n[n] = best_config
        best_source_per_n[n] = best_source
        
        if best_source != "baseline":
            improvement = baseline_scores[n] - best_score
            improvements.append((n, improvement, best_source.split('/')[-1]))
            print(f"N={n}: Improved by {improvement:.9f} from {best_source.split('/')[-1]}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    ensemble_total = sum(calculate_score(best_per_n[n]) for n in range(1, 201))
    total_improvement = baseline_total - ensemble_total
    
    print(f"Baseline total: {baseline_total:.6f}")
    print(f"Ensemble total: {ensemble_total:.6f}")
    print(f"Total improvement: {total_improvement:.6f}")
    print(f"\nN values improved: {len(improvements)} / 200")
    print(f"Rejected (too small): {len(rejected_small)}")
    print(f"Rejected (overlap): {len(rejected_overlap)}")
    print(f"Rejected (NaN): {len(rejected_nan)}")
    
    if rejected_small:
        print(f"\nTop rejected (too small):")
        for n, imp, src in sorted(rejected_small, key=lambda x: -x[1])[:10]:
            print(f"  N={n}: {imp:.9f} from {src}")
    
    if rejected_overlap:
        print(f"\nRejected (overlap):")
        for n, imp, src, msg in rejected_overlap[:10]:
            print(f"  N={n}: {imp:.9f} from {src} - {msg}")
    
    # Final validation of entire submission
    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)
    
    invalid_n = []
    for n in range(1, 201):
        config = best_per_n[n]
        
        # Check number of trees
        if len(config) != n:
            print(f"N={n}: Wrong number of trees: {len(config)}")
            invalid_n.append(n)
            continue
        
        # Check for NaN values
        valid, msg = check_no_nan(config)
        if not valid:
            print(f"N={n}: {msg}")
            invalid_n.append(n)
            continue
        
        # Check overlaps
        valid, msg = validate_no_overlap_strict(config)
        if not valid:
            print(f"N={n}: {msg} - falling back to baseline")
            best_per_n[n] = baseline_configs[n]
            invalid_n.append(n)
    
    if not invalid_n:
        print("✅ All configurations valid!")
    else:
        print(f"\n⚠️ Fell back to baseline for {len(invalid_n)} N values: {invalid_n}")
        # Recalculate total
        ensemble_total = sum(calculate_score(best_per_n[n]) for n in range(1, 201))
        total_improvement = baseline_total - ensemble_total
        print(f"Updated ensemble total: {ensemble_total:.6f}")
        print(f"Updated improvement: {total_improvement:.6f}")
    
    # Save submission
    save_submission(best_per_n, 'submission.csv')
    print("\nSaved submission.csv")
    
    # Validate submission format
    df = pd.read_csv('submission.csv')
    print(f"Rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for NaN values in saved file
    for col in ['x', 'y', 'deg']:
        vals = df[col].astype(str).str.replace('s', '', regex=False)
        nan_count = vals.apply(lambda x: 'nan' in x.lower()).sum()
        if nan_count > 0:
            print(f"❌ WARNING: {nan_count} NaN values in {col}")
        else:
            print(f"✅ No NaN values in {col}")
    
    # Save metrics
    metrics = {
        'cv_score': ensemble_total,
        'baseline_score': baseline_total,
        'improvement': total_improvement,
        'num_improvements': len(improvements),
        'min_improvement_threshold': MIN_IMPROVEMENT,
        'rejected_small': len(rejected_small),
        'rejected_overlap': len(rejected_overlap),
        'rejected_nan': len(rejected_nan),
        'notes': 'High-precision ensemble with strict overlap validation'
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nFinal CV Score: {ensemble_total:.6f}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    
    # Copy to submission folder
    import shutil
    shutil.copy('submission.csv', '/home/submission/submission.csv')
    print("Copied submission to /home/submission/")
    
    return ensemble_total

if __name__ == '__main__':
    os.chdir('/home/code/experiments/009_highprec_ensemble')
    main()
