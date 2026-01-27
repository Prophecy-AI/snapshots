"""
Find better solutions for small N values (N=1-20) that are still at baseline.
Use a lower MIN_IMPROVEMENT threshold but with very strict validation.
"""

import sys
import os
sys.path.insert(0, '/home/code')

import numpy as np
import pandas as pd
import json
import glob
import time
from decimal import Decimal, getcontext
from shapely.geometry import Polygon

from code.tree_geometry import TX, TY, calculate_score, get_tree_vertices_numba
from code.utils import parse_submission, save_submission

getcontext().prec = 50
SCALE = 10**18

# Lower threshold for small N values (they have higher per-N scores)
MIN_IMPROVEMENT = 0.0005

def get_tree_polygon_highprec(x, y, angle_deg):
    """Get tree polygon with high-precision integer coordinates."""
    rx, ry = get_tree_vertices_numba(x, y, angle_deg)
    coords = []
    for xi, yi in zip(rx, ry):
        xi_int = int(Decimal(str(xi)) * SCALE)
        yi_int = int(Decimal(str(yi)) * SCALE)
        coords.append((xi_int, yi_int))
    return Polygon(coords)

def validate_no_overlap_strict(trees):
    """Validate no overlaps using integer arithmetic."""
    if len(trees) <= 1:
        return True
    
    polygons = []
    for x, y, angle in trees:
        poly = get_tree_polygon_highprec(x, y, angle)
        if not poly.is_valid:
            return False
        polygons.append(poly)
    
    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            if polygons[i].intersects(polygons[j]):
                if not polygons[i].touches(polygons[j]):
                    inter = polygons[i].intersection(polygons[j])
                    if inter.area > 0:
                        return False
    return True

def check_no_nan(trees):
    """Check for NaN values."""
    for x, y, angle in trees:
        if np.isnan(x) or np.isnan(y) or np.isnan(angle):
            return False
    return True

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
    print("FIND BETTER SOLUTIONS FOR SMALL N VALUES")
    print("=" * 70)
    
    start_time = time.time()
    
    # Load baseline and exp_010
    baseline_df = pd.read_csv('/home/code/experiments/001_valid_baseline/submission.csv')
    baseline_configs = parse_submission(baseline_df)
    
    exp010_df = pd.read_csv('/home/code/experiments/010_safe_ensemble/submission.csv')
    exp010_configs = parse_submission(exp010_df)
    
    # Calculate baseline scores
    baseline_scores = {n: calculate_score(baseline_configs[n]) for n in range(1, 201)}
    
    # Target N values (small N that are still at baseline)
    target_n = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 23]
    print(f"\nTarget N values: {target_n}")
    print(f"MIN_IMPROVEMENT threshold: {MIN_IMPROVEMENT}")
    
    # Load all submissions from snapshots
    print("\nLoading submissions from snapshots...")
    snapshot_dirs = glob.glob('/home/nonroot/snapshots/santa-2025/*')
    
    all_solutions = {}
    for snap_dir in snapshot_dirs:
        csv_files = glob.glob(f'{snap_dir}/**/*.csv', recursive=True)
        for csv_file in csv_files:
            configs = load_submission_fast(csv_file)
            if configs is not None:
                all_solutions[csv_file] = configs
    
    print(f"Loaded {len(all_solutions)} submissions")
    
    # Find better solutions for target N values
    print("\nSearching for better solutions...")
    
    best_per_n = {}
    improvements = []
    
    for n in target_n:
        best_score = baseline_scores[n]
        best_config = baseline_configs[n]
        best_source = "baseline"
        
        for source, configs in all_solutions.items():
            if n not in configs:
                continue
            
            config = configs[n]
            
            if len(config) != n:
                continue
            
            if not check_no_nan(config):
                continue
            
            score = calculate_score(config)
            improvement = best_score - score
            
            if improvement < MIN_IMPROVEMENT:
                continue
            
            # Strict overlap validation
            if not validate_no_overlap_strict(config):
                continue
            
            best_score = score
            best_config = config
            best_source = source
        
        best_per_n[n] = best_config
        
        if best_source != "baseline":
            improvement = baseline_scores[n] - best_score
            improvements.append((n, improvement, best_source.split('/')[-1]))
            print(f"N={n}: Improved by {improvement:.6f} from {best_source.split('/')[-1]}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_improvement = sum(imp for _, imp, _ in improvements)
    print(f"Total improvement from small N: {total_improvement:.6f}")
    print(f"N values improved: {len(improvements)}")
    
    if improvements:
        print("\nImprovements found:")
        for n, imp, src in sorted(improvements, key=lambda x: -x[1]):
            print(f"  N={n}: {imp:.6f} from {src}")
    
    # Create final submission by combining exp_010 with new improvements
    print("\nCreating final submission...")
    
    final_configs = {}
    for n in range(1, 201):
        if n in best_per_n and n in [x[0] for x in improvements]:
            final_configs[n] = best_per_n[n]
        else:
            final_configs[n] = exp010_configs[n]
    
    # Final validation
    print("\nFinal validation...")
    invalid_n = []
    for n in range(1, 201):
        config = final_configs[n]
        if len(config) != n:
            print(f"N={n}: Wrong number of trees")
            invalid_n.append(n)
            continue
        if not validate_no_overlap_strict(config):
            print(f"N={n}: Overlap detected - falling back to exp_010")
            final_configs[n] = exp010_configs[n]
            invalid_n.append(n)
    
    if not invalid_n:
        print("✅ All configurations valid!")
    else:
        print(f"⚠️ Fell back for {len(invalid_n)} N values")
    
    # Calculate final score
    final_total = sum(calculate_score(final_configs[n]) for n in range(1, 201))
    exp010_total = sum(calculate_score(exp010_configs[n]) for n in range(1, 201))
    
    print(f"\nexp_010 total: {exp010_total:.6f}")
    print(f"Final total: {final_total:.6f}")
    print(f"Improvement: {exp010_total - final_total:.6f}")
    
    # Save submission
    save_submission(final_configs, 'submission.csv')
    print("\nSaved submission.csv")
    
    # Save metrics
    metrics = {
        'cv_score': final_total,
        'exp010_score': exp010_total,
        'improvement': exp010_total - final_total,
        'num_improvements': len(improvements),
        'min_improvement_threshold': MIN_IMPROVEMENT,
        'target_n': target_n,
        'notes': 'Better solutions for small N values'
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nFinal CV Score: {final_total:.6f}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    
    # Copy to submission folder
    import shutil
    shutil.copy('submission.csv', '/home/submission/submission.csv')
    print("Copied submission to /home/submission/")
    
    return final_total

if __name__ == '__main__':
    os.chdir('/home/code/experiments/011_fractional_translation')
    main()
