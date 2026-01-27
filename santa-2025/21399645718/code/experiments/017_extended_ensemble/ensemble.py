"""
Extended Ensemble - Combine ALL available sources
Uses exp_016 as baseline and adds any improvements >= 0.001
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

getcontext().prec = 30
SCALE = 10**18

# CRITICAL: Keep MIN_IMPROVEMENT=0.001 - smaller values fail Kaggle validation
MIN_IMPROVEMENT = 0.001

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
    """Load submission."""
    try:
        df = pd.read_csv(csv_file)
        if 'id' not in df.columns or 'x' not in df.columns or len(df) != 20100:
            return None
        return parse_submission(df)
    except:
        return None

def main():
    print("=" * 70)
    print("EXTENDED ENSEMBLE - ALL AVAILABLE SOURCES")
    print("=" * 70)
    print(f"MIN_IMPROVEMENT threshold: {MIN_IMPROVEMENT}")
    
    start_time = time.time()
    
    # Load exp_016 as baseline (our best valid submission)
    print("\nLoading exp_016 (baseline)...")
    baseline_df = pd.read_csv('/home/code/experiments/016_mega_ensemble_external/submission.csv')
    baseline_configs = parse_submission(baseline_df)
    baseline_scores = {n: calculate_score(baseline_configs[n]) for n in range(1, 201)}
    baseline_total = sum(baseline_scores.values())
    print(f"exp_016 baseline total: {baseline_total:.6f}")
    
    # Load ALL external CSV files
    sources = {}
    
    print("\nLoading external data sources...")
    external_files = glob.glob('/home/code/external_data/**/*.csv', recursive=True)
    for f in external_files:
        configs = load_submission_fast(f)
        if configs is not None:
            name = f.replace('/home/code/external_data/', '')
            sources[name] = configs
            total = sum(calculate_score(configs[n]) for n in range(1, 201))
            print(f"  {name}: {total:.6f}")
    
    # Load ALL internal snapshots
    print("\nLoading internal snapshots...")
    snapshot_files = glob.glob('/home/nonroot/snapshots/santa-2025/**/*.csv', recursive=True)
    loaded = 0
    for f in snapshot_files:
        configs = load_submission_fast(f)
        if configs is not None:
            sources[f] = configs
            loaded += 1
            if loaded % 500 == 0:
                print(f"  Loaded {loaded} snapshots...")
    print(f"  Total snapshots loaded: {loaded}")
    
    print(f"\nTotal sources: {len(sources)}")
    
    # Build extended ensemble
    print("\n" + "=" * 70)
    print("BUILDING EXTENDED ENSEMBLE")
    print("=" * 70)
    
    best_per_n = {}
    improvements = []
    rejected_small = []
    rejected_overlap = []
    
    for n in range(1, 201):
        best_score = baseline_scores[n]
        best_config = baseline_configs[n]
        best_source = "exp_016"
        
        for source_name, configs in sources.items():
            if n not in configs:
                continue
            
            config = configs[n]
            if len(config) != n:
                continue
            
            if not check_no_nan(config):
                continue
            
            score = calculate_score(config)
            improvement = best_score - score
            
            # CRITICAL: Only accept if improvement >= 0.001
            if improvement < MIN_IMPROVEMENT:
                if improvement > 0:
                    rejected_small.append((n, improvement, source_name.split('/')[-1]))
                continue
            
            # Strict overlap validation
            if not validate_no_overlap_strict(config):
                rejected_overlap.append((n, improvement, source_name.split('/')[-1]))
                continue
            
            best_score = score
            best_config = config
            best_source = source_name
        
        best_per_n[n] = best_config
        
        if best_source != "exp_016":
            improvement = baseline_scores[n] - best_score
            improvements.append((n, improvement, best_source.split('/')[-1]))
            print(f"✅ N={n}: IMPROVED by {improvement:.6f} from {best_source.split('/')[-1]}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    final_total = sum(calculate_score(best_per_n[n]) for n in range(1, 201))
    total_improvement = baseline_total - final_total
    
    print(f"exp_016 baseline total: {baseline_total:.6f}")
    print(f"Final total: {final_total:.6f}")
    print(f"Improvement over exp_016: {total_improvement:.6f}")
    print(f"\nN values improved: {len(improvements)}")
    print(f"Rejected (too small < {MIN_IMPROVEMENT}): {len(rejected_small)}")
    print(f"Rejected (overlap): {len(rejected_overlap)}")
    
    # Final validation
    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)
    
    invalid_n = []
    for n in range(1, 201):
        config = best_per_n[n]
        if len(config) != n:
            invalid_n.append(n)
            continue
        if not check_no_nan(config):
            invalid_n.append(n)
            continue
        if not validate_no_overlap_strict(config):
            print(f"N={n}: Overlap - falling back to exp_016")
            best_per_n[n] = baseline_configs[n]
            invalid_n.append(n)
    
    if not invalid_n:
        print("✅ All configurations valid!")
    else:
        print(f"⚠️ Fell back for {len(invalid_n)} N values")
        final_total = sum(calculate_score(best_per_n[n]) for n in range(1, 201))
        print(f"Updated final total: {final_total:.6f}")
    
    # Save submission
    save_submission(best_per_n, 'submission.csv')
    print("\nSaved submission.csv")
    
    # Validate format
    df = pd.read_csv('submission.csv')
    print(f"Rows: {len(df)}")
    for col in ['x', 'y', 'deg']:
        vals = df[col].astype(str).str.replace('s', '', regex=False)
        nan_count = vals.apply(lambda x: 'nan' in x.lower()).sum()
        if nan_count > 0:
            print(f"❌ WARNING: {nan_count} NaN values in {col}")
        else:
            print(f"✅ No NaN values in {col}")
    
    # Save metrics
    metrics = {
        'cv_score': final_total,
        'baseline_score': baseline_total,
        'improvement_over_baseline': total_improvement,
        'num_improvements': len(improvements),
        'min_improvement_threshold': MIN_IMPROVEMENT,
        'rejected_small': len(rejected_small),
        'rejected_overlap': len(rejected_overlap),
        'notes': 'Extended ensemble from ALL available sources with MIN_IMPROVEMENT=0.001'
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
    os.chdir('/home/code/experiments/017_extended_ensemble')
    main()
