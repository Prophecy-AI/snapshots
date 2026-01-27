"""
Conservative Ensemble with MIN_IMPROVEMENT=0.003
- Slightly lower threshold to capture some improvements
- Still safer than 0.001 which caused failures
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

# Slightly lower threshold
MIN_IMPROVEMENT = 0.003

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
    print("CONSERVATIVE ENSEMBLE (MIN_IMPROVEMENT=0.003)")
    print("=" * 70)
    print(f"MIN_IMPROVEMENT threshold: {MIN_IMPROVEMENT}")
    
    start_time = time.time()
    
    # Load exp_010 as our baseline
    print("\nLoading exp_010...")
    exp010_df = pd.read_csv('/home/code/experiments/010_safe_ensemble/submission.csv')
    exp010_configs = parse_submission(exp010_df)
    exp010_scores = {n: calculate_score(exp010_configs[n]) for n in range(1, 201)}
    exp010_total = sum(exp010_scores.values())
    print(f"exp_010 total: {exp010_total:.6f}")
    
    # Load all sources
    sources = {}
    
    # External data
    print("\nLoading external data...")
    external_files = [
        '/home/code/external_data/santa-2025.csv',
        '/home/code/external_data/70.378875862989_20260126_045659.csv',
    ]
    for f in external_files:
        if os.path.exists(f):
            configs = load_submission_fast(f)
            if configs is not None:
                name = f.split('/')[-1]
                sources[name] = configs
                total = sum(calculate_score(configs[n]) for n in range(1, 201))
                print(f"  {name}: {total:.6f}")
    
    # Internal snapshots
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
    
    # Build ensemble
    print("\n" + "=" * 70)
    print("BUILDING ENSEMBLE")
    print("=" * 70)
    
    best_per_n = {}
    improvements = []
    rejected_small = []
    rejected_overlap = []
    
    for n in range(1, 201):
        best_score = exp010_scores[n]
        best_config = exp010_configs[n]
        best_source = "exp_010"
        
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
            
            if improvement < MIN_IMPROVEMENT:
                if improvement > 0:
                    rejected_small.append((n, improvement, source_name.split('/')[-1]))
                continue
            
            if not validate_no_overlap_strict(config):
                rejected_overlap.append((n, improvement, source_name.split('/')[-1]))
                continue
            
            best_score = score
            best_config = config
            best_source = source_name
        
        best_per_n[n] = best_config
        
        if best_source != "exp_010":
            improvement = exp010_scores[n] - best_score
            improvements.append((n, improvement, best_source.split('/')[-1]))
            print(f"✅ N={n}: IMPROVED by {improvement:.6f} from {best_source.split('/')[-1]}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    final_total = sum(calculate_score(best_per_n[n]) for n in range(1, 201))
    total_improvement = exp010_total - final_total
    
    print(f"exp_010 total: {exp010_total:.6f}")
    print(f"Final total: {final_total:.6f}")
    print(f"Improvement over exp_010: {total_improvement:.6f}")
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
            print(f"N={n}: Overlap - falling back to exp_010")
            best_per_n[n] = exp010_configs[n]
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
    
    # Save metrics
    metrics = {
        'cv_score': final_total,
        'exp010_score': exp010_total,
        'improvement_over_exp010': total_improvement,
        'num_improvements': len(improvements),
        'min_improvement_threshold': MIN_IMPROVEMENT,
        'rejected_small': len(rejected_small),
        'rejected_overlap': len(rejected_overlap),
        'notes': f'Conservative ensemble with MIN_IMPROVEMENT={MIN_IMPROVEMENT}'
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
    os.chdir('/home/code/experiments/014_conservative_ensemble')
    main()
