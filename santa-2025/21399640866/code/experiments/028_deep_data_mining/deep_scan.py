"""
Deep scan of ALL CSV files to find any improvements.
Uses high-precision overlap validation.
"""
import pandas as pd
import numpy as np
from shapely import Polygon
from shapely.affinity import rotate, translate
from numba import njit
import math
import json
import glob
import os
from decimal import Decimal, getcontext

# Set high precision for overlap checking
getcontext().prec = 30

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def get_tree_polygon(x, y, angle):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = rotate(poly, angle, origin=(0, 0), use_radians=False)
    poly = translate(poly, x, y)
    return poly

def check_overlaps_strict(xs, ys, angles):
    """Strict overlap checking with high precision."""
    n = len(xs)
    if n <= 1:
        return False
    
    polygons = [get_tree_polygon(xs[i], ys[i], angles[i]) for i in range(n)]
    
    for i in range(n):
        for j in range(i+1, n):
            if polygons[i].intersects(polygons[j]):
                if not polygons[i].touches(polygons[j]):
                    area = polygons[i].intersection(polygons[j]).area
                    if area > 1e-12:  # Very strict threshold
                        return True
    return False

@njit
def compute_bbox_score(xs, ys, angles, tx, ty):
    n = len(xs)
    if n == 0:
        return float('inf')
    V = len(tx)
    mnx = 1e300
    mny = 1e300
    mxx = -1e300
    mxy = -1e300
    
    for i in range(n):
        r = angles[i] * math.pi / 180.0
        c = math.cos(r)
        s = math.sin(r)
        xi = xs[i]
        yi = ys[i]
        for j in range(V):
            X = c * tx[j] - s * ty[j] + xi
            Y = s * tx[j] + c * ty[j] + yi
            if X < mnx: mnx = X
            if X > mxx: mxx = X
            if Y < mny: mny = Y
            if Y > mxy: mxy = Y
    
    side = max(mxx - mnx, mxy - mny)
    return side * side / n

def strip(v):
    return float(str(v).replace("s", ""))

def load_and_validate_csv(filepath):
    """Load CSV and extract valid configurations."""
    try:
        df = pd.read_csv(filepath)
        if not {'id', 'x', 'y', 'deg'}.issubset(df.columns):
            return None
        
        df['N'] = df['id'].str.split('_').str[0].astype(int)
        return df
    except:
        return None

if __name__ == "__main__":
    print("=" * 70)
    print("Deep External Data Mining")
    print("=" * 70)
    
    # Load current best submission
    baseline_df = pd.read_csv('/home/submission/submission.csv')
    baseline_df['N'] = baseline_df['id'].str.split('_').str[0].astype(int)
    
    # Calculate baseline per-N scores
    baseline_scores = {}
    for n in range(1, 201):
        g = baseline_df[baseline_df['N'] == n]
        xs = np.array([strip(v) for v in g['x']])
        ys = np.array([strip(v) for v in g['y']])
        angles = np.array([strip(v) for v in g['deg']])
        baseline_scores[n] = compute_bbox_score(xs, ys, angles, TX, TY)
    
    baseline_total = sum(baseline_scores.values())
    print(f"Baseline total score: {baseline_total:.6f}")
    
    # Collect all CSV files
    all_files = []
    all_files += glob.glob('/home/nonroot/snapshots/santa-2025/*/code/**/*.csv', recursive=True)
    all_files += glob.glob('/home/code/data/external/**/*.csv', recursive=True)
    
    # Filter out known bad files
    bad_patterns = ['ensemble_best.csv', 'candidate_']
    all_files = [f for f in all_files if not any(bad in f for bad in bad_patterns)]
    
    print(f"Scanning {len(all_files)} CSV files...")
    
    # Track best per-N configurations
    best_per_n = {n: {'score': baseline_scores[n], 'data': None, 'source': 'baseline'} for n in range(1, 201)}
    
    # Track improvements
    improvements = []
    files_processed = 0
    overlap_rejections = 0
    
    MIN_IMPROVEMENT = 0.0001  # Very small threshold
    
    for filepath in all_files:
        df = load_and_validate_csv(filepath)
        if df is None:
            continue
        
        files_processed += 1
        
        for n, g in df.groupby('N'):
            if n < 1 or n > 200:
                continue
            if len(g) != n:
                continue
            
            try:
                xs = np.array([strip(v) for v in g['x']])
                ys = np.array([strip(v) for v in g['y']])
                angles = np.array([strip(v) for v in g['deg']])
            except:
                continue
            
            if np.isnan(xs).any() or np.isnan(ys).any() or np.isnan(angles).any():
                continue
            
            score = compute_bbox_score(xs, ys, angles, TX, TY)
            improvement = best_per_n[n]['score'] - score
            
            if improvement >= MIN_IMPROVEMENT:
                # Check for overlaps with strict validation
                if check_overlaps_strict(list(xs), list(ys), list(angles)):
                    overlap_rejections += 1
                    continue
                
                old_score = best_per_n[n]['score']
                best_per_n[n] = {
                    'score': score,
                    'data': g.drop(columns=['N']).copy(),
                    'source': os.path.basename(filepath)
                }
                improvements.append({
                    'n': n,
                    'old_score': old_score,
                    'new_score': score,
                    'improvement': improvement,
                    'source': os.path.basename(filepath)
                })
                print(f"N={n}: {old_score:.6f} -> {score:.6f} (+{improvement:.6f}) from {os.path.basename(filepath)}")
        
        if files_processed % 500 == 0:
            print(f"Progress: {files_processed} files processed...")
    
    print(f"\nFiles processed: {files_processed}")
    print(f"Improvements found: {len(improvements)}")
    print(f"Overlap rejections: {overlap_rejections}")
    
    # Calculate new total score
    new_total = sum(best_per_n[n]['score'] for n in range(1, 201))
    total_improvement = baseline_total - new_total
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Baseline score: {baseline_total:.6f}")
    print(f"New total score: {new_total:.6f}")
    print(f"Total improvement: {total_improvement:.6f}")
    
    # Save metrics
    metrics = {
        'cv_score': new_total,
        'baseline_score': baseline_total,
        'improvement': total_improvement,
        'num_improvements': len(improvements),
        'overlap_rejections': overlap_rejections,
        'files_processed': files_processed
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # If improvements found, build new submission
    if total_improvement > 0:
        print("\nBuilding new submission...")
        rows = []
        for n in range(1, 201):
            if best_per_n[n]['data'] is not None:
                rows.append(best_per_n[n]['data'])
            else:
                # Use baseline
                g = baseline_df[baseline_df['N'] == n].drop(columns=['N'])
                rows.append(g)
        
        out = pd.concat(rows, ignore_index=True)
        out['sn'] = out['id'].str.split('_').str[0].astype(int)
        out['si'] = out['id'].str.split('_').str[1].astype(int)
        out = out.sort_values(['sn', 'si']).drop(columns=['sn', 'si'])
        out = out[['id', 'x', 'y', 'deg']]
        out.to_csv('submission.csv', index=False)
        
        # Copy to submission folder
        import shutil
        shutil.copy('submission.csv', '/home/submission/submission.csv')
        print(f"New submission saved to /home/submission/submission.csv")
    else:
        print("\nNo improvements found - keeping current submission")
