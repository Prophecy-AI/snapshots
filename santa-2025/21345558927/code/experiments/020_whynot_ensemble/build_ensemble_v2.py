import pandas as pd
import numpy as np
import glob
import os
from numba import njit
import math
import json
from shapely import Polygon
from shapely.affinity import rotate, translate

# NO THRESHOLD - accept any improvement
MIN_IMPROVEMENT = 1e-8  # Essentially zero

# Tree polygon vertices
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]

def get_tree_polygon(x, y, angle):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = rotate(poly, angle, origin=(0, 0))
    poly = translate(poly, x, y)
    return poly

def check_overlaps(xs, ys, ds):
    """Check if any trees overlap. Returns True if overlaps exist."""
    n = len(xs)
    if n <= 1:
        return False
    
    polygons = [get_tree_polygon(xs[i], ys[i], ds[i]) for i in range(n)]
    
    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            if polygons[i].intersects(polygons[j]):
                if not polygons[i].touches(polygons[j]):
                    overlap_area = polygons[i].intersection(polygons[j]).area
                    if overlap_area > 1e-10:
                        return True
    return False

@njit
def make_polygon_template():
    tw=0.15; th=0.2; bw=0.7; mw=0.4; ow=0.25
    tip=0.8; t1=0.5; t2=0.25; base=0.0; tbot=-th
    x=np.array([0,ow/2,ow/4,mw/2,mw/4,bw/2,tw/2,tw/2,-tw/2,-tw/2,-bw/2,-mw/4,-mw/2,-ow/4,-ow/2],np.float64)
    y=np.array([tip,t1,t1,t2,t2,base,base,tbot,tbot,base,base,t2,t2,t1,t1],np.float64)
    return x,y

@njit
def score_group(xs,ys,degs,tx,ty):
    n=xs.size; V=tx.size
    mnx=1e300; mny=1e300; mxx=-1e300; mxy=-1e300
    for i in range(n):
        r=degs[i]*math.pi/180.0
        c=math.cos(r); s=math.sin(r)
        xi=xs[i]; yi=ys[i]
        for j in range(V):
            X=c*tx[j]-s*ty[j]+xi
            Y=s*tx[j]+c*ty[j]+yi
            if X<mnx: mnx=X
            if X>mxx: mxx=X
            if Y<mny: mny=Y
            if Y>mxy: mxy=Y
    side=max(mxx-mnx,mxy-mny)
    return side*side/n

def strip(a):
    return np.array([float(str(v).replace("s","")) for v in a],np.float64)

# Collect all CSV files from multiple sources
all_files = []

# 1. New external data (including new kernel outputs)
all_files += glob.glob('/home/code/data/external/**/*.csv', recursive=True)

# 2. Snapshots (existing) - EXCLUDE ensemble_best.csv which has overlaps
snapshot_files = glob.glob('/home/nonroot/snapshots/santa-2025/*/code/**/*.csv', recursive=True)
# Filter out known bad files
bad_patterns = ['ensemble_best.csv', 'candidate_']
for fp in snapshot_files:
    if not any(bad in fp for bad in bad_patterns):
        all_files.append(fp)

print(f"Total CSV files to scan: {len(all_files)}")

tx, ty = make_polygon_template()
best = {n: {"score": 1e300, "data": None, "src": None} for n in range(1, 201)}

# Load bbox3-ensemble-update as the baseline (it's the best kernel output!)
baseline_path = '/home/code/data/external/kernel_outputs/bbox3-ensemble-update/submission.csv'
print("Loading bbox3-ensemble-update as baseline...")
baseline_df = pd.read_csv(baseline_path)
baseline_df['N'] = baseline_df['id'].str.split('_').str[0].astype(int)
baseline_total = 0
for n, g in baseline_df.groupby('N'):
    xs = strip(g['x'].to_numpy())
    ys = strip(g['y'].to_numpy())
    ds = strip(g['deg'].to_numpy())
    if np.isnan(xs).any() or np.isnan(ys).any() or np.isnan(ds).any():
        print(f"WARNING: Baseline N={n} has NaN values!")
        continue
    sc = score_group(xs, ys, ds, tx, ty)
    baseline_total += sc
    best[n] = {"score": float(sc), "data": g.drop(columns=['N']).copy(), "src": "bbox3-ensemble-update"}

print(f"Baseline total score: {baseline_total:.6f}")

# Track improvements
improvements = []
files_processed = 0
files_with_improvements = set()
overlap_rejections = 0

# Scan all files for better solutions
for fp in all_files:
    if fp == baseline_path:
        continue
    try:
        df = pd.read_csv(fp)
    except Exception as e:
        continue
    
    if not {'id', 'x', 'y', 'deg'}.issubset(df.columns):
        continue
    
    files_processed += 1
    
    try:
        df['N'] = df['id'].str.split('_').str[0].astype(int)
    except:
        continue
        
    for n, g in df.groupby('N'):
        if n < 1 or n > 200:
            continue
        try:
            xs = strip(g['x'].to_numpy())
            ys = strip(g['y'].to_numpy())
            ds = strip(g['deg'].to_numpy())
        except:
            continue
            
        # Check for NaN
        if np.isnan(xs).any() or np.isnan(ys).any() or np.isnan(ds).any():
            continue
            
        # Check correct number of trees
        if len(xs) != n:
            continue
            
        sc = score_group(xs, ys, ds, tx, ty)
        improvement = best[n]['score'] - sc
        
        if improvement >= MIN_IMPROVEMENT:
            # CRITICAL: Check for overlaps before accepting
            if check_overlaps(list(xs), list(ys), list(ds)):
                overlap_rejections += 1
                continue
            
            old_score = best[n]['score']
            old_src = best[n]['src']
            best[n] = {"score": float(sc), "data": g.drop(columns=['N']).copy(), "src": fp}
            improvements.append({
                'n': n,
                'old_score': old_score,
                'new_score': sc,
                'improvement': improvement,
                'old_source': old_src,
                'source': os.path.basename(fp)
            })
            files_with_improvements.add(fp)
            if improvement >= 0.0001:  # Only print significant improvements
                print(f"N={n}: {old_score:.6f} -> {sc:.6f} (+{improvement:.6f}) from {os.path.basename(fp)}")

print(f"\nFiles processed: {files_processed}")
print(f"Files with improvements: {len(files_with_improvements)}")
print(f"Total improvements found: {len(improvements)}")
print(f"Overlap rejections: {overlap_rejections}")

# Build final submission
rows = []
for n in range(1, 201):
    if best[n]['data'] is not None:
        rows.append(best[n]['data'])
    else:
        print(f"WARNING: No data for N={n}")

out = pd.concat(rows, ignore_index=True)
out['sn'] = out['id'].str.split('_').str[0].astype(int)
out['si'] = out['id'].str.split('_').str[1].astype(int)
out = out.sort_values(['sn', 'si']).drop(columns=['sn', 'si'])
out = out[['id', 'x', 'y', 'deg']]
out.to_csv('submission.csv', index=False)

# Calculate total score
total = sum(best[n]['score'] for n in range(1, 201))
improvement_from_baseline = baseline_total - total

print(f"\n=== FINAL RESULTS ===")
print(f"Baseline score: {baseline_total:.6f}")
print(f"New total score: {total:.6f}")
print(f"Total improvement: {improvement_from_baseline:.6f}")

# Count sources
source_counts = {}
for n in range(1, 201):
    src = best[n]['src']
    if src:
        src_name = os.path.basename(src) if '/' in src else src
    else:
        src_name = 'unknown'
    if src_name not in source_counts:
        source_counts[src_name] = 0
    source_counts[src_name] += 1

print(f"\nSource distribution:")
for src, count in sorted(source_counts.items(), key=lambda x: -x[1])[:15]:
    print(f"  {src}: {count} N values")

# Save metrics
metrics = {
    'cv_score': total,
    'baseline_score': baseline_total,
    'improvement': improvement_from_baseline,
    'num_improvements': len(improvements),
    'overlap_rejections': overlap_rejections,
    'source_counts': source_counts,
    'improvements': improvements[:100]
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Copy to submission folder
import shutil
shutil.copy('submission.csv', '/home/submission/submission.csv')
print(f"\nSubmission saved to /home/submission/submission.csv")
