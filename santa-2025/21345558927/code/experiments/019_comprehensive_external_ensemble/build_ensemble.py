import pandas as pd
import numpy as np
import glob
import os
from numba import njit
import math
import json

MIN_IMPROVEMENT = 0.001  # Safety threshold

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

# 1. New external data
all_files += glob.glob('/home/code/data/external/**/*.csv', recursive=True)

# 2. Snapshots (existing)
all_files += glob.glob('/home/nonroot/snapshots/santa-2025/*/code/**/*.csv', recursive=True)

# 3. Current best baseline
baseline_path = '/home/code/experiments/016_mega_ensemble_external/submission.csv'
all_files.append(baseline_path)

print(f"Total CSV files to scan: {len(all_files)}")

tx, ty = make_polygon_template()
best = {n: {"score": 1e300, "data": None, "src": None} for n in range(1, 201)}

# Load baseline first
print("Loading baseline...")
baseline_df = pd.read_csv(baseline_path)
baseline_df['N'] = baseline_df['id'].str.split('_').str[0].astype(int)
baseline_total = 0
for n, g in baseline_df.groupby('N'):
    xs = strip(g['x'].to_numpy())
    ys = strip(g['y'].to_numpy())
    ds = strip(g['deg'].to_numpy())
    # Check for NaN
    if np.isnan(xs).any() or np.isnan(ys).any() or np.isnan(ds).any():
        print(f"WARNING: Baseline N={n} has NaN values!")
        continue
    sc = score_group(xs, ys, ds, tx, ty)
    baseline_total += sc
    best[n] = {"score": float(sc), "data": g.drop(columns=['N']).copy(), "src": "baseline"}

print(f"Baseline total score: {baseline_total:.6f}")

# Track improvements
improvements = []
files_processed = 0
files_with_improvements = set()

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
            old_score = best[n]['score']
            best[n] = {"score": float(sc), "data": g.drop(columns=['N']).copy(), "src": fp}
            improvements.append({
                'n': n,
                'old_score': old_score,
                'new_score': sc,
                'improvement': improvement,
                'source': os.path.basename(fp)
            })
            files_with_improvements.add(fp)
            print(f"N={n}: {old_score:.6f} -> {sc:.6f} (+{improvement:.6f}) from {os.path.basename(fp)}")

print(f"\nFiles processed: {files_processed}")
print(f"Files with improvements: {len(files_with_improvements)}")
print(f"Total improvements found: {len(improvements)}")

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

# Save metrics
metrics = {
    'cv_score': total,
    'baseline_score': baseline_total,
    'improvement': improvement_from_baseline,
    'num_improvements': len(improvements),
    'improvements': improvements[:50]  # Save first 50
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Save improvement details
if improvements:
    imp_df = pd.DataFrame(improvements)
    imp_df.to_csv('improvements.csv', index=False)
    print(f"\nTop improvements by N:")
    print(imp_df.sort_values('improvement', ascending=False).head(20))

# Copy to submission folder
import shutil
shutil.copy('submission.csv', '/home/submission/submission.csv')
print(f"\nSubmission saved to /home/submission/submission.csv")
