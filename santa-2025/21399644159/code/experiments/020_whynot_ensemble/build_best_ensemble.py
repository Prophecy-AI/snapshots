import pandas as pd
import numpy as np
import glob
import os
from numba import njit
import math
import json
from shapely import Polygon
from shapely.affinity import rotate, translate

# NO threshold - accept all improvements
MIN_IMPROVEMENT = 1e-10

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

tx, ty = make_polygon_template()

# Load primary sources
sources = {
    'why-not': '/home/code/data/external/kernel_outputs/why-not/submission.csv',
    'team-blend': '/home/code/data/external/kernel_outputs/team-optimization-blend/submission.csv',
    'team-blend-ensemble': '/home/code/data/external/kernel_outputs/team-optimization-blend/submission_ensemble.csv',
    'current': '/home/code/experiments/016_mega_ensemble_external/submission.csv',
    'decent-starting': '/home/code/data/external/kernel_outputs/decent-starting-solution/submission.csv',
    'santa-claude': '/home/code/data/external/kernel_outputs/santa-claude/submission.csv',
}

# Also scan all external data
all_files = glob.glob('/home/code/data/external/**/*.csv', recursive=True)
# Filter out known bad files
bad_patterns = ['ensemble_best.csv', 'candidate_']
all_files = [f for f in all_files if not any(bad in f for bad in bad_patterns)]

# Add snapshot files
snapshot_files = glob.glob('/home/nonroot/snapshots/santa-2025/*/code/**/*.csv', recursive=True)
snapshot_files = [f for f in snapshot_files if not any(bad in f for bad in bad_patterns)]
all_files.extend(snapshot_files)

print(f"Total files to scan: {len(all_files)}")

# Initialize best with why-not as base (it's validated)
best = {n: {"score": 1e300, "data": None, "src": None} for n in range(1, 201)}

# Load why-not first as base
print("Loading why-not as base...")
why_not_df = pd.read_csv(sources['why-not'])
why_not_df['N'] = why_not_df['id'].str.split('_').str[0].astype(int)
why_not_total = 0
for n, g in why_not_df.groupby('N'):
    xs = strip(g['x'].to_numpy())
    ys = strip(g['y'].to_numpy())
    ds = strip(g['deg'].to_numpy())
    if np.isnan(xs).any() or np.isnan(ys).any() or np.isnan(ds).any():
        continue
    sc = score_group(xs, ys, ds, tx, ty)
    why_not_total += sc
    best[n] = {"score": float(sc), "data": g.drop(columns=['N']).copy(), "src": "why-not"}

print(f"Why-not base score: {why_not_total:.6f}")

# Track improvements
improvements = []
overlap_rejections = 0

# Scan all files for better solutions
for fp in all_files:
    try:
        df = pd.read_csv(fp)
    except Exception:
        continue
    
    if not {'id', 'x', 'y', 'deg'}.issubset(df.columns):
        continue
    
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
            
        if np.isnan(xs).any() or np.isnan(ys).any() or np.isnan(ds).any():
            continue
            
        if len(xs) != n:
            continue
            
        sc = score_group(xs, ys, ds, tx, ty)
        improvement = best[n]['score'] - sc
        
        if improvement >= MIN_IMPROVEMENT:
            # Check for overlaps
            if check_overlaps(list(xs), list(ys), list(ds)):
                overlap_rejections += 1
                continue
            
            old_score = best[n]['score']
            best[n] = {"score": float(sc), "data": g.drop(columns=['N']).copy(), "src": os.path.basename(fp)}
            improvements.append({
                'n': n,
                'old_score': old_score,
                'new_score': sc,
                'improvement': improvement,
                'source': os.path.basename(fp)
            })

print(f"\nImprovements found: {len(improvements)}")
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
improvement_from_whynot = why_not_total - total

print(f"\n=== FINAL RESULTS ===")
print(f"Why-not base score: {why_not_total:.6f}")
print(f"New total score: {total:.6f}")
print(f"Improvement over why-not: {improvement_from_whynot:.6f}")

# Count sources
source_counts = {}
for n in range(1, 201):
    src = best[n]['src']
    source_counts[src] = source_counts.get(src, 0) + 1

print(f"\nSource distribution:")
for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
    print(f"  {src}: {count} N values")

# Save metrics
metrics = {
    'cv_score': total,
    'why_not_score': why_not_total,
    'improvement': improvement_from_whynot,
    'num_improvements': len(improvements),
    'overlap_rejections': overlap_rejections,
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Copy to submission folder
import shutil
shutil.copy('submission.csv', '/home/submission/submission.csv')
print(f"\nSubmission saved to /home/submission/submission.csv")
