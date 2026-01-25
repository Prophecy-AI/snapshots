import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from shapely.geometry import Polygon
from shapely import affinity
import json

getcontext().prec = 25

TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]

def create_tree_polygon(x, y, deg):
    initial_polygon = Polygon(list(zip(TX, TY)))
    rotated = affinity.rotate(initial_polygon, deg, origin=(0, 0))
    return affinity.translate(rotated, xoff=x, yoff=y)

def has_overlap(trees):
    if len(trees) <= 1:
        return False
    for i in range(len(trees)):
        for j in range(i+1, len(trees)):
            if trees[i].intersects(trees[j]) and not trees[i].touches(trees[j]):
                intersection = trees[i].intersection(trees[j])
                if intersection.area > 1e-15:
                    return True
    return False

def get_side_length(trees):
    xys = np.concatenate([np.asarray(t.exterior.xy).T for t in trees])
    min_x, min_y = xys.min(axis=0)
    max_x, max_y = xys.max(axis=0)
    return max(max_x - min_x, max_y - min_y)

def load_and_validate(csv_path):
    df = pd.read_csv(csv_path)
    configs = {}
    for _, row in df.iterrows():
        id_parts = row['id'].split('_')
        n = int(id_parts[0])
        x = float(str(row['x']).replace('s', ''))
        y = float(str(row['y']).replace('s', ''))
        deg = float(str(row['deg']).replace('s', ''))
        if n not in configs:
            configs[n] = []
        configs[n].append(create_tree_polygon(x, y, deg))
    
    total_score = 0
    overlap_count = 0
    overlap_ns = []
    
    for n in sorted(configs.keys()):
        trees = configs[n]
        side = get_side_length(trees)
        score = side**2 / n
        total_score += score
        
        if has_overlap(trees):
            overlap_count += 1
            overlap_ns.append(n)
            print(f"WARNING: N={n} has overlaps!")
    
    return total_score, overlap_count, overlap_ns

print("Validating submission_ensemble_strict.csv...")
score, overlaps, overlap_ns = load_and_validate("submission_ensemble_strict.csv")
print(f"\nTotal score: {score:.6f}")
print(f"Overlapping N values: {overlaps}")

if overlaps == 0:
    print("\nValidation PASSED - no overlaps detected")
else:
    print(f"\nValidation FAILED - overlaps at N={overlap_ns}")
