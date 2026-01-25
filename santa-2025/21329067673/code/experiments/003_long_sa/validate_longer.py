import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from shapely.geometry import Polygon
from shapely import affinity
import json

getcontext().prec = 25

TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]

class ChristmasTree:
    def __init__(self, center_x="0", center_y="0", angle="0"):
        self.center_x = Decimal(str(center_x).replace('s', ''))
        self.center_y = Decimal(str(center_y).replace('s', ''))
        self.angle = Decimal(str(angle).replace('s', ''))
        
        initial_polygon = Polygon(list(zip(TX, TY)))
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated, 
            xoff=float(self.center_x), yoff=float(self.center_y))

def has_overlap(trees):
    if len(trees) <= 1:
        return False
    for i, t1 in enumerate(trees):
        for j, t2 in enumerate(trees):
            if i < j:
                if t1.polygon.intersects(t2.polygon) and not t1.polygon.touches(t2.polygon):
                    intersection = t1.polygon.intersection(t2.polygon)
                    if intersection.area > 1e-10:
                        return True
    return False

def get_side_length(trees):
    xys = np.concatenate([np.asarray(t.polygon.exterior.xy).T for t in trees])
    min_x, min_y = xys.min(axis=0)
    max_x, max_y = xys.max(axis=0)
    return max(max_x - min_x, max_y - min_y)

def load_and_validate(csv_path):
    df = pd.read_csv(csv_path)
    configs = {}
    for _, row in df.iterrows():
        id_parts = row['id'].split('_')
        n = int(id_parts[0])
        x = str(row['x']).replace('s', '')
        y = str(row['y']).replace('s', '')
        deg = str(row['deg']).replace('s', '')
        if n not in configs:
            configs[n] = []
        configs[n].append(ChristmasTree(x, y, deg))
    
    total_score = 0
    overlap_count = 0
    
    for n in sorted(configs.keys()):
        trees = configs[n]
        side = get_side_length(trees)
        score = side**2 / n
        total_score += score
        
        if has_overlap(trees):
            overlap_count += 1
            print(f"WARNING: N={n} has overlaps!")
    
    return total_score, overlap_count, configs

print("Validating submission_eazy_longer.csv...")
score, overlaps, configs = load_and_validate("submission_eazy_longer.csv")
print(f"\nTotal score: {score:.6f}")
print(f"Overlapping N values: {overlaps}")

if overlaps == 0:
    print("\nValidation PASSED - no overlaps detected")
    metrics = {
        'cv_score': score,
        'target_score': 68.892266,
        'gap': score - 68.892266,
        'gap_percent': (score - 68.892266) / 68.892266 * 100,
        'baseline_score': 70.676102,
        'improvement': 70.676102 - score
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved")
    print(f"Total improvement from baseline: {70.676102 - score:.6f}")
else:
    print("\nValidation FAILED - overlaps detected!")
