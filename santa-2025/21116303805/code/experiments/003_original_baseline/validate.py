import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree

getcontext().prec = 30

class ChristmasTree:
    def __init__(self, center_x='0', center_y='0', angle='0'):
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        trunk_w = Decimal('0.15')
        trunk_h = Decimal('0.2')
        base_w = Decimal('0.7')
        mid_w = Decimal('0.4')
        top_w = Decimal('0.25')
        tip_y = Decimal('0.8')
        tier_1_y = Decimal('0.5')
        tier_2_y = Decimal('0.25')
        base_y = Decimal('0.0')
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon([
            (float(0), float(tip_y)),
            (float(top_w / 2), float(tier_1_y)),
            (float(top_w / 4), float(tier_1_y)),
            (float(mid_w / 2), float(tier_2_y)),
            (float(mid_w / 4), float(tier_2_y)),
            (float(base_w / 2), float(base_y)),
            (float(trunk_w / 2), float(base_y)),
            (float(trunk_w / 2), float(trunk_bottom_y)),
            (float(-trunk_w / 2), float(trunk_bottom_y)),
            (float(-trunk_w / 2), float(base_y)),
            (float(-base_w / 2), float(base_y)),
            (float(-mid_w / 4), float(tier_2_y)),
            (float(-mid_w / 2), float(tier_2_y)),
            (float(-top_w / 4), float(tier_1_y)),
            (float(-top_w / 2), float(tier_1_y)),
        ])

        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated, xoff=float(self.center_x), yoff=float(self.center_y))

def load_trees_for_n(df, n):
    prefix = f"{n:03d}_"
    subset = df[df['id'].str.startswith(prefix)]
    trees = []
    for _, row in subset.iterrows():
        x = str(row['x']).lstrip('s')
        y = str(row['y']).lstrip('s')
        deg = str(row['deg']).lstrip('s')
        trees.append(ChristmasTree(x, y, deg))
    return trees

def has_overlap(trees):
    if len(trees) <= 1:
        return False
    polygons = [t.polygon for t in trees]
    tree_index = STRtree(polygons)
    
    for i, poly in enumerate(polygons):
        indices = tree_index.query(poly)
        for idx in indices:
            if idx != i:
                if poly.intersects(polygons[idx]) and not poly.touches(polygons[idx]):
                    intersection = poly.intersection(polygons[idx])
                    if intersection.area > 1e-12:
                        return True
    return False

def get_bounding_box_side(trees):
    if not trees:
        return 0
    all_coords = []
    for tree in trees:
        coords = np.array(tree.polygon.exterior.coords)
        all_coords.append(coords)
    all_coords = np.vstack(all_coords)
    x_range = all_coords[:, 0].max() - all_coords[:, 0].min()
    y_range = all_coords[:, 1].max() - all_coords[:, 1].min()
    return max(x_range, y_range)

def score_submission(df, max_n=200):
    total_score = 0
    overlaps = []
    for n in range(1, max_n + 1):
        trees = load_trees_for_n(df, n)
        if len(trees) != n:
            print(f"Warning: n={n} has {len(trees)} trees instead of {n}")
            continue
        if has_overlap(trees):
            overlaps.append(n)
        side = get_bounding_box_side(trees)
        score_n = (side ** 2) / n
        total_score += score_n
    return total_score, overlaps

# Load and validate
df = pd.read_csv('/home/submission/submission.csv')
print(f"Submission shape: {df.shape}")
print(f"First few rows:\n{df.head()}")

print("\nScoring submission...")
score, overlaps = score_submission(df)
print(f"Score: {score:.6f}")
print(f"Overlapping configurations: {overlaps}")
print(f"\nTarget: 68.922808")
print(f"Gap to target: {score - 68.922808:.6f}")

# Save metrics
import json
with open('/home/code/experiments/003_original_baseline/metrics.json', 'w') as f:
    json.dump({'cv_score': score, 'overlaps': overlaps}, f)
