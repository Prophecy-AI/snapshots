import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity

TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]

def create_tree_polygon(x, y, deg):
    initial_polygon = Polygon(list(zip(TX, TY)))
    rotated = affinity.rotate(initial_polygon, deg, origin=(0, 0))
    return affinity.translate(rotated, xoff=x, yoff=y)

# Check N=13 specifically (the one that failed)
df = pd.read_csv("submission_eazy_longer.csv")
n13 = df[df['id'].str.startswith('013_')]
print(f"N=13 has {len(n13)} trees")

trees = []
for _, row in n13.iterrows():
    x = float(str(row['x']).replace('s', ''))
    y = float(str(row['y']).replace('s', ''))
    deg = float(str(row['deg']).replace('s', ''))
    trees.append(create_tree_polygon(x, y, deg))

# Check all pairs
for i in range(len(trees)):
    for j in range(i+1, len(trees)):
        if trees[i].intersects(trees[j]) and not trees[i].touches(trees[j]):
            intersection = trees[i].intersection(trees[j])
            if intersection.area > 0:
                print(f"Trees {i} and {j} overlap with area {intersection.area:.2e}")
