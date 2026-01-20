"""Score a Santa 2025 submission file."""
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import sys

# Tree polygon vertices
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]
BASE_TREE = Polygon(zip(TX, TY))

def parse_value(val):
    """Parse a value that may have 's' prefix."""
    if isinstance(val, str):
        if val.startswith('s'):
            return float(val[1:])
        return float(val)
    return float(val)

def create_tree(x, y, deg):
    """Create a tree polygon at position (x, y) with rotation deg."""
    tree = affinity.rotate(BASE_TREE, deg, origin=(0, 0))
    tree = affinity.translate(tree, x, y)
    return tree

def get_bounding_box_side(trees):
    """Get the side length of the bounding box containing all trees."""
    if not trees:
        return 0
    
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    
    for tree in trees:
        bounds = tree.bounds  # (minx, miny, maxx, maxy)
        min_x = min(min_x, bounds[0])
        min_y = min(min_y, bounds[1])
        max_x = max(max_x, bounds[2])
        max_y = max(max_y, bounds[3])
    
    return max(max_x - min_x, max_y - min_y)

def check_overlaps(trees, tolerance=1e-6):
    """Check if any trees overlap."""
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            if trees[i].buffer(-tolerance).intersects(trees[j].buffer(-tolerance)):
                intersection = trees[i].intersection(trees[j])
                if intersection.area > tolerance:
                    return True, (i, j)
    return False, None

def score_submission(csv_path, check_overlap=False):
    """Calculate total score for a submission."""
    df = pd.read_csv(csv_path)
    
    total_score = 0
    overlap_count = 0
    
    for n in range(1, 201):
        # Get trees for this N
        prefix = f"{n:03d}_"
        n_rows = df[df['id'].str.startswith(prefix)]
        
        if len(n_rows) != n:
            print(f"WARNING: N={n} has {len(n_rows)} trees instead of {n}")
            continue
        
        trees = []
        for _, row in n_rows.iterrows():
            x = parse_value(row['x'])
            y = parse_value(row['y'])
            deg = parse_value(row['deg'])
            trees.append(create_tree(x, y, deg))
        
        side = get_bounding_box_side(trees)
        score_n = side**2 / n
        total_score += score_n
        
        if check_overlap:
            has_overlap, pair = check_overlaps(trees)
            if has_overlap:
                overlap_count += 1
                print(f"N={n}: OVERLAP between trees {pair}")
    
    print(f"\nTotal score: {total_score:.6f}")
    if check_overlap:
        print(f"Configurations with overlaps: {overlap_count}")
    
    return total_score

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python score_submission.py <submission.csv> [--check-overlap]")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    check_overlap = "--check-overlap" in sys.argv
    
    score = score_submission(csv_path, check_overlap)
