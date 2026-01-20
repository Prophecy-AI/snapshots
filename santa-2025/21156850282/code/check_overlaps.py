"""Check for overlapping trees in a submission."""
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

def check_overlaps_in_group(trees, tolerance=1e-9):
    """Check if any trees in a group overlap."""
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            # Check if polygons intersect (excluding just touching)
            if trees[i].intersects(trees[j]):
                intersection = trees[i].intersection(trees[j])
                if intersection.area > tolerance:
                    return True, (i, j), intersection.area
    return False, None, 0

def check_submission(csv_path, verbose=True):
    """Check all groups in a submission for overlaps."""
    df = pd.read_csv(csv_path)
    
    overlapping_groups = []
    
    for n in range(1, 201):
        prefix = f"{n:03d}_"
        n_rows = df[df['id'].str.startswith(prefix)]
        
        if len(n_rows) != n:
            if verbose:
                print(f"WARNING: N={n} has {len(n_rows)} trees instead of {n}")
            continue
        
        trees = []
        for _, row in n_rows.iterrows():
            x = parse_value(row['x'])
            y = parse_value(row['y'])
            deg = parse_value(row['deg'])
            trees.append(create_tree(x, y, deg))
        
        has_overlap, pair, area = check_overlaps_in_group(trees)
        if has_overlap:
            overlapping_groups.append((n, pair, area))
            if verbose:
                print(f"N={n}: OVERLAP between trees {pair}, area={area:.10f}")
    
    if verbose:
        print(f"\nTotal overlapping groups: {len(overlapping_groups)}")
    
    return overlapping_groups

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_overlaps.py <submission.csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    overlaps = check_submission(csv_path)
    
    if len(overlaps) == 0:
        print("✓ No overlaps found - submission is valid!")
    else:
        print(f"✗ Found {len(overlaps)} groups with overlaps")
