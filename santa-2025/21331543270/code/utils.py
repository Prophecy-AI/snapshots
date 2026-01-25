"""
Santa 2025 - Christmas Tree Packing Utilities
Reusable code for tree geometry, scoring, collision detection, and submission I/O.
"""

import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from typing import List, Tuple, Dict, Optional
import os

getcontext().prec = 30

class ChristmasTree:
    """Christmas tree polygon with 15 vertices."""
    
    def __init__(self, center_x: str = '0', center_y: str = '0', angle: str = '0'):
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)
        
        # Tree geometry parameters
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

        # 15-vertex polygon
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


def load_submission(filepath: str) -> pd.DataFrame:
    """Load a submission CSV file."""
    df = pd.read_csv(filepath)
    return df


def parse_value(s) -> float:
    """Parse submission value (remove 's' prefix)."""
    if isinstance(s, str) and s.startswith('s'):
        return s[1:]
    return str(s)


def load_trees_for_n(df: pd.DataFrame, n: int) -> List[ChristmasTree]:
    """Load all trees for a given N value."""
    prefix = f"{n:03d}_"
    subset = df[df['id'].str.startswith(prefix)]
    trees = []
    for _, row in subset.iterrows():
        x = parse_value(row['x'])
        y = parse_value(row['y'])
        deg = parse_value(row['deg'])
        trees.append(ChristmasTree(x, y, deg))
    return trees


def get_trees_data_for_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Get the raw data rows for a given N value."""
    prefix = f"{n:03d}_"
    return df[df['id'].str.startswith(prefix)].copy()


def has_overlap(trees: List[ChristmasTree], tolerance: float = 1e-12) -> Tuple[bool, List]:
    """Check if any trees overlap."""
    if len(trees) <= 1:
        return False, []
    
    polygons = [t.polygon for t in trees]
    tree_index = STRtree(polygons)
    overlaps = []
    
    for i, poly in enumerate(polygons):
        indices = tree_index.query(poly)
        for idx in indices:
            if idx > i:  # Only check each pair once
                if poly.intersects(polygons[idx]) and not poly.touches(polygons[idx]):
                    intersection = poly.intersection(polygons[idx])
                    if intersection.area > tolerance:
                        overlaps.append((i, idx, intersection.area))
    
    return len(overlaps) > 0, overlaps


def get_bounding_box_side(trees: List[ChristmasTree]) -> float:
    """Get the side length of the bounding box."""
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


def calculate_score_for_n(trees: List[ChristmasTree], n: int) -> float:
    """Calculate score contribution for a single N value."""
    side = get_bounding_box_side(trees)
    return (side ** 2) / n


def score_submission(df: pd.DataFrame, max_n: int = 200, check_overlaps: bool = True) -> Tuple[float, Dict, List]:
    """Calculate the competition score for a submission."""
    total_score = 0
    scores_by_n = {}
    overlapping_ns = []
    
    for n in range(1, max_n + 1):
        trees = load_trees_for_n(df, n)
        if len(trees) != n:
            print(f"Warning: n={n} has {len(trees)} trees instead of {n}")
            continue
        
        if check_overlaps:
            has_ovlp, _ = has_overlap(trees)
            if has_ovlp:
                overlapping_ns.append(n)
        
        side = get_bounding_box_side(trees)
        score_n = (side ** 2) / n
        scores_by_n[n] = {'side': side, 'score': score_n}
        total_score += score_n
    
    return total_score, scores_by_n, overlapping_ns


def format_submission_value(val: float) -> str:
    """Format a value for submission (add 's' prefix)."""
    return f"s{val}"


def create_submission_df(trees_by_n: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Create a submission DataFrame from per-N tree data."""
    all_rows = []
    for n in range(1, 201):
        if n in trees_by_n:
            all_rows.append(trees_by_n[n])
    return pd.concat(all_rows, ignore_index=True)


def save_submission(df: pd.DataFrame, filepath: str):
    """Save a submission DataFrame to CSV."""
    df.to_csv(filepath, index=False)


def find_all_submission_csvs(base_path: str) -> List[str]:
    """Find all CSV files that could be submissions in a directory tree."""
    csv_files = []
    for root, dirs, files in os.walk(base_path):
        for f in files:
            if f.endswith('.csv') and 'sample' not in f.lower():
                csv_files.append(os.path.join(root, f))
    return csv_files


def is_valid_submission(df: pd.DataFrame) -> bool:
    """Check if a DataFrame is a valid submission format."""
    required_cols = {'id', 'x', 'y', 'deg'}
    if not required_cols.issubset(df.columns):
        return False
    # Check if it has the expected number of rows (20100 for N=1 to 200)
    expected_rows = sum(range(1, 201))  # 20100
    if len(df) != expected_rows:
        return False
    return True
