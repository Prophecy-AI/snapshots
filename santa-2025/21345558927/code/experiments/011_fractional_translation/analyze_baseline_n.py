"""
Analyze which N values are still at baseline and try to find improvements.
"""

import sys
import os
sys.path.insert(0, '/home/code')

import numpy as np
import pandas as pd
import json
import glob
from decimal import Decimal, getcontext
from shapely.geometry import Polygon

from code.tree_geometry import TX, TY, calculate_score, get_tree_vertices_numba
from code.utils import parse_submission, save_submission

getcontext().prec = 30
SCALE = 10**18

def get_tree_polygon_highprec(x, y, angle_deg):
    """Get tree polygon with high-precision integer coordinates."""
    rx, ry = get_tree_vertices_numba(x, y, angle_deg)
    coords = []
    for xi, yi in zip(rx, ry):
        xi_int = int(Decimal(str(xi)) * SCALE)
        yi_int = int(Decimal(str(yi)) * SCALE)
        coords.append((xi_int, yi_int))
    return Polygon(coords)

def validate_no_overlap_strict(trees):
    """Validate no overlaps using integer arithmetic."""
    if len(trees) <= 1:
        return True
    
    polygons = []
    for x, y, angle in trees:
        poly = get_tree_polygon_highprec(x, y, angle)
        if not poly.is_valid:
            return False
        polygons.append(poly)
    
    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            if polygons[i].intersects(polygons[j]):
                if not polygons[i].touches(polygons[j]):
                    inter = polygons[i].intersection(polygons[j])
                    if inter.area > 0:
                        return False
    return True

def main():
    print("=" * 70)
    print("ANALYZE BASELINE N VALUES")
    print("=" * 70)
    
    # Load baseline and exp_010
    baseline_df = pd.read_csv('/home/code/experiments/001_valid_baseline/submission.csv')
    baseline_configs = parse_submission(baseline_df)
    
    exp010_df = pd.read_csv('/home/code/experiments/010_safe_ensemble/submission.csv')
    exp010_configs = parse_submission(exp010_df)
    
    # Calculate scores
    baseline_scores = {n: calculate_score(baseline_configs[n]) for n in range(1, 201)}
    exp010_scores = {n: calculate_score(exp010_configs[n]) for n in range(1, 201)}
    
    # Find N values still at baseline
    baseline_n = []
    improved_n = []
    for n in range(1, 201):
        diff = baseline_scores[n] - exp010_scores[n]
        if abs(diff) < 1e-9:
            baseline_n.append(n)
        else:
            improved_n.append((n, diff))
    
    print(f"\nN values still at baseline: {len(baseline_n)}")
    print(f"N values improved: {len(improved_n)}")
    
    # Calculate total score contribution of baseline N values
    baseline_n_total = sum(baseline_scores[n] for n in baseline_n)
    print(f"\nTotal score from baseline N values: {baseline_n_total:.6f}")
    print(f"This is {baseline_n_total / sum(baseline_scores.values()) * 100:.1f}% of total score")
    
    # Show top baseline N values by score
    print("\nTop 20 baseline N values by score contribution:")
    baseline_n_scores = [(n, baseline_scores[n]) for n in baseline_n]
    for n, score in sorted(baseline_n_scores, key=lambda x: -x[1])[:20]:
        print(f"  N={n}: {score:.6f}")
    
    # Show improved N values
    print("\nTop 10 improved N values:")
    for n, diff in sorted(improved_n, key=lambda x: -x[1])[:10]:
        print(f"  N={n}: improved by {diff:.6f}")
    
    return baseline_n

if __name__ == '__main__':
    os.chdir('/home/code/experiments/011_fractional_translation')
    baseline_n = main()
