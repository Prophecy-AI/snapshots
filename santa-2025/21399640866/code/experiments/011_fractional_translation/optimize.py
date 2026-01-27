"""
Fractional Translation Optimization
Try tiny position adjustments to find improvements that SA misses.
Uses high-precision arithmetic and strict overlap validation.
"""

import sys
import os
sys.path.insert(0, '/home/code')

import numpy as np
import pandas as pd
import json
import time
from decimal import Decimal, getcontext
from shapely.geometry import Polygon

from code.tree_geometry import TX, TY, calculate_score, get_tree_vertices_numba
from code.utils import parse_submission, save_submission

# Set high precision
getcontext().prec = 30
SCALE = 10**18

# Minimum improvement threshold - must be significant to avoid precision issues
MIN_IMPROVEMENT = 0.001

# Translation steps to try (in decreasing order)
TRANSLATION_STEPS = [0.01, 0.005, 0.002, 0.001]

def get_tree_polygon_highprec(x, y, angle_deg):
    """Get tree polygon with high-precision integer coordinates."""
    rx, ry = get_tree_vertices_numba(x, y, angle_deg)
    
    # Scale to integers for exact arithmetic
    coords = []
    for xi, yi in zip(rx, ry):
        xi_int = int(Decimal(str(xi)) * SCALE)
        yi_int = int(Decimal(str(yi)) * SCALE)
        coords.append((xi_int, yi_int))
    
    return Polygon(coords)

def validate_no_overlap_strict(trees):
    """Validate no overlaps using integer arithmetic for precision."""
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

def calculate_bbox_score(trees):
    """Calculate bounding box score for a set of trees."""
    if len(trees) == 0:
        return 0.0
    
    all_xs = []
    all_ys = []
    for x, y, angle in trees:
        rx, ry = get_tree_vertices_numba(x, y, angle)
        all_xs.extend(rx)
        all_ys.extend(ry)
    
    width = max(all_xs) - min(all_xs)
    height = max(all_ys) - min(all_ys)
    side = max(width, height)
    n = len(trees)
    return (side ** 2) / n

def try_fractional_translation_single_n(trees, max_iterations=3):
    """
    Try fractional translations to improve score for a single N.
    Returns improved configuration and improvement amount.
    """
    n = len(trees)
    current_trees = [list(t) for t in trees]  # Make mutable copies
    current_score = calculate_bbox_score(current_trees)
    initial_score = current_score
    
    for iteration in range(max_iterations):
        improved_this_iteration = False
        
        for step in TRANSLATION_STEPS:
            for tree_idx in range(n):
                # Try all 8 directions + no move
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        
                        # Try moving tree
                        new_trees = [list(t) for t in current_trees]
                        new_trees[tree_idx][0] += dx
                        new_trees[tree_idx][1] += dy
                        
                        # Check validity
                        if not validate_no_overlap_strict(new_trees):
                            continue
                        
                        # Check score
                        new_score = calculate_bbox_score(new_trees)
                        
                        # Only accept if improvement is significant
                        if new_score < current_score - MIN_IMPROVEMENT:
                            current_trees = new_trees
                            current_score = new_score
                            improved_this_iteration = True
        
        if not improved_this_iteration:
            break
    
    total_improvement = initial_score - current_score
    return [(t[0], t[1], t[2]) for t in current_trees], total_improvement

def main():
    print("=" * 70)
    print("FRACTIONAL TRANSLATION OPTIMIZATION")
    print("=" * 70)
    
    start_time = time.time()
    
    # Load the best ensemble (exp_010)
    print("\nLoading exp_010 submission...")
    exp010_df = pd.read_csv('/home/code/experiments/010_safe_ensemble/submission.csv')
    exp010_configs = parse_submission(exp010_df)
    
    # Also load baseline for fallback
    baseline_df = pd.read_csv('/home/code/experiments/001_valid_baseline/submission.csv')
    baseline_configs = parse_submission(baseline_df)
    
    # Calculate initial scores per N
    initial_scores = {}
    for n in range(1, 201):
        initial_scores[n] = calculate_score(exp010_configs[n])
    
    initial_total = sum(initial_scores.values())
    print(f"Initial total score (exp_010): {initial_total:.6f}")
    
    # Try fractional translation on each N
    print(f"\nApplying fractional translation...")
    print(f"Translation steps: {TRANSLATION_STEPS}")
    print(f"Min improvement threshold: {MIN_IMPROVEMENT}")
    
    improved_configs = {}
    improvements = []
    
    for n in range(1, 201):
        config = exp010_configs[n]
        
        # Try fractional translation
        improved_trees, improvement = try_fractional_translation_single_n(config, max_iterations=2)
        
        if improvement >= MIN_IMPROVEMENT:
            # Validate the improvement
            if validate_no_overlap_strict(improved_trees):
                improved_configs[n] = improved_trees
                improvements.append((n, improvement))
                print(f"N={n}: Improved by {improvement:.6f}")
            else:
                # Fall back to original
                improved_configs[n] = config
        else:
            improved_configs[n] = config
        
        if n % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {n}/200 ({elapsed:.1f}s)")
    
    # Calculate final score
    final_total = sum(calculate_score(improved_configs[n]) for n in range(1, 201))
    total_improvement = initial_total - final_total
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Initial total (exp_010): {initial_total:.6f}")
    print(f"Final total: {final_total:.6f}")
    print(f"Total improvement: {total_improvement:.6f}")
    print(f"N values improved: {len(improvements)}")
    
    if improvements:
        print("\nTop improvements:")
        for n, imp in sorted(improvements, key=lambda x: -x[1])[:10]:
            print(f"  N={n}: {imp:.6f}")
    
    # Final validation
    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)
    
    invalid_n = []
    for n in range(1, 201):
        config = improved_configs[n]
        if len(config) != n:
            print(f"N={n}: Wrong number of trees")
            invalid_n.append(n)
            continue
        if not validate_no_overlap_strict(config):
            print(f"N={n}: Overlap detected - falling back to exp_010")
            improved_configs[n] = exp010_configs[n]
            invalid_n.append(n)
    
    if not invalid_n:
        print("✅ All configurations valid!")
    else:
        print(f"⚠️ Fell back for {len(invalid_n)} N values")
        # Recalculate
        final_total = sum(calculate_score(improved_configs[n]) for n in range(1, 201))
        total_improvement = initial_total - final_total
        print(f"Updated final total: {final_total:.6f}")
    
    # Save submission
    save_submission(improved_configs, 'submission.csv')
    print("\nSaved submission.csv")
    
    # Save metrics
    metrics = {
        'cv_score': final_total,
        'initial_score': initial_total,
        'improvement': total_improvement,
        'num_improvements': len(improvements),
        'translation_steps': TRANSLATION_STEPS,
        'min_improvement_threshold': MIN_IMPROVEMENT,
        'notes': 'Fractional translation optimization on exp_010 ensemble'
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nFinal CV Score: {final_total:.6f}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    
    # Copy to submission folder
    import shutil
    shutil.copy('submission.csv', '/home/submission/submission.csv')
    print("Copied submission to /home/submission/")
    
    return final_total

if __name__ == '__main__':
    os.chdir('/home/code/experiments/011_fractional_translation')
    main()
