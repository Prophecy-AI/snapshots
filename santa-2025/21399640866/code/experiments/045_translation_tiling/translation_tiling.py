"""
Translation-Based Tiling - Based on egortrushin kernel approach.
Optimize a small base pattern (2 trees) and tile it to create larger N.
"""
import numpy as np
import pandas as pd
from numba import njit
import math
import time
import json
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

getcontext().prec = 25
scale_factor = Decimal("1e15")

# Tree geometry for fast scoring
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

class ChristmasTree:
    """Represents a single, rotatable Christmas tree."""
    def __init__(self, center_x="0", center_y="0", angle="0"):
        self.center_x = Decimal(str(center_x))
        self.center_y = Decimal(str(center_y))
        self.angle = Decimal(str(angle))

        trunk_w = Decimal("0.15")
        trunk_h = Decimal("0.2")
        base_w = Decimal("0.7")
        mid_w = Decimal("0.4")
        top_w = Decimal("0.25")
        tip_y = Decimal("0.8")
        tier_1_y = Decimal("0.5")
        tier_2_y = Decimal("0.25")
        base_y = Decimal("0.0")
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon([
            (Decimal("0.0") * scale_factor, tip_y * scale_factor),
            (top_w / Decimal("2") * scale_factor, tier_1_y * scale_factor),
            (top_w / Decimal("4") * scale_factor, tier_1_y * scale_factor),
            (mid_w / Decimal("2") * scale_factor, tier_2_y * scale_factor),
            (mid_w / Decimal("4") * scale_factor, tier_2_y * scale_factor),
            (base_w / Decimal("2") * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal("2") * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal("2") * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal("2")) * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal("2")) * scale_factor, base_y * scale_factor),
            (-(base_w / Decimal("2")) * scale_factor, base_y * scale_factor),
            (-(mid_w / Decimal("4")) * scale_factor, tier_2_y * scale_factor),
            (-(mid_w / Decimal("2")) * scale_factor, tier_2_y * scale_factor),
            (-(top_w / Decimal("4")) * scale_factor, tier_1_y * scale_factor),
            (-(top_w / Decimal("2")) * scale_factor, tier_1_y * scale_factor),
        ])
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated,
            xoff=float(self.center_x * scale_factor),
            yoff=float(self.center_y * scale_factor),
        )

    def clone(self):
        return ChristmasTree(str(self.center_x), str(self.center_y), str(self.angle))

def has_collision(trees):
    """Check for collisions between trees."""
    if len(trees) <= 1:
        return False
    for i, tree1 in enumerate(trees):
        for j, tree2 in enumerate(trees):
            if i < j:
                if tree1.polygon.intersects(tree2.polygon) and not tree1.polygon.touches(tree2.polygon):
                    return True
    return False

def calculate_score(trees):
    """Calculate bounding box score."""
    xys = np.concatenate([np.asarray(t.polygon.exterior.xy).T / 1e15 for t in trees])
    min_x, min_y = xys.min(axis=0)
    max_x, max_y = xys.max(axis=0)
    score = max(max_x - min_x, max_y - min_y) ** 2 / len(trees)
    return score

def translate_trees(base_trees, lengthx, lengthy, nx, ny):
    """Create tiled configuration from base pattern."""
    trees = []
    for tree in base_trees:
        for x in range(nx):
            for y in range(ny):
                trees.append(
                    ChristmasTree(
                        center_x=tree.center_x + Decimal(str(x * lengthx)),
                        center_y=tree.center_y + Decimal(str(y * lengthy)),
                        angle=tree.angle,
                    )
                )
    return trees

def get_min_translation(base_trees, nx, ny, delta=0.01):
    """Find minimum translation distances that don't cause overlaps."""
    # Start with bounding box of base pattern
    xys = np.concatenate([np.asarray(t.polygon.exterior.xy).T / 1e15 for t in base_trees])
    min_x, min_y = xys.min(axis=0)
    max_x, max_y = xys.max(axis=0)
    length = max(max_x - min_x, max_y - min_y)
    
    lengthx = length
    lengthy = length
    
    # Shrink lengthx until collision
    while True:
        trees = translate_trees(base_trees, lengthx - delta, lengthy, nx, ny)
        if has_collision(trees):
            break
        lengthx -= delta
    
    # Shrink lengthy until collision
    while True:
        trees = translate_trees(base_trees, lengthx, lengthy - delta, nx, ny)
        if has_collision(trees):
            break
        lengthy -= delta
    
    return lengthx, lengthy

@njit
def compute_bbox_score_fast(xs, ys, angles, tx, ty):
    """Fast bounding box score computation."""
    n = len(xs)
    V = len(tx)
    mnx = 1e300
    mny = 1e300
    mxx = -1e300
    mxy = -1e300
    for i in range(n):
        r = angles[i] * math.pi / 180.0
        c = math.cos(r)
        s = math.sin(r)
        xi = xs[i]
        yi = ys[i]
        for j in range(V):
            X = c * tx[j] - s * ty[j] + xi
            Y = s * tx[j] + c * ty[j] + yi
            if X < mnx: mnx = X
            if X > mxx: mxx = X
            if Y < mny: mny = Y
            if Y > mxy: mxy = Y
    side = max(mxx - mnx, mxy - mny)
    return side * side / n

def strip(v):
    return float(str(v).replace("s", ""))

def df_to_arrays(df):
    xs = np.array([strip(v) for v in df['x']])
    ys = np.array([strip(v) for v in df['y']])
    angles = np.array([strip(v) for v in df['deg']])
    return xs, ys, angles

def main():
    print("="*70)
    print("Translation-Based Tiling Optimization")
    print("="*70)
    
    # Load baseline
    baseline_df = pd.read_csv('/home/submission/submission.csv')
    baseline_df['N'] = baseline_df['id'].str.split('_').str[0].astype(int)
    
    baseline_scores = {}
    for n in range(1, 201):
        g = baseline_df[baseline_df['N'] == n]
        xs, ys, angles = df_to_arrays(g)
        baseline_scores[n] = compute_bbox_score_fast(xs, ys, angles, TX, TY)
    
    baseline_total = sum(baseline_scores.values())
    print(f"Baseline total: {baseline_total:.6f}")
    
    # Test the egortrushin base pattern
    # Initial trees from the kernel: [-2.93069232,-4.24856960,67], [-3.92971914, -4.16631769, 250.00]
    base_trees = [
        ChristmasTree(-2.93069232, -4.24856960, 67),
        ChristmasTree(-3.92971914, -4.16631769, 250.00)
    ]
    
    print("\nBase pattern (2 trees):")
    print(f"  Tree 1: ({float(base_trees[0].center_x):.4f}, {float(base_trees[0].center_y):.4f}, {float(base_trees[0].angle):.2f})")
    print(f"  Tree 2: ({float(base_trees[1].center_x):.4f}, {float(base_trees[1].center_y):.4f}, {float(base_trees[1].angle):.2f})")
    
    # Test on various N values that can be factored
    test_configs = [
        (36, 2, 3, 6),   # N=36 = 2*3*6
        (48, 2, 4, 6),   # N=48 = 2*4*6
        (72, 2, 4, 9),   # N=72 = 2*4*9
        (100, 2, 5, 10), # N=100 = 2*5*10
        (110, 2, 5, 11), # N=110 = 2*5*11
        (144, 2, 6, 12), # N=144 = 2*6*12
        (156, 2, 6, 13), # N=156 = 2*6*13
        (196, 2, 7, 14), # N=196 = 2*7*14
        (200, 2, 10, 10), # N=200 = 2*10*10
    ]
    
    improvements = []
    start_time = time.time()
    
    for target_n, base_n, nx, ny in test_configs:
        print(f"\nTesting N={target_n} (base={base_n}, grid={nx}x{ny})...")
        
        # Find minimum translation distances
        lengthx, lengthy = get_min_translation(base_trees, nx, ny, delta=0.005)
        print(f"  Translation: lengthx={lengthx:.4f}, lengthy={lengthy:.4f}")
        
        # Create tiled configuration
        trees = translate_trees(base_trees, lengthx, lengthy, nx, ny)
        actual_n = len(trees)
        
        if actual_n < target_n:
            print(f"  Not enough trees: got {actual_n}, need {target_n}")
            continue
        
        # Take first target_n trees
        trees = trees[:target_n]
        
        # Check for collisions
        if has_collision(trees):
            print(f"  Configuration has collisions!")
            continue
        
        # Calculate score
        score = calculate_score(trees)
        baseline = baseline_scores[target_n]
        improvement = baseline - score
        
        print(f"  Tiled score: {score:.6f}, Baseline: {baseline:.6f}")
        
        if improvement > 0.0001:
            improvements.append((target_n, improvement, score, baseline, trees))
            print(f"  ✅ IMPROVED by {improvement:.6f}")
        else:
            print(f"  No improvement (diff: {improvement:.6f})")
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"Translation-Based Tiling Complete")
    print(f"  Elapsed time: {elapsed:.1f}s")
    print(f"  Improvements found: {len(improvements)}")
    
    if improvements:
        total_improvement = sum(imp for _, imp, _, _, _ in improvements)
        print(f"  Total improvement: {total_improvement:.6f}")
        print("\nImproved N values:")
        for n, imp, new_score, old_score, _ in sorted(improvements, key=lambda x: -x[1]):
            print(f"  N={n}: {old_score:.6f} -> {new_score:.6f} (+{imp:.6f})")
    else:
        print("  No improvements found")
    
    print("="*70)
    
    # Save results
    results = {
        'improvements': [(n, imp, new_s, old_s) for n, imp, new_s, old_s, _ in improvements],
        'total_improvement': sum(imp for _, imp, _, _, _ in improvements) if improvements else 0,
        'elapsed_time': elapsed
    }
    
    with open('translation_tiling_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return improvements

if __name__ == "__main__":
    improvements = main()
