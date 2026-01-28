"""
Jostle Algorithm for Irregular Packing
- Different from SA: uses COMPACTION after each perturbation
- Compaction moves all trees toward center while avoiding overlaps
- Has been shown to outperform pure SA in irregular packing literature
"""
import numpy as np
import pandas as pd
from numba import njit
import math
import time
import json
from shapely.geometry import Polygon
from shapely import affinity

# Tree geometry
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

@njit
def compute_bbox_score(xs, ys, angles, tx, ty):
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

@njit
def get_centroid(xs, ys):
    """Get centroid of all tree positions."""
    return np.mean(xs), np.mean(ys)

def get_tree_polygon(x, y, angle):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = affinity.rotate(poly, angle, origin=(0, 0))
    poly = affinity.translate(poly, x, y)
    return poly

def check_single_overlap(xs, ys, angles, idx):
    """Check if tree at idx overlaps with any other tree."""
    n = len(xs)
    poly_i = get_tree_polygon(xs[idx], ys[idx], angles[idx])
    for j in range(n):
        if j != idx:
            poly_j = get_tree_polygon(xs[j], ys[j], angles[j])
            if poly_i.intersects(poly_j) and not poly_i.touches(poly_j):
                return True
    return False

def check_all_overlaps(xs, ys, angles):
    """Check for any overlaps between trees."""
    n = len(xs)
    polys = [get_tree_polygon(xs[i], ys[i], angles[i]) for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                return True
    return False

def compact_toward_center(xs, ys, angles, step=0.01, max_iterations=100):
    """
    Compact all trees toward the centroid while avoiding overlaps.
    This is the KEY difference from SA - we always try to compact.
    """
    n = len(xs)
    cx, cy = get_centroid(xs, ys)
    
    for iteration in range(max_iterations):
        moved = False
        
        for i in range(n):
            # Direction toward centroid
            dx = cx - xs[i]
            dy = cy - ys[i]
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < 0.01:
                continue
            
            # Normalize and scale
            dx = dx / dist * step
            dy = dy / dist * step
            
            # Try to move toward center
            old_x, old_y = xs[i], ys[i]
            xs[i] += dx
            ys[i] += dy
            
            # Check for overlap
            if check_single_overlap(xs, ys, angles, i):
                xs[i], ys[i] = old_x, old_y
            else:
                moved = True
        
        if not moved:
            break
    
    return xs, ys, angles

def jostle_algorithm(xs, ys, angles, n_iterations=1000, perturb_size=0.05, angle_perturb=5.0):
    """
    Jostle algorithm:
    1. Select a random tree
    2. Apply small perturbation
    3. COMPACT all trees toward center
    4. If no overlap and score improved, keep
    """
    n = len(xs)
    xs = xs.copy()
    ys = ys.copy()
    angles = angles.copy()
    
    current_score = compute_bbox_score(xs, ys, angles, TX, TY)
    best_score = current_score
    best_xs, best_ys, best_angles = xs.copy(), ys.copy(), angles.copy()
    
    improvements_found = 0
    
    for iteration in range(n_iterations):
        # Select random tree
        i = np.random.randint(n)
        
        # Save current state
        old_xs = xs.copy()
        old_ys = ys.copy()
        old_angles = angles.copy()
        
        # Apply perturbation to selected tree
        xs[i] += np.random.uniform(-perturb_size, perturb_size)
        ys[i] += np.random.uniform(-perturb_size, perturb_size)
        angles[i] += np.random.uniform(-angle_perturb, angle_perturb)
        angles[i] = angles[i] % 360
        
        # Check if perturbation causes overlap
        if check_single_overlap(xs, ys, angles, i):
            xs, ys, angles = old_xs, old_ys, old_angles
            continue
        
        # COMPACT: Move all trees toward center
        xs, ys, angles = compact_toward_center(xs, ys, angles, step=0.005, max_iterations=50)
        
        # Check for any overlaps after compaction
        if check_all_overlaps(xs, ys, angles):
            xs, ys, angles = old_xs, old_ys, old_angles
            continue
        
        # Compute new score
        new_score = compute_bbox_score(xs, ys, angles, TX, TY)
        
        # Accept if improved
        if new_score < current_score - 1e-10:
            current_score = new_score
            if new_score < best_score - 1e-10:
                best_score = new_score
                best_xs, best_ys, best_angles = xs.copy(), ys.copy(), angles.copy()
                improvements_found += 1
        else:
            # Reject - restore
            xs, ys, angles = old_xs, old_ys, old_angles
    
    return best_xs, best_ys, best_angles, best_score, improvements_found

def strip(v):
    return float(str(v).replace("s", ""))

def df_to_arrays(df):
    xs = np.array([strip(v) for v in df['x']])
    ys = np.array([strip(v) for v in df['y']])
    angles = np.array([strip(v) for v in df['deg']])
    return xs, ys, angles

def main():
    print("="*70)
    print("Jostle Algorithm for Irregular Packing")
    print("="*70)
    
    # Load baseline
    baseline_df = pd.read_csv('/home/submission/submission.csv')
    baseline_df['N'] = baseline_df['id'].str.split('_').str[0].astype(int)
    
    baseline_scores = {}
    baseline_configs = {}
    for n in range(1, 201):
        g = baseline_df[baseline_df['N'] == n]
        xs, ys, angles = df_to_arrays(g)
        baseline_scores[n] = compute_bbox_score(xs, ys, angles, TX, TY)
        baseline_configs[n] = (xs.copy(), ys.copy(), angles.copy())
    
    baseline_total = sum(baseline_scores.values())
    print(f"Baseline total: {baseline_total:.6f}")
    
    # Test on small N values first (as required by strategy)
    test_ns = [10, 20, 30]
    improvements = []
    start_time = time.time()
    
    print("\n--- Testing on small N values first ---")
    for n in test_ns:
        print(f"\nTesting N={n}...")
        xs, ys, angles = baseline_configs[n]
        
        # Run jostle algorithm
        t0 = time.time()
        best_xs, best_ys, best_angles, best_score, num_improvements = jostle_algorithm(
            xs, ys, angles, 
            n_iterations=2000,
            perturb_size=0.03,
            angle_perturb=3.0
        )
        elapsed = time.time() - t0
        
        improvement = baseline_scores[n] - best_score
        print(f"  Jostle completed in {elapsed:.1f}s, found {num_improvements} improvements")
        print(f"  Baseline: {baseline_scores[n]:.6f}, Jostle: {best_score:.6f}")
        
        if improvement > 0.0001:
            improvements.append((n, improvement, best_score, baseline_scores[n]))
            print(f"  ✅ IMPROVED by {improvement:.6f}")
        else:
            print(f"  No improvement")
    
    # If any improvement found on small N, run on more N values
    if improvements:
        print("\n--- Improvements found! Running on more N values ---")
        for n in range(5, 51):
            if n in test_ns:
                continue
            
            xs, ys, angles = baseline_configs[n]
            best_xs, best_ys, best_angles, best_score, num_improvements = jostle_algorithm(
                xs, ys, angles,
                n_iterations=1000,
                perturb_size=0.03,
                angle_perturb=3.0
            )
            
            improvement = baseline_scores[n] - best_score
            if improvement > 0.0001:
                improvements.append((n, improvement, best_score, baseline_scores[n]))
                print(f"  N={n}: IMPROVED by {improvement:.6f}")
    
    total_elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"Jostle Algorithm Complete")
    print(f"  Total elapsed time: {total_elapsed:.1f}s")
    print(f"  Improvements found: {len(improvements)}")
    
    if improvements:
        total_improvement = sum(imp for _, imp, _, _ in improvements)
        print(f"  Total improvement: {total_improvement:.6f}")
        print("\nImproved N values:")
        for n, imp, new_score, old_score in sorted(improvements, key=lambda x: -x[1]):
            print(f"  N={n}: {old_score:.6f} -> {new_score:.6f} (+{imp:.6f})")
    else:
        print("  No improvements found - baseline is at local optimum even for jostle")
    
    print("="*70)
    
    # Save results
    results = {
        'improvements': [(n, imp, new_s, old_s) for n, imp, new_s, old_s in improvements],
        'total_improvement': sum(imp for _, imp, _, _ in improvements) if improvements else 0,
        'elapsed_time': total_elapsed
    }
    
    with open('jostle_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return improvements

if __name__ == "__main__":
    improvements = main()
