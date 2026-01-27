"""
Jostle Algorithm for 2D irregular packing.

This is a fundamentally different approach from SA:
1. Start from current best solution
2. Apply small random perturbations to all trees
3. Remove overlaps by pushing trees apart
4. Compact the configuration toward center
5. Repeat until no improvement
"""
import pandas as pd
import numpy as np
from shapely import Polygon
from shapely.affinity import rotate, translate
from numba import njit
import math
import json
import random

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def get_tree_polygon(x, y, angle):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = rotate(poly, angle, origin=(0, 0), use_radians=False)
    poly = translate(poly, x, y)
    return poly

def check_overlap_pair(x1, y1, a1, x2, y2, a2):
    """Check if two trees overlap and return overlap amount."""
    p1 = get_tree_polygon(x1, y1, a1)
    p2 = get_tree_polygon(x2, y2, a2)
    if p1.intersects(p2):
        if not p1.touches(p2):
            return p1.intersection(p2).area
    return 0

def check_any_overlap(xs, ys, angles):
    """Check if any trees overlap."""
    n = len(xs)
    for i in range(n):
        for j in range(i+1, n):
            if check_overlap_pair(xs[i], ys[i], angles[i], xs[j], ys[j], angles[j]) > 1e-10:
                return True
    return False

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

def remove_overlaps(xs, ys, angles, max_iterations=100):
    """Push overlapping trees apart."""
    n = len(xs)
    xs = list(xs)
    ys = list(ys)
    
    for iteration in range(max_iterations):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                overlap = check_overlap_pair(xs[i], ys[i], angles[i], xs[j], ys[j], angles[j])
                if overlap > 1e-10:
                    # Push apart along the line connecting centers
                    dx = xs[j] - xs[i]
                    dy = ys[j] - ys[i]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 0.001:
                        dx, dy = random.uniform(-1, 1), random.uniform(-1, 1)
                        dist = math.sqrt(dx*dx + dy*dy)
                    
                    # Normalize and push
                    push = 0.02  # Push amount
                    dx /= dist
                    dy /= dist
                    xs[i] -= dx * push
                    ys[i] -= dy * push
                    xs[j] += dx * push
                    ys[j] += dy * push
                    moved = True
        
        if not moved:
            break
    
    return np.array(xs), np.array(ys)

def compact_configuration(xs, ys, angles, step=0.01, max_iterations=50):
    """Move all trees toward center to minimize bounding box."""
    xs = np.array(xs)
    ys = np.array(ys)
    n = len(xs)
    
    for iteration in range(max_iterations):
        # Calculate centroid
        cx = np.mean(xs)
        cy = np.mean(ys)
        
        improved = False
        for i in range(n):
            # Try moving toward center
            dx = cx - xs[i]
            dy = cy - ys[i]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 0.001:
                continue
            
            dx /= dist
            dy /= dist
            
            # Try small step toward center
            new_x = xs[i] + dx * step
            new_y = ys[i] + dy * step
            
            # Check if this creates overlap
            has_overlap = False
            for j in range(n):
                if i != j:
                    if check_overlap_pair(new_x, new_y, angles[i], xs[j], ys[j], angles[j]) > 1e-10:
                        has_overlap = True
                        break
            
            if not has_overlap:
                # Check if score improves
                old_xs = xs.copy()
                xs[i] = new_x
                ys[i] = new_y
                new_score = compute_bbox_score(xs, ys, np.array(angles), TX, TY)
                old_score = compute_bbox_score(old_xs, ys, np.array(angles), TX, TY)
                
                if new_score < old_score:
                    improved = True
                else:
                    xs[i] = old_xs[i]
        
        if not improved:
            break
    
    return xs, ys

def jostle_algorithm(n, initial_xs, initial_ys, initial_angles, iterations=100, perturbation=0.01):
    """
    Jostle algorithm for 2D irregular packing.
    """
    xs = np.array(initial_xs)
    ys = np.array(initial_ys)
    angles = np.array(initial_angles)
    
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_angles = angles.copy()
    best_score = compute_bbox_score(xs, ys, angles, TX, TY)
    
    for iteration in range(iterations):
        # Jostle: apply small random perturbations
        new_xs = xs.copy()
        new_ys = ys.copy()
        new_angles = angles.copy()
        
        for i in range(n):
            new_xs[i] += np.random.uniform(-perturbation, perturbation)
            new_ys[i] += np.random.uniform(-perturbation, perturbation)
            new_angles[i] += np.random.uniform(-2, 2)
            new_angles[i] = new_angles[i] % 360
        
        # Remove overlaps
        new_xs, new_ys = remove_overlaps(new_xs, new_ys, new_angles)
        
        # Compact
        new_xs, new_ys = compact_configuration(new_xs, new_ys, new_angles)
        
        # Check if valid and better
        if not check_any_overlap(new_xs, new_ys, new_angles):
            score = compute_bbox_score(new_xs, new_ys, new_angles, TX, TY)
            if score < best_score:
                best_score = score
                best_xs = new_xs.copy()
                best_ys = new_ys.copy()
                best_angles = new_angles.copy()
                print(f"  Iteration {iteration}: New best {score:.6f}")
    
    return best_xs, best_ys, best_angles, best_score

def strip(v):
    return float(str(v).replace("s", ""))

if __name__ == "__main__":
    print("=" * 70)
    print("Jostle Algorithm for Tree Packing")
    print("=" * 70)
    
    # Load baseline
    df = pd.read_csv('/home/submission/submission.csv')
    df['N'] = df['id'].str.split('_').str[0].astype(int)
    
    # Test on a few N values
    test_ns = [10, 20, 50, 100]
    improvements = []
    
    for n in test_ns:
        g = df[df['N'] == n]
        xs = np.array([strip(v) for v in g['x']])
        ys = np.array([strip(v) for v in g['y']])
        angles = np.array([strip(v) for v in g['deg']])
        
        baseline_score = compute_bbox_score(xs, ys, angles, TX, TY)
        print(f"\nN={n}: Baseline = {baseline_score:.6f}")
        
        # Run jostle algorithm
        best_xs, best_ys, best_angles, best_score = jostle_algorithm(
            n, xs, ys, angles, iterations=50, perturbation=0.02
        )
        
        improvement = baseline_score - best_score
        if improvement > 0.0001:
            improvements.append((n, improvement))
            print(f"  ✓ IMPROVEMENT: {improvement:.6f}")
        else:
            print(f"  ✗ No improvement (best: {best_score:.6f})")
    
    print("\n" + "=" * 70)
    if improvements:
        print(f"Found {len(improvements)} improvements:")
        for n, imp in improvements:
            print(f"  N={n}: +{imp:.6f}")
    else:
        print("No improvements found")
    
    # Save metrics
    metrics = {
        'cv_score': 70.316492,
        'baseline_score': 70.316492,
        'improvement': sum(imp for _, imp in improvements) if improvements else 0,
        'num_improvements': len(improvements),
        'notes': f"Jostle algorithm tested on N={test_ns}. Found {len(improvements)} improvements."
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
