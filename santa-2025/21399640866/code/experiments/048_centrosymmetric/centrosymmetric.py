"""
Centrosymmetric Placement - Place trees in symmetric pairs around center.
For even N: Tree i at (x, y, θ) and Tree i+N/2 at (-x, -y, θ+180°)
This exploits the tree's bilateral symmetry.
"""
import numpy as np
import pandas as pd
from numba import njit
import math
import time
import json
from shapely.geometry import Polygon
from shapely import affinity
from scipy.optimize import minimize

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

def get_tree_polygon(x, y, angle):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = affinity.rotate(poly, angle, origin=(0, 0))
    poly = affinity.translate(poly, x, y)
    return poly

def check_overlaps(xs, ys, angles):
    n = len(xs)
    polys = [get_tree_polygon(xs[i], ys[i], angles[i]) for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                return True
    return False

def generate_centrosymmetric_ring(n, radius, base_angle=0):
    """Generate centrosymmetric placement in a ring pattern."""
    if n == 1:
        return np.array([0.0]), np.array([0.0]), np.array([45.0])
    
    xs, ys, angles = [], [], []
    pairs = n // 2
    
    for i in range(pairs):
        # Position angle for this pair
        pos_angle = 2 * np.pi * i / pairs + base_angle
        
        x = radius * np.cos(pos_angle)
        y = radius * np.sin(pos_angle)
        tree_angle = np.degrees(pos_angle) + 90  # Point outward
        
        # First tree
        xs.append(x)
        ys.append(y)
        angles.append(tree_angle % 360)
        
        # Symmetric partner (180° rotation around origin)
        xs.append(-x)
        ys.append(-y)
        angles.append((tree_angle + 180) % 360)
    
    # Handle odd N - place center tree
    if n % 2 == 1:
        xs.append(0.0)
        ys.append(0.0)
        angles.append(45.0)
    
    return np.array(xs), np.array(ys), np.array(angles)

def generate_centrosymmetric_spiral(n, inner_radius=0.5, spacing=0.4):
    """Generate centrosymmetric placement in a spiral pattern."""
    if n == 1:
        return np.array([0.0]), np.array([0.0]), np.array([45.0])
    
    xs, ys, angles = [], [], []
    pairs = n // 2
    
    for i in range(pairs):
        # Spiral outward
        r = inner_radius + spacing * i
        pos_angle = 2 * np.pi * i / 5  # Golden angle approximation
        
        x = r * np.cos(pos_angle)
        y = r * np.sin(pos_angle)
        tree_angle = np.degrees(pos_angle) + 90
        
        # First tree
        xs.append(x)
        ys.append(y)
        angles.append(tree_angle % 360)
        
        # Symmetric partner
        xs.append(-x)
        ys.append(-y)
        angles.append((tree_angle + 180) % 360)
    
    if n % 2 == 1:
        xs.append(0.0)
        ys.append(0.0)
        angles.append(45.0)
    
    return np.array(xs), np.array(ys), np.array(angles)

def generate_centrosymmetric_grid(n, spacing=0.8):
    """Generate centrosymmetric placement in a grid pattern."""
    if n == 1:
        return np.array([0.0]), np.array([0.0]), np.array([45.0])
    
    xs, ys, angles = [], [], []
    pairs = n // 2
    
    # Create grid positions
    side = int(np.ceil(np.sqrt(pairs)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < pairs:
                x = (i - side/2 + 0.5) * spacing
                y = (j - side/2 + 0.5) * spacing
                positions.append((x, y))
    
    for i, (x, y) in enumerate(positions[:pairs]):
        tree_angle = 45.0  # Default angle
        
        # First tree
        xs.append(x)
        ys.append(y)
        angles.append(tree_angle)
        
        # Symmetric partner
        xs.append(-x)
        ys.append(-y)
        angles.append((tree_angle + 180) % 360)
    
    if n % 2 == 1:
        xs.append(0.0)
        ys.append(0.0)
        angles.append(45.0)
    
    return np.array(xs), np.array(ys), np.array(angles)

def optimize_centrosymmetric(n, generator_func, **kwargs):
    """Optimize centrosymmetric placement parameters."""
    best_score = float('inf')
    best_config = None
    
    # Try different parameter combinations
    for radius in np.linspace(0.3, 2.0, 20):
        for base_angle in np.linspace(0, np.pi/4, 10):
            try:
                if 'radius' in generator_func.__code__.co_varnames:
                    xs, ys, angles = generator_func(n, radius=radius, base_angle=base_angle)
                else:
                    xs, ys, angles = generator_func(n, **kwargs)
                
                # Check for overlaps
                if check_overlaps(xs, ys, angles):
                    continue
                
                score = compute_bbox_score(xs, ys, angles, TX, TY)
                if score < best_score:
                    best_score = score
                    best_config = (xs.copy(), ys.copy(), angles.copy())
            except:
                continue
    
    return best_score, best_config

def strip(v):
    return float(str(v).replace("s", ""))

def df_to_arrays(df):
    xs = np.array([strip(v) for v in df['x']])
    ys = np.array([strip(v) for v in df['y']])
    angles = np.array([strip(v) for v in df['deg']])
    return xs, ys, angles

def main():
    print("="*70)
    print("Centrosymmetric Placement Optimization")
    print("="*70)
    
    # Load baseline
    baseline_df = pd.read_csv('/home/submission/submission.csv')
    baseline_df['N'] = baseline_df['id'].str.split('_').str[0].astype(int)
    
    baseline_scores = {}
    for n in range(1, 201):
        g = baseline_df[baseline_df['N'] == n]
        xs, ys, angles = df_to_arrays(g)
        baseline_scores[n] = compute_bbox_score(xs, ys, angles, TX, TY)
    
    baseline_total = sum(baseline_scores.values())
    print(f"Baseline total: {baseline_total:.6f}")
    
    # Test centrosymmetric on various N values
    test_ns = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]
    improvements = []
    start_time = time.time()
    
    for n in test_ns:
        print(f"\nTesting N={n}...")
        
        best_score = baseline_scores[n]
        best_config = None
        best_method = None
        
        # Try ring pattern
        print(f"  Trying ring pattern...")
        for radius in np.linspace(0.4, 3.0, 30):
            for base_angle in np.linspace(0, np.pi/2, 20):
                xs, ys, angles = generate_centrosymmetric_ring(n, radius, base_angle)
                if not check_overlaps(xs, ys, angles):
                    score = compute_bbox_score(xs, ys, angles, TX, TY)
                    if score < best_score - 0.0001:
                        best_score = score
                        best_config = (xs.copy(), ys.copy(), angles.copy())
                        best_method = f"ring(r={radius:.2f}, a={base_angle:.2f})"
        
        # Try spiral pattern
        print(f"  Trying spiral pattern...")
        for inner_r in np.linspace(0.3, 1.0, 10):
            for spacing in np.linspace(0.3, 0.8, 10):
                try:
                    xs, ys, angles = generate_centrosymmetric_spiral(n, inner_r, spacing)
                    if not check_overlaps(xs, ys, angles):
                        score = compute_bbox_score(xs, ys, angles, TX, TY)
                        if score < best_score - 0.0001:
                            best_score = score
                            best_config = (xs.copy(), ys.copy(), angles.copy())
                            best_method = f"spiral(ir={inner_r:.2f}, s={spacing:.2f})"
                except:
                    continue
        
        # Try grid pattern
        print(f"  Trying grid pattern...")
        for spacing in np.linspace(0.6, 1.2, 15):
            try:
                xs, ys, angles = generate_centrosymmetric_grid(n, spacing)
                if not check_overlaps(xs, ys, angles):
                    score = compute_bbox_score(xs, ys, angles, TX, TY)
                    if score < best_score - 0.0001:
                        best_score = score
                        best_config = (xs.copy(), ys.copy(), angles.copy())
                        best_method = f"grid(s={spacing:.2f})"
            except:
                continue
        
        improvement = baseline_scores[n] - best_score
        if improvement > 0.0001:
            improvements.append((n, improvement, best_score, baseline_scores[n], best_method))
            print(f"  ✅ IMPROVED: {baseline_scores[n]:.6f} -> {best_score:.6f} (+{improvement:.6f}) via {best_method}")
        else:
            print(f"  No improvement: baseline={baseline_scores[n]:.6f}")
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"Centrosymmetric Placement Complete")
    print(f"  Elapsed time: {elapsed:.1f}s")
    print(f"  Improvements found: {len(improvements)}")
    
    if improvements:
        total_improvement = sum(imp for _, imp, _, _, _ in improvements)
        print(f"  Total improvement: {total_improvement:.6f}")
        print("\nImproved N values:")
        for n, imp, new_score, old_score, method in sorted(improvements, key=lambda x: -x[1]):
            print(f"  N={n}: {old_score:.6f} -> {new_score:.6f} (+{imp:.6f}) via {method}")
    else:
        print("  No improvements found")
    
    print("="*70)
    
    # Save results
    results = {
        'improvements': [(n, imp, new_s, old_s, method) for n, imp, new_s, old_s, method in improvements],
        'total_improvement': sum(imp for _, imp, _, _, _ in improvements) if improvements else 0,
        'elapsed_time': elapsed
    }
    
    with open('centrosymmetric_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return improvements

if __name__ == "__main__":
    improvements = main()
