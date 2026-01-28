"""
Crystallization Pattern Generator for Tree Packing
Based on "Why Not" kernel analysis:
- Blue trees: pointing up (0° ± 90°)
- Pink trees: pointing down (180° ± 90°)
- Optimal lattice offsets between blue-pink pairs
"""
import numpy as np
from numba import njit
import math
from shapely import Polygon
from shapely.affinity import rotate, translate
import pandas as pd
import json
import time

# Tree geometry
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def get_tree_polygon(x, y, angle):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = rotate(poly, angle, origin=(0, 0), use_radians=False)
    poly = translate(poly, x, y)
    return poly

def check_overlaps(xs, ys, angles):
    """Check if any trees overlap."""
    n = len(xs)
    if n <= 1:
        return False
    polygons = [get_tree_polygon(xs[i], ys[i], angles[i]) for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if polygons[i].intersects(polygons[j]):
                if not polygons[i].touches(polygons[j]):
                    area = polygons[i].intersection(polygons[j]).area
                    if area > 1e-12:
                        return True
    return False

@njit
def compute_bbox_score(xs, ys, angles, tx, ty):
    """Compute bounding box score."""
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

def generate_crystallization_pattern(n, lattice_spacing=0.6, angle_blue=45, angle_pink=225):
    """
    Generate a configuration using lattice-based crystallization.
    
    Blue trees: pointing up (angle_blue)
    Pink trees: pointing down (angle_pink = angle_blue + 180)
    
    Trees are placed in a grid pattern with alternating blue/pink.
    """
    # Calculate grid dimensions
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    xs = []
    ys = []
    angles = []
    
    tree_count = 0
    for row in range(rows):
        for col in range(cols):
            if tree_count >= n:
                break
            
            # Alternate blue and pink based on position
            is_blue = (row + col) % 2 == 0
            
            # Position with offset for pink trees
            x = col * lattice_spacing
            y = row * lattice_spacing
            
            # Small offset for interlocking
            if not is_blue:
                x += lattice_spacing * 0.1
                y += lattice_spacing * 0.05
            
            xs.append(x)
            ys.append(y)
            angles.append(angle_blue if is_blue else angle_pink)
            tree_count += 1
    
    return np.array(xs), np.array(ys), np.array(angles)

def optimize_lattice_spacing(n, baseline_score):
    """Find optimal lattice spacing for N trees."""
    best_score = float('inf')
    best_config = None
    
    # Try different lattice spacings
    for spacing in np.linspace(0.4, 1.0, 31):
        # Try different base angles
        for base_angle in [0, 15, 30, 45, 60, 75, 90]:
            xs, ys, angles = generate_crystallization_pattern(
                n, 
                lattice_spacing=spacing,
                angle_blue=base_angle,
                angle_pink=base_angle + 180
            )
            
            # Check for overlaps
            if check_overlaps(list(xs), list(ys), list(angles)):
                continue
            
            score = compute_bbox_score(xs, ys, angles, TX, TY)
            if score < best_score:
                best_score = score
                best_config = (xs.copy(), ys.copy(), angles.copy(), spacing, base_angle)
    
    return best_score, best_config

def main():
    print("=" * 70)
    print("Crystallization Pattern Generator")
    print("=" * 70)
    
    # Load baseline
    baseline_df = pd.read_csv('/home/submission/submission.csv')
    baseline_df['N'] = baseline_df['id'].str.split('_').str[0].astype(int)
    
    # Test on small N values first
    test_ns = [10, 15, 20, 25, 30, 40, 50]
    
    results = []
    
    for n in test_ns:
        # Get baseline score
        g = baseline_df[baseline_df['N'] == n]
        baseline_xs = np.array([strip(v) for v in g['x']])
        baseline_ys = np.array([strip(v) for v in g['y']])
        baseline_angles = np.array([strip(v) for v in g['deg']])
        baseline_score = compute_bbox_score(baseline_xs, baseline_ys, baseline_angles, TX, TY)
        
        # Try crystallization pattern
        crystal_score, crystal_config = optimize_lattice_spacing(n, baseline_score)
        
        improvement = baseline_score - crystal_score
        pct_improvement = improvement / baseline_score * 100
        
        results.append({
            'n': n,
            'baseline': baseline_score,
            'crystal': crystal_score,
            'improvement': improvement,
            'pct': pct_improvement
        })
        
        if improvement > 0:
            print(f"N={n:3d}: baseline={baseline_score:.6f}, crystal={crystal_score:.6f}, improvement={improvement:.6f} ({pct_improvement:.2f}%) ✓")
        else:
            print(f"N={n:3d}: baseline={baseline_score:.6f}, crystal={crystal_score:.6f}, no improvement")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    improvements = [r for r in results if r['improvement'] > 0]
    print(f"  Improvements found: {len(improvements)} / {len(test_ns)}")
    if improvements:
        total_imp = sum(r['improvement'] for r in improvements)
        print(f"  Total improvement: {total_imp:.6f}")
    
    # Save results
    with open('crystal_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    results = main()
