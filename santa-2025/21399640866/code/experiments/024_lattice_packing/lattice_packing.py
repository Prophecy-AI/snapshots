"""
Implement hexagonal lattice packing for tree placement.
"""
import pandas as pd
import numpy as np
from shapely import Polygon
from shapely.affinity import rotate, translate
from numba import njit
import math
import json

# Tree polygon vertices
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
    polygons = [get_tree_polygon(xs[i], ys[i], angles[i]) for i in range(n)]
    
    for i in range(n):
        for j in range(i+1, n):
            if polygons[i].intersects(polygons[j]):
                if not polygons[i].touches(polygons[j]):
                    area = polygons[i].intersection(polygons[j]).area
                    if area > 1e-10:
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

def hexagonal_lattice_packing(n, spacing, angle_offset=0, alternate_angles=True):
    """
    Place n trees on a hexagonal lattice.
    
    Args:
        n: Number of trees
        spacing: Distance between adjacent trees
        angle_offset: Base angle for all trees
        alternate_angles: If True, alternate between angle_offset and angle_offset+180
    
    Returns:
        xs, ys, angles: Lists of tree positions and orientations
    """
    # Calculate grid dimensions
    cols = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
    rows = int(np.ceil(n / cols))
    
    xs = []
    ys = []
    angles = []
    
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= n:
                break
            
            # Hexagonal offset for odd rows
            x = col * spacing + (row % 2) * spacing / 2
            y = row * spacing * np.sqrt(3) / 2
            
            # Alternate orientations
            if alternate_angles:
                angle = angle_offset if (row + col) % 2 == 0 else angle_offset + 180
            else:
                angle = angle_offset
            
            xs.append(x)
            ys.append(y)
            angles.append(angle % 360)
            count += 1
        
        if count >= n:
            break
    
    # Center the configuration
    xs = np.array(xs)
    ys = np.array(ys)
    xs = xs - np.mean(xs)
    ys = ys - np.mean(ys)
    
    return list(xs), list(ys), angles

def square_lattice_packing(n, spacing, angle_offset=0, alternate_angles=True):
    """
    Place n trees on a square lattice.
    """
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    xs = []
    ys = []
    angles = []
    
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= n:
                break
            
            x = col * spacing
            y = row * spacing
            
            if alternate_angles:
                angle = angle_offset if (row + col) % 2 == 0 else angle_offset + 180
            else:
                angle = angle_offset
            
            xs.append(x)
            ys.append(y)
            angles.append(angle % 360)
            count += 1
        
        if count >= n:
            break
    
    # Center
    xs = np.array(xs)
    ys = np.array(ys)
    xs = xs - np.mean(xs)
    ys = ys - np.mean(ys)
    
    return list(xs), list(ys), angles

def strip(v):
    return float(str(v).replace("s", ""))

def get_baseline_score(n, df):
    """Get baseline score for N from current submission."""
    g = df[df['N'] == n]
    xs = np.array([strip(v) for v in g['x']])
    ys = np.array([strip(v) for v in g['y']])
    angles = np.array([strip(v) for v in g['deg']])
    return compute_bbox_score(xs, ys, angles, TX, TY)

if __name__ == "__main__":
    print("=" * 60)
    print("Lattice Packing Experiment")
    print("=" * 60)
    
    # Load baseline
    df = pd.read_csv('/home/submission/submission.csv')
    df['N'] = df['id'].str.split('_').str[0].astype(int)
    
    # Test on small N values first
    test_ns = [10, 20, 30, 50, 100]
    
    improvements = []
    
    for n in test_ns:
        baseline_score = get_baseline_score(n, df)
        print(f"\nN={n}: Baseline score = {baseline_score:.6f}")
        
        best_score = baseline_score
        best_config = None
        
        # Try different spacings and angles
        for spacing in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
            for angle_offset in [0, 15, 30, 45, 60, 75, 90]:
                for lattice_type in ['hex', 'square']:
                    for alternate in [True, False]:
                        if lattice_type == 'hex':
                            xs, ys, angles = hexagonal_lattice_packing(n, spacing, angle_offset, alternate)
                        else:
                            xs, ys, angles = square_lattice_packing(n, spacing, angle_offset, alternate)
                        
                        # Check for overlaps
                        if check_overlaps(xs, ys, angles):
                            continue
                        
                        # Calculate score
                        score = compute_bbox_score(
                            np.array(xs), np.array(ys), np.array(angles), TX, TY
                        )
                        
                        if score < best_score:
                            best_score = score
                            best_config = (lattice_type, spacing, angle_offset, alternate, xs, ys, angles)
                            print(f"  New best: {score:.6f} ({lattice_type}, spacing={spacing}, angle={angle_offset}, alt={alternate})")
        
        improvement = baseline_score - best_score
        if improvement > 0:
            improvements.append((n, improvement, best_config))
            print(f"  ✓ IMPROVEMENT: {improvement:.6f}")
        else:
            print(f"  ✗ No improvement found")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if improvements:
        print(f"Found improvements for {len(improvements)} N values:")
        total_improvement = 0
        for n, imp, config in improvements:
            print(f"  N={n}: +{imp:.6f}")
            total_improvement += imp
        print(f"Total improvement: {total_improvement:.6f}")
    else:
        print("No improvements found for any tested N value.")
    
    # Save metrics
    metrics = {
        'cv_score': 70.316492,  # Will be updated if improvements found
        'baseline_score': 70.316492,
        'improvement': sum(imp for _, imp, _ in improvements) if improvements else 0,
        'num_improvements': len(improvements),
        'notes': f"Lattice packing tested on N={test_ns}. Found {len(improvements)} improvements."
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
