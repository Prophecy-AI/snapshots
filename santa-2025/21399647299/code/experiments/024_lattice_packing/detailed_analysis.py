"""
Detailed analysis of lattice packing vs baseline.
"""
import pandas as pd
import numpy as np
from shapely import Polygon
from shapely.affinity import rotate, translate
from numba import njit
import math

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
            x = col * spacing + (row % 2) * spacing / 2
            y = row * spacing * np.sqrt(3) / 2
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
    
    xs = np.array(xs)
    ys = np.array(ys)
    xs = xs - np.mean(xs)
    ys = ys - np.mean(ys)
    
    return list(xs), list(ys), angles

def strip(v):
    return float(str(v).replace("s", ""))

# Load baseline
df = pd.read_csv('/home/submission/submission.csv')
df['N'] = df['id'].str.split('_').str[0].astype(int)

print("Detailed comparison: Lattice vs Baseline")
print("=" * 70)

for n in [10, 20, 50, 100, 150, 200]:
    # Baseline
    g = df[df['N'] == n]
    xs_base = np.array([strip(v) for v in g['x']])
    ys_base = np.array([strip(v) for v in g['y']])
    angles_base = np.array([strip(v) for v in g['deg']])
    baseline_score = compute_bbox_score(xs_base, ys_base, angles_base, TX, TY)
    
    # Best lattice (search more spacings)
    best_lattice_score = float('inf')
    best_lattice_config = None
    
    for spacing in np.arange(0.25, 0.8, 0.02):
        for angle_offset in range(0, 180, 5):
            for alternate in [True, False]:
                xs, ys, angles = hexagonal_lattice_packing(n, spacing, angle_offset, alternate)
                
                if not check_overlaps(xs, ys, angles):
                    score = compute_bbox_score(
                        np.array(xs), np.array(ys), np.array(angles), TX, TY
                    )
                    if score < best_lattice_score:
                        best_lattice_score = score
                        best_lattice_config = (spacing, angle_offset, alternate)
    
    if best_lattice_config:
        diff = baseline_score - best_lattice_score
        status = "✓ BETTER" if diff > 0 else "✗ WORSE"
        print(f"N={n:3d}: Baseline={baseline_score:.6f}, Best Lattice={best_lattice_score:.6f}, Diff={diff:+.6f} {status}")
        print(f"        Config: spacing={best_lattice_config[0]:.2f}, angle={best_lattice_config[1]}, alt={best_lattice_config[2]}")
    else:
        print(f"N={n:3d}: Baseline={baseline_score:.6f}, No valid lattice found")
