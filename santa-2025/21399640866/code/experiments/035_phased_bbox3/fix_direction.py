"""
fix_direction() - Optimize rotation of entire configuration to minimize bounding box.
Based on the "Best-Keeping bbox3 Runner" kernel approach.
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull
from numba import njit
import math
import json

# Tree geometry
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def strip(v):
    return float(str(v).replace("s", ""))

@njit
def get_tree_vertices(x, y, angle, tx, ty):
    """Get all vertices of a tree polygon."""
    r = angle * math.pi / 180.0
    c = math.cos(r)
    s = math.sin(r)
    vertices = np.zeros((len(tx), 2))
    for i in range(len(tx)):
        vertices[i, 0] = c * tx[i] - s * ty[i] + x
        vertices[i, 1] = s * tx[i] + c * ty[i] + y
    return vertices

@njit
def compute_bbox_side(xs, ys, angles, tx, ty):
    """Compute bounding box side length."""
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
    
    return max(mxx - mnx, mxy - mny)

def rotate_configuration(xs, ys, angles, rotation_angle):
    """Rotate entire configuration by a given angle."""
    rad = np.radians(rotation_angle)
    c, s = np.cos(rad), np.sin(rad)
    
    # Rotate positions
    new_xs = c * xs - s * ys
    new_ys = s * xs + c * ys
    
    # Adjust tree angles
    new_angles = (angles + rotation_angle) % 360
    
    return new_xs, new_ys, new_angles

def fix_direction_for_n(xs, ys, angles, n):
    """Find optimal rotation angle for a single N configuration."""
    current_side = compute_bbox_side(xs, ys, angles, TX, TY)
    
    def bbox_at_angle(rotation_angle):
        new_xs, new_ys, new_angles = rotate_configuration(xs, ys, angles, rotation_angle)
        return compute_bbox_side(new_xs, new_ys, new_angles, TX, TY)
    
    # Search for optimal rotation angle
    result = minimize_scalar(bbox_at_angle, bounds=(0, 90), method='bounded', options={'xatol': 0.001})
    
    if result.fun < current_side - 1e-7:
        new_xs, new_ys, new_angles = rotate_configuration(xs, ys, angles, result.x)
        improvement = current_side - result.fun
        return new_xs, new_ys, new_angles, improvement, result.x
    
    return xs, ys, angles, 0.0, 0.0

def fix_direction(input_csv, output_csv, passes=1):
    """Apply fix_direction to all N values."""
    print(f"Running fix_direction with {passes} passes...")
    
    df = pd.read_csv(input_csv)
    df['N'] = df['id'].str.split('_').str[0].astype(int)
    
    total_improvement = 0
    improvements_found = 0
    
    for pass_num in range(passes):
        pass_improvement = 0
        
        for n in range(1, 201):
            g = df[df['N'] == n].copy()
            xs = np.array([strip(v) for v in g['x']])
            ys = np.array([strip(v) for v in g['y']])
            angles = np.array([strip(v) for v in g['deg']])
            
            new_xs, new_ys, new_angles, improvement, rotation = fix_direction_for_n(xs, ys, angles, n)
            
            if improvement > 1e-7:
                # Update dataframe
                indices = g.index
                for i, idx in enumerate(indices):
                    df.loc[idx, 'x'] = f's{new_xs[i]:.20f}'
                    df.loc[idx, 'y'] = f's{new_ys[i]:.20f}'
                    df.loc[idx, 'deg'] = f's{new_angles[i]:.20f}'
                
                pass_improvement += improvement
                improvements_found += 1
                if improvement > 0.0001:
                    print(f"  N={n}: rotated by {rotation:.3f}°, improvement={improvement:.6f}")
        
        total_improvement += pass_improvement
        print(f"  Pass {pass_num + 1}: improvement={pass_improvement:.6f}")
    
    # Save result
    df = df.drop(columns=['N'])
    df.to_csv(output_csv, index=False)
    
    print(f"Total improvement: {total_improvement:.6f}")
    print(f"Configurations improved: {improvements_found}")
    
    return total_improvement

if __name__ == "__main__":
    import sys
    
    input_csv = sys.argv[1] if len(sys.argv) > 1 else '/home/submission/submission.csv'
    output_csv = sys.argv[2] if len(sys.argv) > 2 else 'submission_fixed.csv'
    passes = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    improvement = fix_direction(input_csv, output_csv, passes)
    print(f"\nFinal improvement: {improvement:.6f}")
