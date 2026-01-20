"""
Perturbed restart optimization.
Add noise to current solution and re-optimize to escape local optima.
"""

import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from shapely.geometry import Polygon
import subprocess
import os
import shutil

getcontext().prec = 25

TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]

def parse_value(val):
    if isinstance(val, str) and val.startswith('s'):
        return val[1:]
    return str(val)

def build_tree_polygon(x, y, angle):
    angle_rad = float(angle) * np.pi / 180.0
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    vertices = []
    for i in range(15):
        px = TX[i] * cos_a - TY[i] * sin_a + float(x)
        py = TX[i] * sin_a + TY[i] * cos_a + float(y)
        vertices.append((px, py))
    
    return Polygon(vertices)

def has_overlap(polygons):
    """Check if any polygons overlap"""
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            if polygons[i].intersects(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > 1e-12:
                    return True
    return False

def get_bounding_box_side(polygons):
    """Get bounding box side length"""
    if not polygons:
        return float('inf')
    
    all_points = []
    for poly in polygons:
        coords = np.asarray(poly.exterior.xy).T
        all_points.append(coords)
    all_points = np.concatenate(all_points)
    
    min_coords = all_points.min(axis=0)
    max_coords = all_points.max(axis=0)
    return max(max_coords - min_coords)

def perturb_submission(input_path, output_path, noise_level=0.1, angle_noise=5.0, seed=42):
    """
    Add noise to a submission to create a perturbed starting point.
    """
    np.random.seed(seed)
    df = pd.read_csv(input_path)
    
    perturbed_rows = []
    for _, row in df.iterrows():
        x = float(parse_value(row['x']))
        y = float(parse_value(row['y']))
        deg = float(parse_value(row['deg']))
        
        # Add noise
        x += np.random.uniform(-noise_level, noise_level)
        y += np.random.uniform(-noise_level, noise_level)
        deg += np.random.uniform(-angle_noise, angle_noise)
        
        perturbed_rows.append({
            'id': row['id'],
            'x': f's{x}',
            'y': f's{y}',
            'deg': f's{deg}'
        })
    
    perturbed_df = pd.DataFrame(perturbed_rows)
    perturbed_df.to_csv(output_path, index=False)
    return perturbed_df

def calculate_score(csv_path):
    """Calculate total score"""
    df = pd.read_csv(csv_path)
    total = 0
    
    for n in range(1, 201):
        prefix = f"{n:03d}_"
        rows = df[df['id'].str.startswith(prefix)]
        
        polygons = []
        for _, row in rows.iterrows():
            x = parse_value(row['x'])
            y = parse_value(row['y'])
            deg = parse_value(row['deg'])
            polygons.append(build_tree_polygon(x, y, deg))
        
        side = get_bounding_box_side(polygons)
        total += (side ** 2) / n
    
    return total

def run_bbox3_optimization(input_csv, output_csv, n_iter=2000, n_rounds=20):
    """Run bbox3 optimizer"""
    # Copy input to submission.csv (bbox3 expects this)
    shutil.copy(input_csv, '/home/code/submission.csv')
    
    # Run bbox3
    cmd = f'cd /home/code && ./bbox3_compiled -n {n_iter} -r {n_rounds} 2>&1'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    
    # Copy result
    shutil.copy('/home/code/submission.csv', output_csv)
    
    return result.stdout

def main():
    baseline_path = '/home/code/submission_candidates/candidate_000.csv'
    baseline_score = calculate_score(baseline_path)
    print(f"Baseline score: {baseline_score:.6f}")
    print("="*60)
    
    best_score = baseline_score
    best_path = baseline_path
    
    # Try different perturbation levels
    for noise_level in [0.05, 0.1, 0.2]:
        for angle_noise in [2.0, 5.0, 10.0]:
            for seed in [42, 123, 456]:
                print(f"\nTrying noise={noise_level}, angle_noise={angle_noise}, seed={seed}")
                
                # Create perturbed version
                perturbed_path = f'/home/code/experiments/003_strict_validation/perturbed_{noise_level}_{angle_noise}_{seed}.csv'
                perturb_submission(baseline_path, perturbed_path, noise_level, angle_noise, seed)
                
                perturbed_score = calculate_score(perturbed_path)
                print(f"  Perturbed score (before optimization): {perturbed_score:.6f}")
                
                # Run bbox3 optimization
                optimized_path = f'/home/code/experiments/003_strict_validation/optimized_{noise_level}_{angle_noise}_{seed}.csv'
                try:
                    output = run_bbox3_optimization(perturbed_path, optimized_path, n_iter=3000, n_rounds=30)
                    
                    # Check final score
                    final_score = calculate_score(optimized_path)
                    print(f"  Optimized score: {final_score:.6f}")
                    print(f"  Improvement over baseline: {baseline_score - final_score:.6f}")
                    
                    if final_score < best_score:
                        best_score = final_score
                        best_path = optimized_path
                        print(f"  *** NEW BEST! ***")
                except Exception as e:
                    print(f"  Error: {e}")
    
    print("\n" + "="*60)
    print(f"Best score found: {best_score:.6f}")
    print(f"Improvement over baseline: {baseline_score - best_score:.6f}")
    
    if best_score < baseline_score:
        shutil.copy(best_path, '/home/code/experiments/003_strict_validation/best_perturbed.csv')
        print(f"Best result saved to best_perturbed.csv")

if __name__ == "__main__":
    main()
