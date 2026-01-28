"""
Create incremental submission test.

Replace ONLY N=123 from the improved ensemble into exp_022.
This tests if single N value replacement passes Kaggle validation.
"""

import pandas as pd
import numpy as np
from shapely import Polygon
import math
import json

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def get_tree_vertices(x, y, angle_deg):
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    vertices = []
    for tx, ty in zip(TX, TY):
        rx = tx * cos_a - ty * sin_a
        ry = tx * sin_a + ty * cos_a
        vertices.append((rx + x, ry + y))
    
    return vertices

def get_tree_polygon(x, y, angle_deg):
    vertices = get_tree_vertices(x, y, angle_deg)
    return Polygon(vertices)

def compute_bbox(trees):
    all_x = []
    all_y = []
    
    for x, y, angle in trees:
        vertices = get_tree_vertices(x, y, angle)
        for vx, vy in vertices:
            all_x.append(vx)
            all_y.append(vy)
    
    return max(max(all_x) - min(all_x), max(all_y) - min(all_y))

def compute_score(trees, n):
    bbox = compute_bbox(trees)
    return bbox ** 2 / n

def validate_no_overlap(trees, tolerance=1e-15):
    n = len(trees)
    polygons = [get_tree_polygon(x, y, a) for x, y, a in trees]
    
    for i in range(n):
        for j in range(i + 1, n):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > tolerance:
                    return False, f"Trees {i} and {j} overlap (area={intersection.area:.2e})"
    
    return True, "OK"

def load_solution(csv_path):
    df = pd.read_csv(csv_path)
    
    solutions = {}
    for _, row in df.iterrows():
        id_parts = str(row['id']).split('_')
        n = int(id_parts[0])
        i = int(id_parts[1])
        
        x = float(str(row['x']).replace('s', ''))
        y = float(str(row['y']).replace('s', ''))
        deg = float(str(row['deg']).replace('s', ''))
        
        if n not in solutions:
            solutions[n] = []
        solutions[n].append((x, y, deg))
    
    return solutions

def main():
    print("=" * 60)
    print("CREATING INCREMENTAL SUBMISSION TEST")
    print("=" * 60)
    
    # Load exp_022 (base)
    exp022_path = "/home/code/experiments/022_extended_cpp_optimization/optimized.csv"
    print(f"\nLoading exp_022 from {exp022_path}...")
    exp022 = load_solution(exp022_path)
    
    # Load improved ensemble
    improved_path = "/home/code/experiments/038_improved_ensemble/submission.csv"
    print(f"Loading improved ensemble from {improved_path}...")
    improved = load_solution(improved_path)
    
    # Compute baseline scores
    exp022_score_123 = compute_score(exp022[123], 123)
    improved_score_123 = compute_score(improved[123], 123)
    
    print(f"\nN=123 scores:")
    print(f"  exp_022: {exp022_score_123:.6f}")
    print(f"  improved: {improved_score_123:.6f}")
    print(f"  improvement: {exp022_score_123 - improved_score_123:.6f}")
    
    # Validate improved N=123
    ok, msg = validate_no_overlap(improved[123], tolerance=1e-15)
    print(f"\nImproved N=123 validation (tol=1e-15): {ok} - {msg}")
    
    # Create incremental submission: exp_022 with N=123 replaced
    print("\n" + "=" * 60)
    print("CREATING INCREMENTAL SUBMISSION")
    print("=" * 60)
    
    # Start with exp_022
    incremental = {}
    for n in range(1, 201):
        if n in exp022:
            incremental[n] = exp022[n]
    
    # Replace N=123 with improved version
    incremental[123] = improved[123]
    
    # Compute total score
    total_exp022 = sum(compute_score(exp022[n], n) for n in range(1, 201) if n in exp022)
    total_incremental = sum(compute_score(incremental[n], n) for n in range(1, 201) if n in incremental)
    
    print(f"\nTotal scores:")
    print(f"  exp_022: {total_exp022:.6f}")
    print(f"  incremental: {total_incremental:.6f}")
    print(f"  improvement: {total_exp022 - total_incremental:.6f}")
    
    # Validate the replaced N=123
    ok, msg = validate_no_overlap(incremental[123], tolerance=1e-15)
    print(f"\nIncremental N=123 validation: {ok} - {msg}")
    
    # Save incremental submission
    print("\n" + "=" * 60)
    print("SAVING INCREMENTAL SUBMISSION")
    print("=" * 60)
    
    # Load original exp_022 CSV to preserve exact format
    df_exp022 = pd.read_csv(exp022_path)
    
    # Get improved N=123 rows
    df_improved = pd.read_csv(improved_path)
    improved_123 = df_improved[df_improved['id'].str.startswith('123_')]
    
    # Replace N=123 rows in exp_022
    df_incremental = df_exp022[~df_exp022['id'].str.startswith('123_')]
    df_incremental = pd.concat([df_incremental, improved_123], ignore_index=True)
    
    # Sort by id
    df_incremental['sort_key'] = df_incremental['id'].apply(lambda x: (int(x.split('_')[0]), int(x.split('_')[1])))
    df_incremental = df_incremental.sort_values('sort_key').drop('sort_key', axis=1)
    
    # Save
    output_path = "/home/code/experiments/040_incremental_test/submission.csv"
    df_incremental.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    # Also save to submission folder
    submission_path = "/home/submission/submission.csv"
    df_incremental.to_csv(submission_path, index=False)
    print(f"Saved to {submission_path}")
    
    # Verify the submission
    print("\n" + "=" * 60)
    print("VERIFYING SUBMISSION")
    print("=" * 60)
    
    df_verify = pd.read_csv(submission_path)
    print(f"Total rows: {len(df_verify)}")
    
    # Check N=123 is from improved
    verify_123 = df_verify[df_verify['id'].str.startswith('123_')]
    print(f"N=123 rows: {len(verify_123)}")
    
    # Compare first row of N=123
    print(f"\nN=123 first row (should match improved):")
    print(f"  Submission: {verify_123.iloc[0]['x']}")
    print(f"  Improved: {improved_123.iloc[0]['x']}")
    
    # Save metrics
    metrics = {
        'cv_score': total_incremental,
        'baseline_score': total_exp022,
        'improvement': total_exp022 - total_incremental,
        'modified_n_values': [123],
        'notes': 'Incremental test - replaced ONLY N=123 from improved ensemble into exp_022. Testing if single N value replacement passes Kaggle validation.'
    }
    
    with open('/home/code/experiments/040_incremental_test/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nMetrics saved")
    
    return total_incremental

if __name__ == "__main__":
    total = main()
