"""
Analyze external data to understand why no improvements were found.
"""

import numpy as np
import pandas as pd
import math
import os
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

def compute_bbox(xs, ys, angles):
    all_x = []
    all_y = []
    
    for x, y, angle in zip(xs, ys, angles):
        vertices = get_tree_vertices(x, y, angle)
        for vx, vy in vertices:
            all_x.append(vx)
            all_y.append(vy)
    
    return max(max(all_x) - min(all_x), max(all_y) - min(all_y))

def compute_score(xs, ys, angles, n):
    bbox = compute_bbox(xs, ys, angles)
    return bbox ** 2 / n

def load_solution_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        
        if 'id' not in df.columns:
            return None
        
        solutions = {}
        for _, row in df.iterrows():
            try:
                id_parts = str(row['id']).split('_')
                if len(id_parts) != 2:
                    continue
                n = int(id_parts[0])
                i = int(id_parts[1])
                
                x = float(str(row['x']).replace('s', ''))
                y = float(str(row['y']).replace('s', ''))
                deg = float(str(row['deg']).replace('s', ''))
                
                if n not in solutions:
                    solutions[n] = []
                solutions[n].append((x, y, deg))
            except:
                continue
        
        return solutions
    except:
        return None

def compute_total_score(solutions):
    total = 0
    for n in range(1, 201):
        if n in solutions and len(solutions[n]) == n:
            trees = solutions[n]
            xs = [t[0] for t in trees]
            ys = [t[1] for t in trees]
            angles = [t[2] for t in trees]
            total += compute_score(xs, ys, angles, n)
    return total

def main():
    print("=" * 60)
    print("ANALYZING EXTERNAL DATA SCORES")
    print("=" * 60)
    
    # Load baseline
    baseline_path = "/home/submission/submission.csv"
    baseline = load_solution_from_csv(baseline_path)
    baseline_total = compute_total_score(baseline)
    print(f"\nBaseline total score: {baseline_total:.6f}")
    
    # Check key external files
    external_dir = "/home/code/data/external"
    
    key_files = [
        "70.378875862989_20260126_045659.csv",
        "71.97.csv",
        "72.49.csv",
        "santa2025_ver2_v69.csv",
        "santa2025_ver2_v76.csv",
        "submission_best.csv",
        "submission_JKoT1.csv",
        "submission_JKoT2.csv",
        "smartmanoj_github.csv",
    ]
    
    print("\nExternal file scores:")
    for filename in key_files:
        filepath = os.path.join(external_dir, filename)
        if os.path.exists(filepath):
            solutions = load_solution_from_csv(filepath)
            if solutions:
                total = compute_total_score(solutions)
                diff = total - baseline_total
                print(f"  {filename}: {total:.6f} ({diff:+.6f})")
    
    # Check kernel outputs
    kernel_dir = os.path.join(external_dir, "kernel_outputs")
    if os.path.exists(kernel_dir):
        print("\nKernel output scores:")
        for folder in sorted(os.listdir(kernel_dir))[:10]:
            folder_path = os.path.join(kernel_dir, folder)
            if os.path.isdir(folder_path):
                for f in os.listdir(folder_path):
                    if f.endswith('.csv'):
                        filepath = os.path.join(folder_path, f)
                        solutions = load_solution_from_csv(filepath)
                        if solutions:
                            total = compute_total_score(solutions)
                            diff = total - baseline_total
                            print(f"  {folder}/{f}: {total:.6f} ({diff:+.6f})")
                        break  # Only check first CSV in each folder
    
    # Check snapshots
    snapshots_dir = "/home/nonroot/snapshots"
    if os.path.exists(snapshots_dir):
        print("\nSnapshot scores (first 10):")
        count = 0
        for folder in sorted(os.listdir(snapshots_dir)):
            folder_path = os.path.join(snapshots_dir, folder)
            if os.path.isdir(folder_path):
                for f in os.listdir(folder_path):
                    if f.endswith('.csv') and 'submission' in f.lower():
                        filepath = os.path.join(folder_path, f)
                        solutions = load_solution_from_csv(filepath)
                        if solutions:
                            total = compute_total_score(solutions)
                            diff = total - baseline_total
                            print(f"  {folder}/{f}: {total:.6f} ({diff:+.6f})")
                            count += 1
                            if count >= 10:
                                break
                if count >= 10:
                    break

if __name__ == "__main__":
    main()
