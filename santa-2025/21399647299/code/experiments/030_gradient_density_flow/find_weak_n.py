"""
Find N values where the baseline might be weak and could be improved.

Look for:
1. N values with unusually low efficiency
2. N values where the score contribution is high
3. N values where small improvements would have big impact
"""

import numpy as np
import pandas as pd
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

def load_baseline_solution(csv_path):
    df = pd.read_csv(csv_path)
    
    solutions = {}
    for _, row in df.iterrows():
        id_parts = row['id'].split('_')
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
    print("FINDING WEAK N VALUES")
    print("=" * 60)
    
    baseline_path = "/home/submission/submission.csv"
    solutions = load_baseline_solution(baseline_path)
    
    # Compute tree area
    tree_poly = Polygon(list(zip(TX, TY)))
    tree_area = tree_poly.area
    
    # Analyze all N values
    results = []
    total_score = 0
    
    for n in range(1, 201):
        if n not in solutions:
            continue
            
        trees = solutions[n]
        xs = [t[0] for t in trees]
        ys = [t[1] for t in trees]
        angles = [t[2] for t in trees]
        
        bbox = compute_bbox(xs, ys, angles)
        score = bbox ** 2 / n
        total_score += score
        
        # Efficiency
        total_tree_area = n * tree_area
        bbox_area = bbox ** 2
        efficiency = total_tree_area / bbox_area
        
        # Theoretical minimum
        theoretical_min = tree_area
        gap = score - theoretical_min
        gap_pct = (score / theoretical_min - 1) * 100
        
        results.append({
            'n': n,
            'score': score,
            'bbox': bbox,
            'efficiency': efficiency,
            'gap': gap,
            'gap_pct': gap_pct,
            'contribution_pct': score / 70.316 * 100  # Approximate
        })
    
    # Sort by different criteria
    print("\n" + "=" * 60)
    print("TOP 20 N VALUES BY SCORE CONTRIBUTION")
    print("=" * 60)
    by_score = sorted(results, key=lambda x: -x['score'])[:20]
    for r in by_score:
        print(f"N={r['n']:3d}: score={r['score']:.4f}, efficiency={r['efficiency']*100:.1f}%, gap={r['gap_pct']:.1f}%")
    
    print("\n" + "=" * 60)
    print("TOP 20 N VALUES BY LOWEST EFFICIENCY (MOST ROOM TO IMPROVE)")
    print("=" * 60)
    by_efficiency = sorted(results, key=lambda x: x['efficiency'])[:20]
    for r in by_efficiency:
        print(f"N={r['n']:3d}: score={r['score']:.4f}, efficiency={r['efficiency']*100:.1f}%, gap={r['gap_pct']:.1f}%")
    
    print("\n" + "=" * 60)
    print("TOP 20 N VALUES BY GAP TO THEORETICAL")
    print("=" * 60)
    by_gap = sorted(results, key=lambda x: -x['gap'])[:20]
    for r in by_gap:
        print(f"N={r['n']:3d}: score={r['score']:.4f}, efficiency={r['efficiency']*100:.1f}%, gap={r['gap']:.4f}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total score: {total_score:.6f}")
    print(f"Target score: 68.873342")
    print(f"Gap to target: {total_score - 68.873342:.6f}")
    print(f"Need to reduce by: {(total_score - 68.873342) / total_score * 100:.2f}%")
    
    # If we improved efficiency by X% across all N, what would the new score be?
    print("\n" + "=" * 60)
    print("IMPACT OF EFFICIENCY IMPROVEMENTS")
    print("=" * 60)
    for improvement in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        new_score = sum(r['score'] * (1 - improvement/100) for r in results)
        print(f"  {improvement}% efficiency improvement -> score {new_score:.4f} (gap: {new_score - 68.873342:.4f})")
    
    # Save results
    with open('/home/code/experiments/030_gradient_density_flow/weak_n_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
