"""
Find "slack" in the baseline solution.

Slack = trees that could be moved without affecting the bounding box.
If we can find slack, we might be able to rearrange trees to create
a tighter packing.
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
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    return max(max_x - min_x, max_y - min_y), min_x, max_x, min_y, max_y

def compute_score(xs, ys, angles, n):
    bbox, _, _, _, _ = compute_bbox(xs, ys, angles)
    return bbox ** 2 / n

def analyze_bbox_contributors(xs, ys, angles):
    """
    Find which trees contribute to each boundary of the bbox.
    """
    bbox, min_x, max_x, min_y, max_y = compute_bbox(xs, ys, angles)
    
    eps = 1e-6
    
    contributors = {
        'min_x': [],
        'max_x': [],
        'min_y': [],
        'max_y': []
    }
    
    for i in range(len(xs)):
        vertices = get_tree_vertices(xs[i], ys[i], angles[i])
        
        for vx, vy in vertices:
            if abs(vx - min_x) < eps:
                contributors['min_x'].append(i)
                break
        
        for vx, vy in vertices:
            if abs(vx - max_x) < eps:
                contributors['max_x'].append(i)
                break
        
        for vx, vy in vertices:
            if abs(vy - min_y) < eps:
                contributors['min_y'].append(i)
                break
        
        for vx, vy in vertices:
            if abs(vy - max_y) < eps:
                contributors['max_y'].append(i)
                break
    
    # Remove duplicates
    for key in contributors:
        contributors[key] = list(set(contributors[key]))
    
    return contributors, bbox, min_x, max_x, min_y, max_y

def find_slack_trees(xs, ys, angles):
    """
    Find trees that are NOT on any boundary - these have "slack".
    """
    contributors, bbox, min_x, max_x, min_y, max_y = analyze_bbox_contributors(xs, ys, angles)
    
    all_boundary_trees = set()
    for key in contributors:
        all_boundary_trees.update(contributors[key])
    
    slack_trees = [i for i in range(len(xs)) if i not in all_boundary_trees]
    
    return slack_trees, contributors

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
    print("FINDING SLACK IN BASELINE SOLUTION")
    print("=" * 60)
    
    baseline_path = "/home/submission/submission.csv"
    solutions = load_baseline_solution(baseline_path)
    
    # Analyze all N values
    test_ns = list(range(1, 201))
    
    slack_analysis = []
    
    for n in test_ns:
        if n not in solutions:
            continue
            
        trees = solutions[n]
        xs = [t[0] for t in trees]
        ys = [t[1] for t in trees]
        angles = [t[2] for t in trees]
        
        slack_trees, contributors = find_slack_trees(xs, ys, angles)
        
        slack_pct = len(slack_trees) / n * 100
        
        slack_analysis.append({
            'n': n,
            'total_trees': n,
            'slack_trees': len(slack_trees),
            'slack_pct': slack_pct,
            'boundary_trees': n - len(slack_trees),
            'min_x_contributors': len(contributors['min_x']),
            'max_x_contributors': len(contributors['max_x']),
            'min_y_contributors': len(contributors['min_y']),
            'max_y_contributors': len(contributors['max_y'])
        })
    
    # Sort by slack percentage
    slack_analysis.sort(key=lambda x: -x['slack_pct'])
    
    print("\n" + "=" * 60)
    print("TOP 20 N VALUES WITH MOST SLACK")
    print("=" * 60)
    
    for item in slack_analysis[:20]:
        print(f"N={item['n']:3d}: {item['slack_trees']:3d}/{item['total_trees']:3d} slack trees ({item['slack_pct']:.1f}%)")
    
    print("\n" + "=" * 60)
    print("N VALUES WITH ZERO SLACK (FULLY CONSTRAINED)")
    print("=" * 60)
    
    zero_slack = [item for item in slack_analysis if item['slack_trees'] == 0]
    print(f"N values with zero slack: {len(zero_slack)}")
    for item in zero_slack[:10]:
        print(f"  N={item['n']}")
    
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    avg_slack_pct = sum(item['slack_pct'] for item in slack_analysis) / len(slack_analysis)
    max_slack_pct = max(item['slack_pct'] for item in slack_analysis)
    min_slack_pct = min(item['slack_pct'] for item in slack_analysis)
    
    print(f"Average slack: {avg_slack_pct:.1f}%")
    print(f"Max slack: {max_slack_pct:.1f}%")
    print(f"Min slack: {min_slack_pct:.1f}%")
    
    # Save results
    with open('/home/code/experiments/031_asymmetric_tessellation/slack_analysis.json', 'w') as f:
        json.dump(slack_analysis, f, indent=2)

if __name__ == "__main__":
    main()
