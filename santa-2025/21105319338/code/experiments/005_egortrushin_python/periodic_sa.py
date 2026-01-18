#!/usr/bin/env python3
"""
Periodic Structure Simulated Annealing - Based on egortrushin's approach
Uses Decimal precision and Shapely for robust geometry
"""

import datetime
import copy
import random
import math
import time
import numpy as np
import pandas as pd
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

getcontext().prec = 25
scale_factor = Decimal("1e15")


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x="0", center_y="0", angle="0"):
        """Initializes the Christmas tree with a specific position and rotation."""
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        trunk_w = Decimal("0.15")
        trunk_h = Decimal("0.2")
        base_w = Decimal("0.7")
        mid_w = Decimal("0.4")
        top_w = Decimal("0.25")
        tip_y = Decimal("0.8")
        tier_1_y = Decimal("0.5")
        tier_2_y = Decimal("0.25")
        base_y = Decimal("0.0")
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                (Decimal("0.0") * scale_factor, tip_y * scale_factor),
                (top_w / Decimal("2") * scale_factor, tier_1_y * scale_factor),
                (top_w / Decimal("4") * scale_factor, tier_1_y * scale_factor),
                (mid_w / Decimal("2") * scale_factor, tier_2_y * scale_factor),
                (mid_w / Decimal("4") * scale_factor, tier_2_y * scale_factor),
                (base_w / Decimal("2") * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal("2") * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal("2") * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal("2")) * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal("2")) * scale_factor, base_y * scale_factor),
                (-(base_w / Decimal("2")) * scale_factor, base_y * scale_factor),
                (-(mid_w / Decimal("4")) * scale_factor, tier_2_y * scale_factor),
                (-(mid_w / Decimal("2")) * scale_factor, tier_2_y * scale_factor),
                (-(top_w / Decimal("4")) * scale_factor, tier_1_y * scale_factor),
                (-(top_w / Decimal("2")) * scale_factor, tier_1_y * scale_factor),
            ]
        )
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated, xoff=float(self.center_x * scale_factor), yoff=float(self.center_y * scale_factor)
        )

    def get_params(self):
        return self.center_x, self.center_y, self.angle

    def set_params(self, center_x, center_y, angle):
        self.__init__(str(center_x), str(center_y), str(angle))

    def clone(self):
        """Create a deep copy of the tree."""
        return ChristmasTree(str(self.center_x), str(self.center_y), str(self.angle))


def get_tree_list_side_length(tree_list):
    """Get the side length of the bounding square."""
    all_polygons = [t.polygon for t in tree_list]
    bounds = unary_union(all_polygons).bounds
    return Decimal(max(bounds[2] - bounds[0], bounds[3] - bounds[1])) / scale_factor


def has_overlap(trees):
    """Check if any trees overlap."""
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            if trees[i].polygon.intersects(trees[j].polygon) and not trees[i].polygon.touches(trees[j].polygon):
                return True
    return False


class SimulatedAnnealing:
    """Simulated Annealing for periodic tree structures."""
    
    def __init__(self, initial_trees, nt, T0=0.1, T1=0.0001, n_iter=100000,
                 position_delta=0.05, angle_delta=10.0, append_y=False):
        """
        Initialize SA optimizer.
        
        Args:
            initial_trees: List of initial tree configurations
            nt: Grid dimensions [rows, cols]
            T0: Initial temperature
            T1: Final temperature
            n_iter: Number of iterations
            position_delta: Maximum position perturbation
            angle_delta: Maximum angle perturbation
            append_y: Whether to add extra row
        """
        self.initial_trees = initial_trees
        self.nt = nt
        self.T0 = T0
        self.T1 = T1
        self.n_iter = n_iter
        self.position_delta = position_delta
        self.angle_delta = angle_delta
        self.append_y = append_y
        
        # Calculate target N
        self.n_trees = nt[0] * nt[1]
        if append_y:
            self.n_trees += nt[1]
    
    def generate_periodic_config(self, tx, ty, base_angle, offset_x=0, offset_y=0):
        """Generate trees in a periodic grid pattern."""
        trees = []
        rows, cols = self.nt
        
        for r in range(rows):
            for c in range(cols):
                x = offset_x + c * tx
                y = offset_y + r * ty
                trees.append(ChristmasTree(str(x), str(y), str(base_angle)))
        
        # Add extra row if append_y
        if self.append_y:
            for c in range(cols):
                x = offset_x + c * tx
                y = offset_y + rows * ty
                trees.append(ChristmasTree(str(x), str(y), str(base_angle)))
        
        return trees
    
    def solve(self):
        """Run simulated annealing optimization."""
        # Initialize with a grid configuration
        best_score = float('inf')
        best_trees = None
        
        # Try multiple starting configurations
        for restart in range(5):
            # Random initial parameters
            tx = Decimal(str(0.5 + random.random() * 0.5))
            ty = Decimal(str(0.5 + random.random() * 0.5))
            base_angle = Decimal(str(random.random() * 360))
            
            trees = self.generate_periodic_config(float(tx), float(ty), float(base_angle))
            
            if has_overlap(trees):
                continue
            
            current_score = float(get_tree_list_side_length(trees) ** 2 / len(trees))
            
            if current_score < best_score:
                best_score = current_score
                best_trees = [t.clone() for t in trees]
            
            # SA optimization
            T = self.T0
            alpha = (self.T1 / self.T0) ** (1.0 / self.n_iter)
            
            for it in range(self.n_iter):
                # Choose move type
                move_type = random.randint(0, 3)
                
                old_tx, old_ty, old_angle = tx, ty, base_angle
                
                if move_type == 0:
                    # Perturb tx
                    tx = Decimal(str(max(0.3, float(tx) + (random.random() - 0.5) * self.position_delta * T * 10)))
                elif move_type == 1:
                    # Perturb ty
                    ty = Decimal(str(max(0.3, float(ty) + (random.random() - 0.5) * self.position_delta * T * 10)))
                elif move_type == 2:
                    # Perturb angle
                    base_angle = Decimal(str((float(base_angle) + (random.random() - 0.5) * self.angle_delta * T * 10) % 360))
                else:
                    # Combined move
                    tx = Decimal(str(max(0.3, float(tx) + (random.random() - 0.5) * self.position_delta * T * 5)))
                    ty = Decimal(str(max(0.3, float(ty) + (random.random() - 0.5) * self.position_delta * T * 5)))
                    base_angle = Decimal(str((float(base_angle) + (random.random() - 0.5) * self.angle_delta * T * 5) % 360))
                
                new_trees = self.generate_periodic_config(float(tx), float(ty), float(base_angle))
                
                if has_overlap(new_trees):
                    tx, ty, base_angle = old_tx, old_ty, old_angle
                    T *= alpha
                    continue
                
                new_score = float(get_tree_list_side_length(new_trees) ** 2 / len(new_trees))
                
                # Accept or reject
                delta = new_score - current_score
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_score = new_score
                    trees = new_trees
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_trees = [t.clone() for t in new_trees]
                else:
                    tx, ty, base_angle = old_tx, old_ty, old_angle
                
                T *= alpha
                
                if it % 10000 == 0:
                    print(f"  Restart {restart}, Iter {it}/{self.n_iter}, T={T:.6f}, Score={current_score:.8f}, Best={best_score:.8f}")
        
        return best_score, best_trees


def load_csv(csv_path):
    """Load submission CSV."""
    df = pd.read_csv(csv_path)
    df['x'] = df['x'].str.strip('s')
    df['y'] = df['y'].str.strip('s')
    df['deg'] = df['deg'].str.strip('s')
    df[['group_id', 'item_id']] = df['id'].str.split('_', n=2, expand=True)
    
    configs = {}
    for group_id, group_data in df.groupby('group_id'):
        n = int(group_id)
        trees = [ChristmasTree(row['x'], row['y'], row['deg']) for _, row in group_data.iterrows()]
        configs[n] = trees
    
    return configs


def save_csv(csv_path, configs):
    """Save submission CSV."""
    rows = []
    for n in range(1, 201):
        if n in configs:
            for i, tree in enumerate(configs[n]):
                rows.append({
                    'id': f'{n:03d}_{i}',
                    'x': f's{tree.center_x}',
                    'y': f's{tree.center_y}',
                    'deg': f's{tree.angle}'
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)


def calculate_total_score(configs):
    """Calculate total score."""
    total = Decimal('0')
    for n, trees in configs.items():
        side = get_tree_list_side_length(trees)
        total += side ** 2 / Decimal(str(n))
    return float(total)


def main():
    print("Periodic Structure SA Optimizer (Python)")
    print("=" * 50)
    
    # Load starting submission
    input_file = "submission.csv"
    output_file = "submission_periodic.csv"
    
    configs = load_csv(input_file)
    initial_score = calculate_total_score(configs)
    print(f"Initial score: {initial_score:.10f}")
    
    # Target N values with grid configurations
    # Format: (N, [rows, cols], append_y)
    targets = [
        (20, [4, 5], False),   # 4x5 = 20
        (24, [4, 6], False),   # 4x6 = 24
        (28, [4, 7], False),   # 4x7 = 28
        (35, [5, 7], False),   # 5x7 = 35
        (40, [5, 8], False),   # 5x8 = 40
        (42, [6, 7], False),   # 6x7 = 42
        (48, [6, 8], False),   # 6x8 = 48
        (56, [7, 8], False),   # 7x8 = 56
        (63, [7, 9], False),   # 7x9 = 63
        (64, [8, 8], False),   # 8x8 = 64
        (72, [8, 9], False),   # 8x9 = 72
        (80, [8, 10], False),  # 8x10 = 80
        (81, [9, 9], False),   # 9x9 = 81
        (90, [9, 10], False),  # 9x10 = 90
        (100, [10, 10], False), # 10x10 = 100
    ]
    
    improvements = 0
    
    for n, nt, append_y in targets:
        print(f"\n--- Optimizing N={n} with grid {nt[0]}x{nt[1]} (append_y={append_y}) ---")
        
        current_trees = configs[n]
        current_side = get_tree_list_side_length(current_trees)
        current_score = float(current_side ** 2 / Decimal(str(n)))
        print(f"Current score for N={n}: {current_score:.10f}")
        
        # Run SA
        sa = SimulatedAnnealing(
            initial_trees=current_trees,
            nt=nt,
            T0=0.1,
            T1=0.0001,
            n_iter=50000,  # Reduced for faster testing
            position_delta=0.05,
            angle_delta=10.0,
            append_y=append_y
        )
        
        new_score, new_trees = sa.solve()
        
        if new_trees and new_score < current_score:
            print(f"IMPROVED N={n}: {current_score:.10f} -> {new_score:.10f}")
            configs[n] = new_trees
            improvements += 1
        else:
            print(f"No improvement for N={n}")
    
    # Calculate final score
    final_score = calculate_total_score(configs)
    print(f"\n{'=' * 50}")
    print(f"Initial score: {initial_score:.10f}")
    print(f"Final score: {final_score:.10f}")
    print(f"Improvement: {initial_score - final_score:.10f}")
    print(f"Total improvements: {improvements}")
    
    # Save result
    save_csv(output_file, configs)
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
