import argparse
import math
import random
import os
from decimal import Decimal, getcontext
from copy import deepcopy
import numpy as np
import pandas as pd
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass


@dataclass
class OptimizationResult:
    config_id: int
    trees: list
    best_score: Decimal
    best_side: Decimal
    best_source: str
    old_side: Decimal
    old_score: Decimal
    elapsed: float
    attempt_id: int

# Set precision for Decimal
getcontext().prec = 25
scale_factor = Decimal('1e15')


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x='0', center_y='0', angle='0'):
        """Initializes the Christmas tree with a specific position and rotation."""
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        trunk_w = Decimal('0.15')
        trunk_h = Decimal('0.2')
        base_w = Decimal('0.7')
        mid_w = Decimal('0.4')
        top_w = Decimal('0.25')
        tip_y = Decimal('0.8')
        tier_1_y = Decimal('0.5')
        tier_2_y = Decimal('0.25')
        base_y = Decimal('0.0')
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon([
            (Decimal('0.0') * scale_factor, tip_y * scale_factor),
            (top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
            (top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
            (mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
            (mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
            (base_w / Decimal('2') * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
            (-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
            (-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
            (-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
            (-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
            (-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
        ])
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated,
                                          xoff=float(self.center_x * scale_factor),
                                          yoff=float(self.center_y * scale_factor))

    def clone(self):
        """Create a deep copy of the tree."""
        return ChristmasTree(str(self.center_x), str(self.center_y), str(self.angle))


def check_collision(tree_polygon, placed_polygons, tree_index):
    """Check if a tree collides with any placed trees."""
    possible_indices = tree_index.query(tree_polygon)
    for i in possible_indices:
        if tree_polygon.intersects(placed_polygons[i]) and not tree_polygon.touches(placed_polygons[i]):
            return True
    return False


def calculate_bounding_square(placed_trees):
    """Calculate the side length of the minimum bounding square."""
    if not placed_trees:
        return Decimal('0')
    
    all_polygons = [t.polygon for t in placed_trees]
    bounds = unary_union(all_polygons).bounds
    
    minx = Decimal(bounds[0]) / scale_factor
    miny = Decimal(bounds[1]) / scale_factor
    maxx = Decimal(bounds[2]) / scale_factor
    maxy = Decimal(bounds[3]) / scale_factor
    
    width = maxx - minx
    height = maxy - miny
    side_length = max(width, height)
    
    return side_length


def rebuild_tree_polygon(tree):
    """Rebuild a tree's polygon after modifying position/angle."""
    trunk_w = Decimal('0.15')
    trunk_h = Decimal('0.2')
    base_w = Decimal('0.7')
    mid_w = Decimal('0.4')
    top_w = Decimal('0.25')
    tip_y = Decimal('0.8')
    tier_1_y = Decimal('0.5')
    tier_2_y = Decimal('0.25')
    base_y = Decimal('0.0')
    trunk_bottom_y = -trunk_h

    initial_polygon = Polygon([
        (Decimal('0.0') * scale_factor, tip_y * scale_factor),
        (top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
        (top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
        (mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
        (mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
        (base_w / Decimal('2') * scale_factor, base_y * scale_factor),
        (trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
        (trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
        (-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
        (-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
        (-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
        (-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
        (-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
        (-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
        (-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
    ])
    rotated = affinity.rotate(initial_polygon, float(tree.angle), origin=(0, 0))
    tree.polygon = affinity.translate(rotated,
                                     xoff=float(tree.center_x * scale_factor),
                                     yoff=float(tree.center_y * scale_factor))


def create_grid_aligned_solution(n, n_even, n_odd):
    """
    Create a grid-aligned solution inspired by the well-aligned approach.
    Trees are placed in alternating rows (upright and inverted) for optimal packing.
    """
    all_trees = []
    rest = n
    r = 0
    
    while rest > 0:
        m = min(rest, n_even if r % 2 == 0 else n_odd)
        rest -= m

        angle = 0 if r % 2 == 0 else 180
        x_offset = 0 if r % 2 == 0 else Decimal("0.7") / 2
        y = r // 2 * Decimal("1.0") if r % 2 == 0 else (Decimal("0.8") + (r - 1) // 2 * Decimal("1.0"))
        
        row_trees = [
            ChristmasTree(
                center_x=str(Decimal("0.7") * i + x_offset),
                center_y=str(y),
                angle=str(angle)
            ) for i in range(m)
        ]
        all_trees.extend(row_trees)
        r += 1
    
    return all_trees


def find_best_grid_configuration(n):
    """
    Find the best grid configuration by trying different row sizes.
    This is inspired by the well-aligned notebook approach.
    """
    best_score = Decimal('Infinity')
    best_trees = None
    
    for n_even in range(1, n + 1):
        for n_odd in [n_even, n_even - 1]:
            if n_odd <= 0:
                continue
            
            all_trees = create_grid_aligned_solution(n, n_even, n_odd)
            
            # Calculate bounding box
            xys = np.concatenate([
                np.asarray(t.polygon.exterior.xy).T / float(scale_factor) 
                for t in all_trees
            ])
            
            min_x, min_y = xys.min(axis=0)
            max_x, max_y = xys.max(axis=0)
            
            side = max(max_x - min_x, max_y - min_y)
            score = Decimal(str(side ** 2))
            
            if score < best_score:
                best_score = score
                best_trees = all_trees
    
    return best_trees


def greedy_placement_optimized(num_trees, seed=None):
    """
    Optimized placement combining grid-based and random strategies.
    Uses grid alignment for better initial solutions.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Start with grid-based solution
    placed_trees = find_best_grid_configuration(num_trees)
    
    return placed_trees


def simulated_annealing(placed_trees, max_iterations=5000, initial_temp=1.0):
    """
    Apply simulated annealing to optimize tree placement.
    Uses multiple mutation strategies to find better local optima.
    """
    current_trees = [tree.clone() for tree in placed_trees]
    current_score = calculate_bounding_square(current_trees)
    best_trees = [tree.clone() for tree in current_trees]
    best_score = current_score
    
    temperature = initial_temp
    cooling_rate = 0.99985
    
    for iteration in range(max_iterations):
        # Make a random modification
        new_trees = [tree.clone() for tree in current_trees]
        
        # Multiple mutation strategies with adaptive step sizes
        modification_type = random.choices(
            ['move', 'rotate', 'swap', 'compact', 'nudge_all', 'optimize_spacing'],
            weights=[30, 25, 10, 15, 10, 10]
        )[0]
        
        if modification_type == 'move' and len(new_trees) > 1:
            # Move a single tree
            tree_idx = random.randint(0, len(new_trees) - 1)
            delta = Decimal(str(random.gauss(0, 0.03 * temperature)))
            angle = random.uniform(0, 2 * math.pi)
            
            new_trees[tree_idx].center_x += delta * Decimal(str(math.cos(angle)))
            new_trees[tree_idx].center_y += delta * Decimal(str(math.sin(angle)))
            rebuild_tree_polygon(new_trees[tree_idx])
            
        elif modification_type == 'rotate':
            # Rotate a single tree
            tree_idx = random.randint(0, len(new_trees) - 1)
            delta_angle = Decimal(str(random.gauss(0, 8 * temperature)))
            new_trees[tree_idx].angle += delta_angle
            rebuild_tree_polygon(new_trees[tree_idx])
            
        elif modification_type == 'swap' and len(new_trees) > 1:
            # Swap two trees
            idx1, idx2 = random.sample(range(len(new_trees)), 2)
            new_trees[idx1].center_x, new_trees[idx2].center_x = new_trees[idx2].center_x, new_trees[idx1].center_x
            new_trees[idx1].center_y, new_trees[idx2].center_y = new_trees[idx2].center_y, new_trees[idx1].center_y
            rebuild_tree_polygon(new_trees[idx1])
            rebuild_tree_polygon(new_trees[idx2])
            
        elif modification_type == 'compact' and len(new_trees) > 1:
            # Compact toward center
            center_x = sum(t.center_x for t in new_trees) / len(new_trees)
            center_y = sum(t.center_y for t in new_trees) / len(new_trees)
            
            for tree in new_trees:
                dx = center_x - tree.center_x
                dy = center_y - tree.center_y
                move_factor = Decimal(str(0.015 * temperature))
                
                tree.center_x += dx * move_factor
                tree.center_y += dy * move_factor
                rebuild_tree_polygon(tree)
                
        elif modification_type == 'nudge_all':
            # Nudge all trees in same direction
            angle = random.uniform(0, 2 * math.pi)
            delta = Decimal(str(random.gauss(0, 0.01 * temperature)))
            dx = delta * Decimal(str(math.cos(angle)))
            dy = delta * Decimal(str(math.sin(angle)))
            
            for tree in new_trees:
                tree.center_x += dx
                tree.center_y += dy
                rebuild_tree_polygon(tree)
                
        elif modification_type == 'optimize_spacing' and len(new_trees) > 3:
            # Try to optimize spacing in a subset
            subset_size = min(5, len(new_trees))
            indices = random.sample(range(len(new_trees)), subset_size)
            
            # Small random adjustments to spacing
            for idx in indices:
                delta_x = Decimal(str(random.gauss(0, 0.01 * temperature)))
                delta_y = Decimal(str(random.gauss(0, 0.01 * temperature)))
                new_trees[idx].center_x += delta_x
                new_trees[idx].center_y += delta_y
                rebuild_tree_polygon(new_trees[idx])
        
        # Check for collisions
        all_polygons = [t.polygon for t in new_trees]
        tree_index = STRtree(all_polygons)
        has_collision = False
        
        for i, poly in enumerate(all_polygons):
            possible_indices = tree_index.query(poly)
            for j in possible_indices:
                if i != j and poly.intersects(all_polygons[j]) and not poly.touches(all_polygons[j]):
                    has_collision = True
                    break
            if has_collision:
                break
        
        if not has_collision:
            new_score = calculate_bounding_square(new_trees)
            
            # Accept or reject the new configuration
            delta = float(new_score - current_score)
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_trees = new_trees
                current_score = new_score
                
                if current_score < best_score:
                    best_trees = [tree.clone() for tree in current_trees]
                    best_score = current_score
        
        temperature *= cooling_rate
    
    return best_trees, best_score


def load_existing_configuration(n, existing_df):
    """
    Load existing configuration from submission CSV.
    """
    group_data = existing_df[existing_df['id'].str.startswith(f'{n:03d}_')]
    trees = []
    for _, row in group_data.iterrows():
        x = row['x'][1:]  # Remove 's' prefix
        y = row['y'][1:]
        deg = row['deg'][1:]
        trees.append(ChristmasTree(x, y, deg))
    return trees


def optimize_configuration_attempt(n, existing_df, attempt_seed=None, attempt_id=0):
    """Run a single randomized optimization attempt for configuration n."""
    if attempt_seed is not None:
        random.seed(attempt_seed)
        np.random.seed(attempt_seed % (2 ** 32 - 1))

    start_time = time.time()
    
    best_trees = None
    best_score = Decimal('Infinity')
    best_side = Decimal('Infinity')
    best_source = "new"
    
    # Load and evaluate existing solution
    old_trees = load_existing_configuration(n, existing_df)
    old_side = calculate_bounding_square(old_trees)
    old_score = (old_side ** 2) / Decimal(n)
    
    best_trees = old_trees
    best_score = old_score
    best_side = old_side
    best_source = "old"
    
    # Strategy 1: Apply simulated annealing directly to existing solution
    if n <= 10:
        iterations = 20000
    elif n <= 25:
        iterations = 15000
    else:
        iterations = 10000
    
    try:
        optimized_trees, side_length = simulated_annealing(
            old_trees, 
            max_iterations=iterations, 
            initial_temp=1.0
        )
        score = (side_length ** 2) / Decimal(n)
        
        if score < best_score:
            best_score = score
            best_side = side_length
            best_trees = optimized_trees
            best_source = "new"
    except Exception:
        pass
    
    # Strategy 2: Try grid-based approach with different configurations
    grid_attempts = min(30, n * 2)
    for _ in range(grid_attempts):
        n_even = random.randint(max(1, n // 30), min(n + 1, 25))
        n_odd = n_even + random.choice([-1, 0, 1])
        if n_odd <= 0:
            continue
        
        try:
            trees = create_grid_aligned_solution(n, n_even, n_odd)
            
            optimized_trees, side_length = simulated_annealing(
                trees, 
                max_iterations=5000, 
                initial_temp=0.8
            )
            score = (side_length ** 2) / Decimal(n)
            
            if score < best_score:
                best_score = score
                best_side = side_length
                best_trees = optimized_trees
                best_source = "new"
        except Exception:
            continue
    
    elapsed = time.time() - start_time
    
    return OptimizationResult(
        config_id=n,
        trees=best_trees,
        best_score=best_score,
        best_side=best_side,
        best_source=best_source,
        old_side=old_side,
        old_score=old_score,
        elapsed=elapsed,
        attempt_id=attempt_id,
    )


def optimize_configuration(n, existing_df, attempt_workers=1):
    """Optimize a single configuration using multiple randomized attempts in parallel."""
    if attempt_workers <= 1:
        result = optimize_configuration_attempt(n, existing_df, attempt_seed=random.randint(0, int(1e9)))
        print(
            f"{n:3d}: OLD side={float(result.old_side):.6f} score={float(result.old_score):.6f}, "
            f"NEW side={float(result.best_side):.6f} score={float(result.best_score):.6f}, "
            f"delta={float(result.old_score - result.best_score):.6f}, time={result.elapsed:.1f}s "
            f"{'✓' if result.best_source == 'new' else '(kept)'}"
        )
        return result

    seeds = [random.randint(0, int(1e9)) for _ in range(attempt_workers)]
    start_time = time.time()
    best_result = None

    with ProcessPoolExecutor(max_workers=attempt_workers) as executor:
        futures = [
            executor.submit(
                optimize_configuration_attempt,
                n,
                existing_df,
                seed,
                idx,
            )
            for idx, seed in enumerate(seeds)
        ]

        for future in as_completed(futures):
            try:
                result = future.result()
                if best_result is None or result.best_score < best_result.best_score:
                    best_result = result
            except Exception as exc:
                print(f"Attempt failed for configuration {n}: {exc}")

    total_elapsed = time.time() - start_time
    if best_result is None:
        raise RuntimeError(f"All attempts failed for configuration {n}")

    improvement = float(best_result.old_score - best_result.best_score)
    status = "✓" if best_result.best_source == "new" else "(kept)"
    print(
        f"{n:3d}: OLD side={float(best_result.old_side):.6f} score={float(best_result.old_score):.6f}, "
        f"NEW side={float(best_result.best_side):.6f} score={float(best_result.best_score):.6f}, "
        f"delta={improvement:.6f}, time={total_elapsed:.1f}s attempts={attempt_workers} {status}"
    )

    return best_result


def parallel_optimize(config_ids, attempt_workers=1, existing_df=None):
    """Optimize the provided configuration IDs sequentially, with per-example parallel attempts."""
    if not config_ids:
        print("No configurations requested for optimization.")
        return {}, {}

    ordered_ids = sorted(config_ids)
    print(f"Starting per-example parallelish optimization for configurations: {ordered_ids}")
    print("Each example runs sequentially; each example uses all workers for randomized attempts.")
    print("-" * 70)

    results = {}
    sources = {}
    improvements_count = 0

    for n in ordered_ids:
        try:
            result = optimize_configuration(n, existing_df, attempt_workers=attempt_workers)
            results[n] = result.trees
            sources[n] = result.best_source
            if result.best_source == "new":
                improvements_count += 1
        except Exception as exc:
            print(f"Configuration {n} generated an exception: {exc}")

    print("-" * 70)
    print(f"Optimization complete! Improved {improvements_count}/{len(ordered_ids)} configurations")

    return results, sources


def calculate_total_score(results_dict, optimized_ids, existing_df):
    """Calculate total score, mixing optimized and existing configurations."""
    total_score = Decimal('0')
    optimized_set = set(optimized_ids)

    for n in range(1, 201):
        if n in optimized_set and n in results_dict:
            trees = results_dict[n]
        else:
            trees = load_existing_configuration(n, existing_df)

        side_length = calculate_bounding_square(trees)
        group_score = (side_length ** 2) / Decimal(n)
        total_score += group_score

    return total_score


def create_submission(optimized_results, optimized_ids, input_csv='submission.csv', output_csv='submission_optimized.csv'):
    """Create final submission mixing optimized IDs with existing solutions."""
    print(f"\nCreating submission file: {output_csv}")

    existing_df = pd.read_csv(input_csv)
    optimized_set = set(optimized_ids)

    records = []
    for n in range(1, 201):
        if n in optimized_set and n in optimized_results:
            trees = optimized_results[n]
            for idx, tree in enumerate(trees):
                records.append({
                    'id': f'{n:03d}_{idx}',
                    'x': float(tree.center_x),
                    'y': float(tree.center_y),
                    'deg': float(tree.angle)
                })
        else:
            group_data = existing_df[existing_df['id'].str.startswith(f'{n:03d}_')]
            for _, row in group_data.iterrows():
                records.append({
                    'id': row['id'],
                    'x': float(row['x'][1:]),
                    'y': float(row['y'][1:]),
                    'deg': float(row['deg'][1:])
                })

    submission = pd.DataFrame.from_records(records).set_index('id')
    submission = submission.sort_index()

    for col in ['x', 'y', 'deg']:
        submission[col] = submission[col].astype(float).round(6)
        submission[col] = 's' + submission[col].astype('string')

    submission.to_csv(output_csv)
    print(f"Submission saved to: {output_csv}")

    return submission


def parse_example_ids(example_arg, max_config_id=200):
    """Parse a comma/space separated list of ids or ranges into sorted integers."""
    if not example_arg:
        return []

    tokens = example_arg.replace(',', ' ').split()
    selected = set()

    for token in tokens:
        if '-' in token:
            start_str, end_str = token.split('-', 1)
            if not start_str.isdigit() or not end_str.isdigit():
                raise ValueError(f"Invalid range token '{token}' in --examples")
            start = int(start_str)
            end = int(end_str)
            if start > end:
                start, end = end, start
            if start < 1 or end > max_config_id:
                raise ValueError(f"Example ids must be between 1 and {max_config_id}")
            selected.update(range(start, end + 1))
        else:
            if not token.isdigit():
                raise ValueError(f"Invalid token '{token}' in --examples")
            value = int(token)
            if value < 1 or value > max_config_id:
                raise ValueError(f"Example ids must be between 1 and {max_config_id}")
            selected.add(value)

    return sorted(selected)


def main(argv=None):
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Optimize selected Santa 2025 tree packing examples."
    )
    parser.add_argument(
        "--examples",
        help=(
            "Space or comma separated list of example ids or ranges (e.g. '2-10,15,18'). "
            "Defaults to the scripted window if omitted."
        ),
    )
    args = parser.parse_args(argv)

    print("=" * 70)
    print("Santa 2025 - Parallel Tree Packing Optimizer")
    print("=" * 70)

    # Determine number of workers (per-example parallelism)
    num_cores = mp.cpu_count()
    attempt_workers = max(1, num_cores)

    print(f"Available CPU cores: {num_cores}")
    print(f"Using per-example workers: {attempt_workers}")

    # Load existing submission
    input_csv = 'submission_optimized.csv'
    existing_df = pd.read_csv(input_csv)

    # Calculate old score for configurations 1-200
    print(f"\nCalculating score for existing submission...")
    old_results = {}
    for n in range(1, 201):
        old_results[n] = load_existing_configuration(n, existing_df)
    old_total_score = calculate_total_score(old_results, range(1, 201), existing_df)
    print(f"OLD submission total score: {float(old_total_score):.6f}")

    if args.examples:
        try:
            optimized_ids = parse_example_ids(args.examples)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        skip_first_example = True
        examples_to_run = 10
        first_example = 1 + int(skip_first_example)
        optimized_ids = list(range(first_example, min(201, first_example + examples_to_run)))

    if not optimized_ids:
        print("No configuration ids selected. Exiting.")
        return

    print(f"\nOptimizing configurations: {optimized_ids}\n")
    optimized_results, sources = parallel_optimize(
        config_ids=optimized_ids,
        attempt_workers=attempt_workers,
        existing_df=existing_df,
    )

    # Calculate new total score
    print(f"\nCalculating new total score...")
    new_total_score = calculate_total_score(optimized_results, optimized_ids, existing_df)
    print(f"NEW submission total score: {float(new_total_score):.6f}")

    # Calculate improvement
    improvement = float((old_total_score - new_total_score) / old_total_score * 100)
    print(f"Overall improvement: {improvement:.4f}%")

    # Create submission file
    submission = create_submission(
        optimized_results,
        optimized_ids=optimized_ids,
        input_csv=input_csv,
        output_csv='submission_optimized.csv'
    )

    print("\n" + "=" * 70)
    print("Optimization complete!")
    print("Best solution saved to: submission_optimized.csv")
    print("=" * 70)


if __name__ == "__main__":
    while True:
        main()