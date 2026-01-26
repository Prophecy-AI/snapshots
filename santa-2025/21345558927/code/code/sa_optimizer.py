"""Simulated Annealing optimizer for Santa 2025 tree packing."""

import numpy as np
from numba import njit
import time

from .tree_geometry import calculate_bbox_numba, calculate_score_numba
from .overlap_check import has_any_overlap_numba

@njit
def sa_optimize_single_n(trees, n_iterations=10000, T_start=1.0, T_end=0.001, seed=42):
    """
    Simulated Annealing optimization for a single N configuration.
    
    Args:
        trees: numpy array of shape (n, 3) with columns [x, y, angle_deg]
        n_iterations: number of SA iterations
        T_start: starting temperature
        T_end: ending temperature
        seed: random seed
    
    Returns:
        best_trees: best configuration found
        best_score: best score achieved
    """
    np.random.seed(seed)
    n = len(trees)
    
    # Initialize
    current = trees.copy()
    current_score = calculate_score_numba(current)
    best = current.copy()
    best_score = current_score
    
    # Cooling schedule
    cooling_rate = (T_end / T_start) ** (1.0 / n_iterations)
    T = T_start
    
    # Move parameters
    translate_std = 0.05  # Standard deviation for translation
    rotate_std = 5.0  # Standard deviation for rotation (degrees)
    
    accepted = 0
    improved = 0
    
    for iteration in range(n_iterations):
        # Generate neighbor
        neighbor = current.copy()
        
        # Choose move type
        move_type = np.random.randint(4)
        
        if move_type == 0:  # Translate single tree
            idx = np.random.randint(n)
            neighbor[idx, 0] += np.random.normal(0, translate_std)
            neighbor[idx, 1] += np.random.normal(0, translate_std)
            
        elif move_type == 1:  # Rotate single tree
            idx = np.random.randint(n)
            neighbor[idx, 2] += np.random.normal(0, rotate_std)
            # Normalize angle to [0, 360)
            neighbor[idx, 2] = neighbor[idx, 2] % 360.0
            
        elif move_type == 2 and n >= 2:  # Swap two trees
            i1 = np.random.randint(n)
            i2 = np.random.randint(n)
            while i2 == i1:
                i2 = np.random.randint(n)
            # Swap positions and angles
            temp = neighbor[i1].copy()
            neighbor[i1] = neighbor[i2].copy()
            neighbor[i2] = temp
            
        else:  # Translate all trees (shift entire configuration)
            dx = np.random.normal(0, translate_std * 0.5)
            dy = np.random.normal(0, translate_std * 0.5)
            for i in range(n):
                neighbor[i, 0] += dx
                neighbor[i, 1] += dy
        
        # Check for overlaps
        if has_any_overlap_numba(neighbor):
            continue
        
        # Calculate new score
        neighbor_score = calculate_score_numba(neighbor)
        delta = neighbor_score - current_score
        
        # Accept or reject
        if delta < 0:
            # Always accept improvements
            current = neighbor.copy()
            current_score = neighbor_score
            accepted += 1
            
            if current_score < best_score:
                best = current.copy()
                best_score = current_score
                improved += 1
        else:
            # Accept worse solutions with probability exp(-delta/T)
            if np.random.random() < np.exp(-delta / T):
                current = neighbor.copy()
                current_score = neighbor_score
                accepted += 1
        
        # Cool down
        T *= cooling_rate
    
    return best, best_score

def sa_optimize_all_n(configs, n_iterations=5000, T_start=0.5, T_end=0.001, verbose=True):
    """
    Run SA optimization on all N configurations.
    
    Args:
        configs: dict of n -> list of (x, y, angle) tuples
        n_iterations: iterations per N
        T_start: starting temperature
        T_end: ending temperature
        verbose: print progress
    
    Returns:
        improved_configs: dict of n -> list of (x, y, angle) tuples
        improvements: list of (n, improvement) tuples
    """
    improved_configs = {}
    improvements = []
    
    start_time = time.time()
    
    for n in range(1, 201):
        trees = configs[n]
        trees_arr = np.array(trees, dtype=np.float64)
        
        original_score = calculate_score_numba(trees_arr)
        
        # Run SA
        best_trees, best_score = sa_optimize_single_n(
            trees_arr, 
            n_iterations=n_iterations,
            T_start=T_start,
            T_end=T_end,
            seed=n  # Different seed for each N
        )
        
        improvement = original_score - best_score
        
        if improvement > 1e-9:
            improvements.append((n, improvement))
            improved_configs[n] = [(best_trees[i, 0], best_trees[i, 1], best_trees[i, 2]) 
                                   for i in range(n)]
            if verbose:
                print(f"N={n}: {original_score:.6f} -> {best_score:.6f} (improved by {improvement:.6f})")
        else:
            improved_configs[n] = trees
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"\nCompleted in {elapsed:.1f} seconds")
        print(f"Total improvements: {len(improvements)}")
    
    return improved_configs, improvements
