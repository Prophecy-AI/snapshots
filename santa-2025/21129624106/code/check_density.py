import numpy as np
from shapely.geometry import Polygon

TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]
TREE_COORDS = list(zip(TX, TY))
poly = Polygon(TREE_COORDS)
area = poly.area
print(f"Tree Area: {area}")

# Current Lattice
dx = 0.41
dy = 0.50
Lx = 0.83 # 2*dx + margin
Ly = 1.02 # 2*dy + margin
# Unit cell (Lx * Ly) contains 2 trees.
unit_cell_area = Lx * Ly
density = (2 * area) / unit_cell_area
print(f"Current Lattice Density: {density}")

# Optimal Density?
# If we remove margins:
Lx_opt = 2 * dx
Ly_opt = 2 * dy
density_opt = (2 * area) / (Lx_opt * Ly_opt)
print(f"Theoretical Density (no margin): {density_opt}")
