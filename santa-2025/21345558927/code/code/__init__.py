"""Santa 2025 optimization code."""

from .tree_geometry import (
    TX, TY,
    get_tree_vertices,
    get_tree_vertices_numba,
    calculate_bbox,
    calculate_bbox_numba,
    calculate_score,
    calculate_score_numba
)

from .overlap_check import (
    has_overlap,
    has_any_overlap_numba,
    validate_no_overlap_shapely,
    polygons_overlap_numba
)
