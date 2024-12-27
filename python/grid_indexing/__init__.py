from grid_indexing import grid_indexing
from grid_indexing.grid_indexing import Index  # noqa: F401
from grid_indexing.grids import infer_grid_type

__doc__ = grid_indexing.__doc__
if hasattr(grid_indexing, "__all__"):
    __all__ = grid_indexing.__all__ + ["infer_grid_type"]
