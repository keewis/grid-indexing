from importlib.metadata import PackageNotFoundError, version

from grid_indexing import tutorial
from grid_indexing.grid_indexing import RTree, create_empty
from grid_indexing.grids import infer_cell_geometries, infer_grid_type

__all__ = [
    "infer_grid_type",
    "infer_cell_geometries",
    "tutorial",
    "RTree",
    "create_empty",
]

try:
    __version__ = version("grid-indexing")
except PackageNotFoundError:  # noqa # pragma: no cover
    # package is not installed
    __version__ = "9999"
