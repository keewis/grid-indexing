[![rust ci](https://github.com/keewis/grid-indexing/actions/workflows/rust-ci.yml/badge.svg?branch=main&event=push)](https://github.com/keewis/grid-indexing/actions/rust-ci.yml?query=branch%3Amain+event%3Apush)
[![python ci](https://github.com/keewis/grid-indexing/actions/workflows/python-ci.yml/badge.svg?branch=main&event=push)](https://github.com/keewis/grid-indexing/actions/python-ci.yml?query=branch%3Amain+event%3Apush)
[![PyPI version](https://img.shields.io/pypi/v/grid-indexing.svg)](https://pypi.org/project/grid-indexing)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

# `grid-indexing`: Fast and scalable indexing of grids

## inferring grid geometries

For grids that fit into memory, or for datasets that include coordinate bounds, the cell polygons can be inferred:

```python
source_geometries = grid_indexing.infer_cell_geometries(ds)
```

## in-memory index

If the geometries fit comfortably into memory (mostly for small grids), we can use the in-memory implementation:

```python
# geoarrow does not support multiple dimensions, so we need to also pass along the shape
source_shape = ...
index = grid_indexing.RTree(source_geometries, source_shape)

overlapping_cells = index.query_overlap(target_geometries, target_shape)
```

The result is a sparse boolean matrix with the same shape as the source / target polygons combined (dimension order is `(target_dim1, ..., source_dim1, ...)`).

## distributed index

The distributed index allows searching for overlapping cells, even when the grids are larger than memory. It is currently built using `dask`.

The procedure is almost the same:

```python
# dask.array objects containing shapely polygons
chunked_source_geoms = ...
chunked_target_geoms = ...

index = grid_indexing.distributed.DistributedRTree(chunked_source_geoms)
overlapping_cells = index.query_overlap(chunked_target_geoms)
```

Note that this will compute both source and target geometries to determine chunk boundaries. `overlapping_cells`, however, is truly lazy.
