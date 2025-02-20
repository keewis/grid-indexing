import pytest

dask = pytest.importorskip("dask")

import dask.array as da
import geoarrow.rust.core as geoarrow
import numpy as np
import shapely
import shapely.testing

from grid_indexing import Index
from grid_indexing.distributed import (
    ChunkGrid,
    DistributedRTree,
    _chunk_boundaries,
    _infer_chunksizes,
)
from grid_indexing.tests import example_geometries


@pytest.fixture
def example_grid():
    vertices = np.array(
        [
            [[0, 0], [0, 1], [1, 1], [1, 0]],
            [[1, 0], [1, 1], [2, 1], [2, 0]],
            [[0, 1], [0, 2], [1, 2], [1, 1]],
            [[1, 1], [1, 2], [2, 2], [2, 1]],
        ]
    )
    return shapely.polygons(vertices)


@pytest.fixture(params=range(2))
def example_query(request):
    queries = [
        shapely.polygons(
            np.array(
                [
                    [[0.1, 0], [0.15, 0.45], [0.35, 0.4], [0.3, 0.05]],
                    [[0.4, 0.2], [0.4, 1.4], [1.7, 1.4], [1.7, 0.2]],
                ]
            )
        ),
        shapely.polygons(
            np.array(
                [
                    [[0.3, 0.1], [0.5, 0.45], [0.9, 0.4], [0.7, 0.05]],
                    [[0.7, 0.2], [1.2, 1.4], [1.7, 1.4], [1.7, 0.2]],
                ]
            )
        ),
    ]

    return queries[request.param]


def test_chunk_boundaries():
    vertices = np.array(
        [
            [[0, 0], [0, 1], [1, 1], [1, 0]],
            [[1, 0], [1, 1], [2, 1], [2, 0]],
            [[0, 1], [0, 2], [1, 2], [1, 1]],
            [[1, 1], [1, 2], [2, 2], [2, 1]],
        ]
    )
    chunk = shapely.polygons(vertices)

    actual = _chunk_boundaries(chunk)
    expected = shapely.polygons(np.array([[[0, 0], [0, 2], [2, 2], [2, 0]]]))

    shapely.testing.assert_geometries_equal(actual, expected)


@pytest.mark.parametrize(
    ["arr", "expected"],
    (
        (
            da.zeros((12, 10), chunks=(6, 5)),
            np.array([[[6, 5], [6, 5]], [[6, 5], [6, 5]]]),
        ),
        (
            da.zeros((5, 3), chunks=(2, 3)),
            np.array([[[2, 3]], [[2, 3]], [[1, 3]]]),
        ),
    ),
)
def test_infer_chunksizes(arr, expected):
    actual = _infer_chunksizes(arr)

    np.testing.assert_equal(actual, expected)


class TestChunkGrid:
    @pytest.mark.parametrize(
        ["arr", "expected_chunksizes"],
        (
            (da.zeros((4,), chunks=(2,)), np.array([[2], [2]])),
            (
                da.zeros((4, 2), chunks=(2, 1)),
                np.array([[[2, 1], [2, 1]], [[2, 1], [2, 1]]]),
            ),
        ),
    )
    def test_from_dask(self, arr, expected_chunksizes):
        chunk_grid = ChunkGrid.from_dask(arr)

        assert chunk_grid.shape == arr.shape
        np.testing.assert_equal(chunk_grid.chunksizes, expected_chunksizes)

    def test_grid_shape(self):
        arr = da.zeros((7, 3), chunks=(2, 1))
        shape = arr.shape
        chunksizes = _infer_chunksizes(arr)

        grid = ChunkGrid(shape, chunksizes, arr.to_delayed())
        expected = (4, 3)

        assert grid.grid_shape == expected

    def test_repr(self):
        arr = da.zeros((7, 6), chunks=(4, 3))
        shape = arr.shape
        chunksizes = _infer_chunksizes(arr)

        grid = ChunkGrid(shape, chunksizes, arr.to_delayed())

        actual = repr(grid)

        assert f"shape={shape}" in actual
        assert "chunks=4" in actual

    def test_getitem(self):
        arr = da.zeros((7, 6), chunks=(5, 4))

        shape = arr.shape
        chunksizes = _infer_chunksizes(arr)
        delayed = arr.to_delayed()

        grid = ChunkGrid(shape, chunksizes, delayed)

        indices = (1, 0)

        actual = grid[indices]
        expected = delayed[indices]

        assert actual is expected

    def test_map(self):
        func = lambda c: c * 2

        arr = da.from_array(np.arange(12).reshape(4, 3), chunks=(3, 2))

        shape = arr.shape
        chunksizes = _infer_chunksizes(arr)
        delayed = arr.to_delayed()

        grid = ChunkGrid(shape, chunksizes, delayed)

        actual = grid.map(func)
        computed_ = np.asarray(
            dask.compute(actual.delayed.flatten().tolist())[0], dtype=object
        )
        actual_ = da.block(np.reshape(computed_, delayed.shape).tolist())
        expected = func(arr).compute()

        np.testing.assert_equal(actual_, expected)


class TestDistributedRTree:
    @pytest.mark.parametrize(
        "grid_type", ["1d-rectilinear", "2d-rectilinear", "2d-curvilinear"]
    )
    def test_init(self, grid_type):
        polygons = example_geometries(grid_type)
        chunked_polygons = da.from_array(polygons, chunks=(1, 1))

        index = DistributedRTree(chunked_polygons)

        assert isinstance(index, DistributedRTree)

    @pytest.mark.parametrize("chunks", (1, 2))
    def test_query_overlap(self, example_grid, example_query, chunks):
        chunked_polygons = da.from_array(example_grid, chunks=(2,))

        index = Index.from_shapely(example_grid.flatten())
        distributed_index = DistributedRTree(chunked_polygons)

        chunked_query = da.from_array(example_query, chunks=chunks)

        expected = index.query_overlap(geoarrow.from_shapely(example_query))
        [actual] = dask.compute(distributed_index.query_overlap(chunked_query))

        np.testing.assert_equal(actual.todense(), expected.todense())
