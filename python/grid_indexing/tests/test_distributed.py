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
    extract_chunk_boundaries,
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


def test_extract_chunk_boundaries():
    vertices = np.array(
        [
            [[0, 0], [0, 1], [1, 1], [1, 0]],
            [[1, 0], [1, 1], [2, 1], [2, 0]],
            [[0, 1], [0, 2], [1, 2], [1, 1]],
            [[1, 1], [1, 2], [2, 2], [2, 1]],
        ]
    )
    geometries = shapely.polygons(vertices)

    arr = da.from_array(geometries, chunks=(2,))
    chunks = arr.to_delayed().flatten()

    [actual] = dask.compute(extract_chunk_boundaries(chunks))
    expected = shapely.polygons(
        np.array([[[0, 0], [0, 1], [2, 1], [2, 0]], [[0, 1], [0, 2], [2, 2], [2, 1]]])
    )

    shapely.testing.assert_geometries_equal(actual, expected)


class TestChunkGrid:
    @pytest.mark.parametrize(
        ["arr", "expected_chunks"],
        (
            (da.zeros((4,), chunks=(2,)), np.array([[2], [2]])),
            (
                da.zeros((4, 2), chunks=(2, 1)),
                np.array([[[2, 1], [2, 1]], [[2, 1], [2, 1]]]),
            ),
        ),
    )
    def test_from_dask(self, arr, expected_chunks):
        chunk_grid = ChunkGrid.from_dask(arr)

        assert chunk_grid.shape == arr.shape
        np.testing.assert_equal(chunk_grid.chunks, expected_chunks)

    def test_grid_shape(self):
        shape = (7, 3)
        chunks = np.array(
            [
                [[2, 1], [2, 1], [2, 1]],
                [[2, 1], [2, 1], [2, 1]],
                [[2, 1], [2, 1], [2, 1]],
                [[1, 1], [1, 1], [1, 1]],
            ]
        )

        grid = ChunkGrid(shape, chunks)
        expected = (4, 3)

        assert grid.grid_shape == expected

    @pytest.mark.parametrize(["flattened_index", "expected"], ((0, 12), (3, 9)))
    def test_chunk_size(self, flattened_index, expected):
        shape = (7, 6)
        chunks = np.array([[[4, 3], [4, 3]], [[3, 3], [3, 3]]])

        grid = ChunkGrid(shape, chunks)

        assert grid.chunk_size(flattened_index) == expected

    def test_repr(self):
        shape = (7, 6)
        chunks = np.array([[[4, 3], [4, 3]], [[3, 3], [3, 3]]])

        grid = ChunkGrid(shape, chunks)

        actual = repr(grid)

        assert f"shape={shape}" in actual
        assert "chunks=4" in actual


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
