import pytest

dask = pytest.importorskip("dask")

import dask.array as da
import numpy as np
import shapely
import shapely.testing

from grid_indexing.distributed import ChunkGrid, extract_chunk_boundaries


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
