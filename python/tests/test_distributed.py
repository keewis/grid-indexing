import pytest

dask = pytest.importorskip("dask")

import dask.array as da
import numpy as np

from grid_indexing import distributed as gid


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
        chunk_grid = gid.ChunkGrid.from_dask(arr)

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

        grid = gid.ChunkGrid(shape, chunks)
        expected = (4, 3)

        assert grid.grid_shape == expected

    @pytest.mark.parametrize(["flattened_index", "expected"], ((0, 12), (3, 9)))
    def test_chunk_size(self, flattened_index, expected):
        shape = (7, 6)
        chunks = np.array([[[4, 3], [4, 3]], [[3, 3], [3, 3]]])

        grid = gid.ChunkGrid(shape, chunks)

        assert grid.chunk_size(flattened_index) == expected
