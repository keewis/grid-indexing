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
