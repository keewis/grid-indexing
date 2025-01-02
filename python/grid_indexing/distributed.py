from dataclasses import dataclass

import geoarrow.rust.core as ga
import numpy as np
import shapely
import sparse

from grid_indexing import Index


def extract_chunk_boundaries(chunks):
    import dask

    coverage = dask.delayed(shapely.unary_union)

    return list(map(coverage, chunks))


def _index_from_shapely(chunk):
    return Index(ga.from_shapely(chunk.flatten()))


def _empty_chunk(index, chunk, shape):
    arr = sparse.full(shape=shape, fill_value=False, dtype=bool)
    return sparse.GCXS.from_coo(arr)


def _query_overlap(index, chunk, shape):
    result = index.query_overlap(ga.from_shapely(chunk.flatten()))

    if result.nnz == 0:
        return _empty_chunk(index, chunk, shape)

    return result


@dataclass
class ChunkGrid:
    shape: tuple
    chunks: np.ndarray

    @classmethod
    def from_dask(cls, arr):
        shape = arr.shape
        chunksizes = arr.chunks

        grid = np.stack(np.meshgrid(*chunksizes), axis=-1)

        return cls(shape, grid)

    @property
    def grid_shape(self):
        return self.chunks.shape[:-1]

    def __repr__(self):
        name = type(self).__name__

        return f"{name}(shape={self.shape}, chunks={np.prod(self.grid_shape)})"

    def chunk_size(self, flattened_index):
        indices = np.unravel_index(flattened_index, self.chunks.shape[:-1])

        return np.prod(self.chunks[*indices, :])
