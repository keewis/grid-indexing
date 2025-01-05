from dataclasses import dataclass

import geoarrow.rust.core as ga
import numpy as np
import shapely
import sparse

from grid_indexing import Index


def extract_chunk_boundaries(chunks):
    def _chunk_boundaries(chunk):
        union = shapely.unary_union(chunk)

        # TODO: does the minimum rotated rectangle make sense?
        return shapely.minimum_rotated_rectangle(union)

    import dask

    coverage = dask.delayed(_chunk_boundaries)

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
        grid = self.chunks[..., 0]

        return f"{name}(shape={self.shape}, chunks={grid.size})"

    def chunk_size(self, flattened_index):
        indices = np.unravel_index(flattened_index, self.chunks.shape[:-1])

        return np.prod(self.chunks[*indices, :])


class DistributedRTree:
    def __init__(self, grid):
        import dask

        geoms = grid["geometry"].data
        self.source_grid = ChunkGrid.from_dask(geoms)

        self.chunks = geoms.to_delayed().flatten()
        [boundaries] = dask.compute(extract_chunk_boundaries(self.chunks))

        self.chunk_indexes = list(map(dask.delayed(_index_from_shapely), self.chunks))
        self.index = Index.from_shapely(np.array(boundaries))

    def query_overlap(self, grid):
        import dask
        import dask.array as da

        # prepare
        geoms = grid["geometry"].data
        target_grid = ChunkGrid.from_dask(geoms)
        input_chunks = geoms.to_delayed().flatten()

        # query overlapping indices
        [boundaries] = dask.compute(extract_chunk_boundaries(input_chunks))
        geoms = ga.from_shapely(np.array(boundaries))
        overlapping_chunks = self.index.query_overlap(geoms).todense()

        # actual distributed query
        chunks = np.full_like(overlapping_chunks, dtype=object, fill_value=None)
        meta = sparse.GCXS.from_coo(sparse.empty((), dtype=bool))

        for target_index, input_chunk in enumerate(input_chunks):
            for source_index, mask in enumerate(overlapping_chunks[target_index]):
                func = _query_overlap if mask else _empty_chunk
                shape = (
                    target_grid.chunk_size(target_index),
                    self.source_grid.chunk_size(source_index),
                )

                task = dask.delayed(func)(
                    self.chunk_indexes[source_index],
                    input_chunk,
                    shape=shape,
                )
                chunk = da.from_delayed(task, shape=shape, dtype=bool, meta=meta)

                chunks[target_index, source_index] = chunk

        return da.concatenate(
            [
                da.concatenate(chunks[row, :].tolist(), axis=1)
                for row in range(chunks.shape[0])
            ],
            axis=0,
        )
