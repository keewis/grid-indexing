from dataclasses import dataclass

import geoarrow.rust.core as ga
import numpy as np
import shapely
import sparse

from grid_indexing import Index


def _chunk_boundaries(chunk):
    union = shapely.unary_union(chunk)

    # TODO: does the minimum rotated rectangle make sense?
    return shapely.minimum_rotated_rectangle(union)


def _index_from_shapely(chunk):
    return Index(ga.from_shapely(chunk.flatten()))


def _empty_chunk(index, chunk, shape):
    arr = sparse.full(shape=shape, fill_value=False, dtype=bool)
    return sparse.GCXS.from_coo(arr)


def _query_overlap(index, chunk, shape):
    result = index.query_overlap(ga.from_shapely(chunk.flatten()))

    if result.nnz == 0:
        return _empty_chunk(index, chunk, shape)

    return np.reshape(result, shape)


def _infer_chunksizes(arr):
    return np.stack(np.meshgrid(*arr.chunks, indexing="ij"), axis=-1)


@dataclass
class ChunkGrid:
    shape: tuple[int, ...]
    chunksizes: np.ndarray

    delayed: np.ndarray

    @classmethod
    def from_dask(cls, arr):
        shape = arr.shape
        chunksizes = _infer_chunksizes(arr)

        return cls(shape, chunksizes, arr.to_delayed())

    @property
    def grid_shape(self):
        return self.delayed.shape

    def __getitem__(self, key):
        return self.delayed[key]

    def map(self, func, flatten=False):
        mapped = np.array([func(chunk) for chunk in self.delayed.flatten()])

        result = mapped if flatten else np.reshape(mapped, self.delayed.shape)
        return type(self)(self.shape, self.chunksizes, result)

    def flat_iter_chunks(self):
        shape = self.grid_shape
        size = int(np.prod(shape))

        for flat_index in range(size):
            indices = tuple(map(int, np.unravel_index(flat_index, shape)))
            chunk_shape = tuple(map(int, self.chunksizes[*indices, :]))

            yield indices, chunk_shape, self.delayed[indices]

    def compute(self):
        import dask

        [computed] = dask.compute(self.delayed.flatten().tolist())

        return np.reshape(np.asarray(computed, dtype=object), self.delayed.shape)

    def __repr__(self):
        name = type(self).__name__
        grid = self.chunksizes[..., 0]

        return f"{name}(shape={self.shape}, chunks={grid.size})"


class DistributedRTree:
    def __init__(self, geoms):
        import dask

        self.source_grid = ChunkGrid.from_dask(geoms)

        boundaries = self.source_grid.map(
            dask.delayed(_chunk_boundaries), flatten=True
        ).compute()

        self.chunk_indexes = self.source_grid.map(dask.delayed(_index_from_shapely))
        self.index = Index.from_shapely(np.array(boundaries))

    def query_overlap(self, geoms):
        import dask
        import dask.array as da

        # prepare
        target_grid = ChunkGrid.from_dask(geoms)
        chunk_grid_shape = target_grid.grid_shape + self.source_grid.grid_shape

        # query overlapping indices
        [boundaries] = dask.compute(
            target_grid.map(
                dask.delayed(_chunk_boundaries), flatten=True
            ).delayed.tolist()
        )
        geoms = ga.from_shapely(np.array(boundaries))
        query_result = self.index.query_overlap(geoms).todense()
        overlapping_chunks = np.reshape(query_result, chunk_grid_shape)

        # actual distributed query
        output_chunks = np.full(chunk_grid_shape, dtype=object, fill_value=None)
        meta = sparse.GCXS.from_coo(sparse.empty((), dtype=bool))

        # TODO: maybe use `itertools.product` instead?
        for (
            target_indices,
            target_chunk_shape,
            target_chunk,
        ) in target_grid.flat_iter_chunks():
            for (
                source_indices,
                source_chunk_shape,
                source_chunk,
            ) in self.source_grid.flat_iter_chunks():
                indices = target_indices + source_indices
                mask = overlapping_chunks[indices]

                func = _query_overlap if mask else _empty_chunk
                chunk_shape = target_chunk_shape + source_chunk_shape

                task = dask.delayed(func)(
                    self.chunk_indexes[source_indices],
                    target_chunk,
                    shape=chunk_shape,
                )
                chunk = da.from_delayed(task, shape=chunk_shape, dtype=bool, meta=meta)

                output_chunks[indices] = chunk

        return da.block(output_chunks.tolist())
