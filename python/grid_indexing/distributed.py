from dataclasses import dataclass

import numpy as np


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
