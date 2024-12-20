import numpy as np
import shapely

from grid_indexing import Index


def test_create_index():
    x = np.linspace(-10, 10, 6)
    y = np.linspace(40, 60, 4)

    x_step = abs(x[1] - x[0]) / 2
    y_step = abs(y[1] - y[0]) / 2

    X, Y = np.meshgrid(x, y)

    xmin = X - x_step
    xmax = X + x_step
    ymin = Y - y_step
    ymax = Y + y_step

    vertices_ = np.array(
        [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]]
    )
    vertices = np.moveaxis(vertices_, (0, 1), (-2, -1))
    polygons = shapely.polygons(vertices)

    geom_type, coords, (ring_offsets, geom_offsets) = shapely.to_ragged_array(
        polygons.flatten()
    )

    index = Index(coords, geom_offsets, ring_offsets)
    assert isinstance(index, Index)
