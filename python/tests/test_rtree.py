import numpy as np
import shapely
import sparse

from grid_indexing import Index, as_parts


def create_cells(x, y):
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
    return shapely.polygons(vertices)


def test_create_index():
    x = np.linspace(-10, 10, 6)
    y = np.linspace(40, 60, 4)

    cells = create_cells(x, y).flatten()

    type_, coords, (ring_offsets, geom_offsets) = shapely.to_ragged_array(cells)

    index = Index(coords, geom_offsets, ring_offsets)
    assert isinstance(index, Index)


def test_query_overlap():
    source_cells = create_cells(
        np.linspace(-10, 10, 6), np.linspace(40, 60, 4)
    ).flatten()
    target_cells = source_cells

    index = Index(*as_parts(source_cells))

    actual = index.query_overlap(*as_parts(target_cells))

    assert isinstance(actual, sparse.GCXS)
    sum_ = np.sum(actual, axis=1).todense()
    assert np.all(
        np.isin(sum_, np.array([1, 4, 6, 9]))
    ), "unexpected number of cells found"
