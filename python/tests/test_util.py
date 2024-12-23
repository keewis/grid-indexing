import numpy as np
import pytest
import shapely

from grid_indexing import util


@pytest.mark.parametrize(
    ["geom", "expected"],
    (
        pytest.param(
            shapely.points([[0, 1], [2, 3]]), ValueError("got POINT"), id="invalid"
        ),
        pytest.param(
            shapely.polygons(
                np.array([[[0, 1], [0, 2], [1, 0]], [[2, 3], [4, 3], [3, 4]]])
            ),
            (
                np.array(
                    [[0, 1], [0, 2], [1, 0], [0, 1], [2, 3], [4, 3], [3, 4], [2, 3]],
                    dtype="float64",
                ),
                np.array([0, 1, 2]),
                np.array([0, 4, 8]),
            ),
            id="polygons",
        ),
    ),
)
def test_as_parts(geom, expected):
    if isinstance(expected, Exception):
        with pytest.raises(type(expected), match=expected.args[0]):
            util.as_parts(geom)
        return

    actual = util.as_parts(geom)

    np.testing.assert_allclose(actual[0], expected[0], err_msg="coordinates differ")
    np.testing.assert_equal(actual[1], expected[1], err_msg="geometry offsets differ")
    np.testing.assert_equal(actual[2], expected[2], err_msg="ring offsets differ")
