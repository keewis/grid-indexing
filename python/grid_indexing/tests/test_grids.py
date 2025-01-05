import geoarrow.rust.core as geoarrow
import numpy as np
import pytest
import shapely
import shapely.testing
import xarray as xr

from grid_indexing import grids
from grid_indexing.tests import example_geometries, example_grid


class TestInferGridType:
    @pytest.mark.parametrize(
        "grid_type",
        [
            "1d-rectilinear",
            "2d-rectilinear",
            "2d-curvilinear",
            "1d-unstructured",
            "2d-crs",
        ],
    )
    def test_infer_grid_type(self, grid_type):
        ds = example_grid(grid_type)
        actual = grids.infer_grid_type(ds)
        assert actual == grid_type

    def test_missing_spatial_coordinates(self):
        ds = xr.Dataset()

        with pytest.raises(ValueError, match="without geographic coordinates"):
            grids.infer_grid_type(ds)

    def test_unknown_grid_type(self):
        lat = np.linspace(-10, 10, 24).reshape(2, 3, 4)
        lon = np.linspace(-5, 5, 24).reshape(2, 3, 4)
        ds = xr.Dataset(
            coords={
                "lat": (["y", "x", "z"], lat, {"standard_name": "latitude"}),
                "lon": (["y", "x", "z"], lon, {"standard_name": "longitude"}),
            }
        )

        with pytest.raises(ValueError, match="unable to infer the grid type"):
            grids.infer_grid_type(ds)


class TestInferCellGeometries:
    @pytest.mark.parametrize(
        ["grid_type", "error", "pattern"],
        (
            pytest.param("2d-crs", NotImplementedError, "geotransform", id="2d-crs"),
            pytest.param(
                "1d-unstructured",
                ValueError,
                "unstructured grids",
                id="1d-unstructured",
            ),
        ),
    )
    def test_not_supported(self, grid_type, error, pattern):
        ds = example_grid(grid_type)
        with pytest.raises(error, match=pattern):
            grids.infer_cell_geometries(ds)

    def test_infer_coords(self):
        ds = xr.Dataset()
        with pytest.raises(ValueError, match="cannot infer geographic coordinates"):
            grids.infer_cell_geometries(ds, grid_type="2d-rectilinear")

    @pytest.mark.parametrize(
        "grid_type",
        [
            "1d-rectilinear",
            "2d-rectilinear",
            "2d-curvilinear",
            pytest.param(
                "2d-crs", marks=pytest.mark.xfail(reason="not yet implemented")
            ),
        ],
    )
    def test_infer_geoms(self, grid_type):
        ds = example_grid(grid_type)
        expected = example_geometries(grid_type)

        actual = grids.infer_cell_geometries(ds, grid_type=grid_type)

        actual_geoms = np.reshape(geoarrow.to_shapely(actual), expected.shape)
        shapely.testing.assert_geometries_equal(actual_geoms, expected)
