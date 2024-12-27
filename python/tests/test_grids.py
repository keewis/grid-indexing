import numpy as np
import pytest
import xarray as xr

from grid_indexing import grids


class TestInferGridType:
    def test_rectilinear_1d(self):
        lat = xr.Variable("lat", np.linspace(-10, 10, 3), {"standard_name": "latitude"})
        lon = xr.Variable("lon", np.linspace(-5, 5, 4), {"standard_name": "longitude"})
        ds = xr.Dataset(coords={"lat": lat, "lon": lon})

        actual = grids.infer_grid_type(ds)
        assert actual == "1d-rectilinear"

    def test_rectilinear_2d(self):
        lat_, lon_ = np.meshgrid(np.linspace(-10, 10, 3), np.linspace(-5, 5, 4))
        lat = xr.Variable(["y", "x"], lat_, {"standard_name": "latitude"})
        lon = xr.Variable(["y", "x"], lon_, {"standard_name": "longitude"})
        ds = xr.Dataset(coords={"lat": lat, "lon": lon})

        actual = grids.infer_grid_type(ds)
        assert actual == "2d-rectilinear"

    def test_curvilinear_2d(self):
        lat_ = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])
        lon_ = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]])

        lat = xr.Variable(["y", "x"], lat_, {"standard_name": "latitude"})
        lon = xr.Variable(["y", "x"], lon_, {"standard_name": "longitude"})
        ds = xr.Dataset(coords={"lat": lat, "lon": lon})

        actual = grids.infer_grid_type(ds)
        assert actual == "2d-curvilinear"

    def test_unstructured_1d(self):
        lat = xr.Variable(
            "cells", np.linspace(-10, 10, 12), {"standard_name": "latitude"}
        )
        lon = xr.Variable(
            "cells", np.linspace(-5, 5, 12), {"standard_name": "longitude"}
        )
        ds = xr.Dataset(coords={"lat": lat, "lon": lon})

        actual = grids.infer_grid_type(ds)

        assert actual == "1d-unstructured"

    def test_crs_2d(self):
        data = np.linspace(-10, 10, 12).reshape(3, 4)
        geo_transform = "101985.0 300.0379266750948 0.0 2826915.0 0.0 -300.041782729805"

        ds = xr.Dataset(
            {"band_data": (["y", "x"], data)},
            coords={
                "spatial_ref": (
                    (),
                    np.array(0),
                    {
                        "grid_mapping_name": "transverse_mercator",
                        "GeoTransform": geo_transform,
                    },
                )
            },
        )

        actual = grids.infer_grid_type(ds)
        assert actual == "2d-crs"

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
