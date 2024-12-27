import numpy as np
import pytest
import shapely
import shapely.testing
import xarray as xr

from grid_indexing import grids


def example_dataset(grid_type):
    match grid_type:
        case "1d-rectilinear":
            lat_ = np.array([0, 2])
            lon_ = np.array([0, 2, 4])
            lat = xr.Variable("lat", lat_, {"standard_name": "latitude"})
            lon = xr.Variable("lon", lon_, {"standard_name": "longitude"})
            ds = xr.Dataset(coords={"lat": lat, "lon": lon})
        case "2d-rectilinear":
            lat_, lon_ = np.meshgrid(np.array([0, 2]), np.array([0, 2, 4]))
            lat = xr.Variable(["y", "x"], lat_, {"standard_name": "latitude"})
            lon = xr.Variable(["y", "x"], lon_, {"standard_name": "longitude"})
            ds = xr.Dataset(coords={"lat": lat, "lon": lon})
        case "2d-curvilinear":
            lat_ = np.array([[0, 0, 0], [2, 2, 2]])
            lon_ = np.array([[0, 2, 4], [2, 4, 6]])

            lat = xr.Variable(["y", "x"], lat_, {"standard_name": "latitude"})
            lon = xr.Variable(["y", "x"], lon_, {"standard_name": "longitude"})
            ds = xr.Dataset(coords={"lat": lat, "lon": lon})
        case "1d-unstructured":
            lat_ = np.arange(12)
            lon_ = np.arange(-5, 7)
            lat = xr.Variable("cells", lat_, {"standard_name": "latitude"})
            lon = xr.Variable("cells", lon_, {"standard_name": "longitude"})
            ds = xr.Dataset(coords={"lat": lat, "lon": lon})
        case "2d-crs":
            data = np.linspace(-10, 10, 12).reshape(3, 4)
            geo_transform = (
                "101985.0 300.0379266750948 0.0 2826915.0 0.0 -300.041782729805"
            )

            attrs = {
                "grid_mapping_name": "transverse_mercator",
                "GeoTransform": geo_transform,
            }

            ds = xr.Dataset(
                {"band_data": (["y", "x"], data)},
                coords={"spatial_ref": ((), np.array(0), attrs)},
            )

    return ds


def example_geometries(grid_type):
    if grid_type == "2d-crs":
        raise NotImplementedError

    match grid_type:
        case "1d-rectilinear":
            boundaries = np.array(
                [
                    [
                        [[-1, -1], [-1, 1], [1, 1], [1, -1]],
                        [[-1, 1], [-1, 3], [1, 3], [1, 1]],
                    ],
                    [
                        [[1, -1], [1, 1], [3, 1], [3, -1]],
                        [[1, 1], [1, 3], [3, 3], [3, 1]],
                    ],
                    [
                        [[3, -1], [3, 1], [5, 1], [5, -1]],
                        [[3, 1], [3, 3], [5, 3], [5, 1]],
                    ],
                ]
            )
        case "2d-rectilinear":
            boundaries = np.array(
                [
                    [
                        [[-1, -1], [-1, 1], [1, 1], [1, -1]],
                        [[-1, 1], [-1, 3], [1, 3], [1, 1]],
                    ],
                    [
                        [[1, -1], [1, 1], [3, 1], [3, -1]],
                        [[1, 1], [1, 3], [3, 3], [3, 1]],
                    ],
                    [
                        [[3, -1], [3, 1], [5, 1], [5, -1]],
                        [[3, 1], [3, 3], [5, 3], [5, 1]],
                    ],
                ]
            )
        case "2d-curvilinear":
            boundaries = np.array(
                [
                    [
                        [[-2, -1], [0, -1], [2, 1], [0, 1]],
                        [[0, -1], [2, -1], [4, 1], [2, 1]],
                        [[2, -1], [4, -1], [6, 1], [4, 1]],
                    ],
                    [
                        [[0, 1], [2, 1], [4, 3], [2, 3]],
                        [[2, 1], [4, 1], [6, 3], [4, 3]],
                        [[4, 1], [6, 1], [8, 3], [6, 3]],
                    ],
                ]
            )

    return shapely.polygons(boundaries)


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
        ds = example_dataset(grid_type)
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
        ds = example_dataset(grid_type)
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
        ds = example_dataset(grid_type)
        expected = example_geometries(grid_type)

        actual = grids.infer_cell_geometries(ds, grid_type=grid_type)

        shapely.testing.assert_geometries_equal(actual, expected)
