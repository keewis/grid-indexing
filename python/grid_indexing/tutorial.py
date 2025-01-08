import numpy as np
import pyproj
import xarray as xr


def _bounds(min_, max_, size):
    offset = abs(max_ - min_) / (2 * size)

    return min_ + offset, max_ - offset


def rectilinear_1d(resolution):
    sizes = {
        "tiny": (18, 9),
        "small": (180, 90),
        "mid": (1800, 900),
        "large": (7200, 3600),
    }
    if resolution not in sizes:
        raise ValueError(f"unknown resolution: {resolution!r}")

    size_lon, size_lat = sizes[resolution]
    min_lat, max_lat = _bounds(-90, 90, size_lat)
    min_lon, max_lon = _bounds(-180, 180, size_lon)

    lat = np.linspace(min_lat, max_lat, size_lat)
    lon = np.linspace(min_lon, max_lon, size_lon)

    return xr.Dataset(
        coords={
            "lat": ("lat", lat, {"standard_name": "latitude"}),
            "lon": ("lon", lon, {"standard_name": "longitude"}),
        }
    )


def rectilinear_2d(resolution):
    sizes = {
        "tiny": (18, 9),
        "small": (180, 90),
        "mid": (1800, 900),
        "large": (7200, 3600),
    }
    if resolution not in sizes:
        raise ValueError(f"unknown resolution: {resolution!r}")

    size_lon, size_lat = sizes[resolution]

    min_lat, max_lat = _bounds(-90, 90, size_lat)
    min_lon, max_lon = _bounds(-180, 180, size_lon)

    lat_ = np.linspace(min_lat, max_lat, size_lat)
    lon_ = np.linspace(min_lon, max_lon, size_lon)
    lon, lat = np.meshgrid(lon_, lat_)

    return xr.Dataset(
        coords={
            "lat": (["y", "x"], lat, {"standard_name": "latitude"}),
            "lon": (["y", "x"], lon, {"standard_name": "longitude"}),
        }
    )


def curvilinear_2d(resolution):
    sizes = {
        "tiny": (18, 9),
        "small": (180, 90),
        "mid": (1800, 900),
        "large": (7200, 3600),
    }
    if resolution not in sizes:
        raise ValueError(f"unknown resolution: {resolution!r}")

    size_x, size_y = sizes[resolution]

    bbox = (-180, -90, 180, 90)
    target_crs = pyproj.CRS.from_epsg(4326)
    source_crs = pyproj.CRS(proj="stere")

    transformer = pyproj.Transformer.from_crs(source_crs, target_crs)

    minx, miny, maxx, maxy = transformer.transform_bounds(*bbox, direction="INVERSE")

    x_ = np.linspace(minx, maxx, size_x)
    y_ = np.linspace(miny, maxy, size_y)

    x, y = np.meshgrid(x_, y_)

    lon, lat = transformer.transform(x, y, direction="FORWARD")

    return xr.Dataset(
        coords={
            "lat": (["y", "x"], lat, {"standard_name": "latitude"}),
            "lon": (["y", "x"], lon, {"standard_name": "longitude"}),
        }
    )


def generate_grid(grid_type, resolution):
    generators = {
        "1d-rectilinear": rectilinear_1d,
        "2d-rectilinear": rectilinear_2d,
        "2d-curvilinear": curvilinear_2d,
        # "1d-unstructured": unstructured_1d,
        # "2d-crs": crs_2d
    }
    generator = generators.get(grid_type)
    if generator is None:
        raise ValueError(f"unknown grid type: {grid_type!r}")

    return generator(resolution)
