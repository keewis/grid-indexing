import cf_xarray  # noqa: F401
import numpy as np


def is_meshgrid(coord):
    return np.all(coord[0, :] == coord[1, :]) or np.all(coord[:, 0] == coord[:, 1])


def infer_grid_type(ds, coords=None):
    # grid types:
    # - 2d crs (affine transform)
    # - 1d orthogonal (rectilinear)
    # - 2d orthogonal (rectilinear)
    # - 2d curvilinear
    # - "unstructured" 1d
    # - "unstructured" n-d
    #
    # Needs to inspect values (except for 2d crs), so must allow computing (so
    # calling `infer_grid_type` should be avoided if possible)
    if "crs" in ds.coords and "affine_transform" in ds["crs"].attrs:
        return "2d-crs"

    if "longitude" not in ds.cf or "latitude" not in ds.cf:
        # projected coords or no spatial coords. Raise for now
        raise ValueError("cannot infer the grid type without geographical coordinates")

    longitude = ds.cf["longitude"]
    latitude = ds.cf["latitude"]

    if longitude.ndim == 1 and latitude.ndim == 1:
        if longitude.dims != latitude.dims:
            return "1d-rectilinear"
        else:
            return "1d-unstructured"
    elif (longitude.ndim == 2 and latitude.ndim == 2) and (
        longitude.dims == latitude.dims
    ):
        # can be unstructured, rectilinear or curvilinear
        if is_meshgrid(longitude.data) and is_meshgrid(latitude.data):
            return "2d-rectilinear"
        else:
            # must be curvilinear (this is not entirely accurate, but
            # "n-d-unstructured" is really hard to check)
            return "2d-curvilinear"
    else:
        raise ValueError("unable to infer the grid type")
