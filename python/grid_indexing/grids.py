import cf_xarray  # noqa: F401
import numpy as np


def is_meshgrid(coord1, coord2):
    return (
        np.all(coord1[0, :] == coord1[1, :]) and np.all(coord2[:, 0] == coord2[:, 1])
    ) or (np.all(coord1[:, 0] == coord1[:, 1]) and np.all(coord2[0, :] == coord2[1, :]))


def infer_grid_type(ds):
    # grid types (all geographic):
    # - 2d crs (affine transform)
    # - 1d orthogonal (rectilinear)
    # - 2d orthogonal (rectilinear)
    # - 2d curvilinear
    # - "unstructured" 1d
    # - "unstructured" n-d
    #
    # Needs to inspect values (except for 1d and 2d crs), so must allow
    # computing (so calling `infer_grid_type` should be avoided if possible)
    if "crs" in ds.coords and "affine_transform" in ds["crs"].attrs:
        return "2d-crs"

    if "longitude" not in ds.cf or "latitude" not in ds.cf:
        # projected coords or no spatial coords. Raise for now
        raise ValueError("cannot infer the grid type without geographic coordinates")

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
        if is_meshgrid(longitude.data, latitude.data):
            return "2d-rectilinear"
        else:
            # must be curvilinear (this is not entirely accurate, but
            # "nd-unstructured" is really hard to check)
            return "2d-curvilinear"
    else:
        raise ValueError("unable to infer the grid type")
