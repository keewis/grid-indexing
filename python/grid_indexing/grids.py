from typing import Literal

import cf_xarray  # noqa: F401
import geoarrow.rust.core as geoarrow
import numpy as np
import xarray as xr
from numpy.typing import ArrayLike


def is_meshgrid(coord1: ArrayLike, coord2: ArrayLike):
    return (
        np.all(coord1[0, :] == coord1[1, :]) and np.all(coord2[:, 0] == coord2[:, 1])
    ) or (np.all(coord1[:, 0] == coord1[:, 1]) and np.all(coord2[0, :] == coord2[1, :]))


def as_components(boundaries):
    vertices = np.concatenate([boundaries, boundaries[..., :1, :]], axis=-2)

    coords = np.reshape(vertices, (-1, 2))

    coords_per_pixel = vertices.shape[-2]
    geom_offsets = np.arange(np.prod(vertices.shape[:-2]) + 1, dtype="int32")
    ring_offsets = geom_offsets * coords_per_pixel

    return coords.astype("float64"), geom_offsets, ring_offsets


def infer_grid_type(ds: xr.Dataset):
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
    if ds.cf.grid_mapping_names and "GeoTransform" in ds.cf["grid_mapping"].attrs:
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


def infer_cell_geometries(
    ds: xr.Dataset,
    *,
    grid_type: str = "infer",
    coords: Literal["infer"] | list[str] = "infer",
):
    # TODO: short-circuit for existing geometries
    if grid_type == "infer":
        grid_type = infer_grid_type(ds)

    if grid_type == "2d-crs":
        raise NotImplementedError(
            "inferring cell geometries is not yet implemented"
            " for geotransform-based grids"
        )
    elif grid_type == "1d-unstructured":
        raise ValueError(
            "inferring cell geometries is not implemented for unstructured grids."
            " This is hard to get right in all cases, so please manually"
            " create the geometries."
        )

    if coords == "infer":
        coords = ["longitude", "latitude"]
        if any(coord not in ds.cf.coordinates for coord in coords):
            raise ValueError(
                "cannot infer geographic coordinates. Please add them"
                " or explicitly pass the names if they exist."
            )

    coords_only = ds.cf[coords]
    if grid_type == "1d-rectilinear":
        coord_names = [ds.cf.coordinates[name][0] for name in coords]
        [broadcasted] = xr.broadcast(
            coords_only.drop_indexes(coord_names).reset_coords(coord_names)
        )
        coords_only = broadcasted.set_coords(coord_names)

    if any(coord not in coords_only.cf.bounds for coord in coords):
        with_bounds = coords_only.cf.add_bounds(coords)
    else:
        with_bounds = coords_only

    bound_names = [with_bounds.cf.bounds[name][0] for name in coords]
    boundaries = np.stack([with_bounds.variables[n].data for n in bound_names], axis=-1)

    return geoarrow.polygons(*as_components(boundaries), crs=4326)
