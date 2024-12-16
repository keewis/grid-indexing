import xarray as xr
import cf_xarray
import numpy as np
import shapely

def broadcast_spatial_coords(ds):
    [broadcast] = xr.broadcast(
        ds[["lon", "lat"]].drop_indexes(["lon", "lat"]).reset_coords()
    )

    return (
        ds.drop_vars(["lon", "lat"])
        .merge(broadcast)
        .rename_dims({"lon": "x", "lat": "y"})
    )


def bounds_to_geometries(ds):
    lons = ds["lon_bounds"].data
    lats = ds["lat_bounds"].data

    coords = np.stack([lons, lats], axis=-1)

    polygons = shapely.polygons(coords)
    return ds.assign_coords(geometries=(["x", "y"], polygons))


def shapely_to_cf(ds):
    geometries = ds["geometries"]
    result = cf_xarray.shapely_to_cf(geometries)
    return ds.merge(result).drop_vars("geometries")


# open data
ds = xr.tutorial.open_dataset("air_temperature")
# construct polygons
with_geometries = (
    ds
    .pipe(broadcast_spatial_coords)
    .cf.add_bounds(["lon", "lat"])
    .pipe(bounds_to_geometries)
    .drop_vars(["lon_bounds", "lat_bounds"])
)

encoded = (
    with_geometries
    .cf.add_bounds(["lon", "lat"])
    .pipe(bounds_to_geometries)
    .drop_vars(["lon_bounds", "lat_bounds"])
)

encoded.to_netcdf("air_temperature.nc")
