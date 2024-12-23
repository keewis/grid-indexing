import shapely


def as_parts(geoms):
    type_, coords, (ring_offsets, geom_offsets) = shapely.to_ragged_array(geoms)

    if type_ != shapely.GeometryType.POLYGON:
        raise ValueError(f"only polygons are supported (got {type_})")

    geom_offsets = geom_offsets.astype("int64")
    ring_offsets = geom_offsets.astype("int64")

    return coords, geom_offsets, ring_offsets
