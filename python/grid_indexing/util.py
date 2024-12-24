import shapely


def as_parts(geoms):
    result = shapely.to_ragged_array(geoms)

    try:
        type_, coords, (ring_offsets, geom_offsets) = result
    except ValueError:
        type_, *_ = result

    if type_ != shapely.GeometryType.POLYGON:
        raise ValueError(f"only polygons are supported (got {type_.name})")

    geom_offsets = geom_offsets.astype("int64")
    ring_offsets = ring_offsets.astype("int64")

    return coords, geom_offsets, ring_offsets
