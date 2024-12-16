use geoarrow::array::PolygonArray;
use rstar::RTree;

struct Index {
    rtree: RTree;
}

impl Index {
    fn create(geoms: PolygonArray) -> Self {
        let rtree = geoms.create_rtree();

        return Index {
            rtree: rtree,
        };
    }
}
