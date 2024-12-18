use geoarrow::ArrayBase;
use rstar::{RTree, RTreeObject};

pub struct Index<T>
where T: RTreeObject {
    rtree: RTree<T>;
}

impl Index {
    fn create(geoms: PolygonArray) -> Self {
        let rtree = geoms.create_rtree();

        return Index {
            rtree: rtree,
        };
    }
}
