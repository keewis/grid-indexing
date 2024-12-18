use geo::Polygon;
use geoarrow::array::PolygonArray;
use rstar::{primitives::CachedEnvelope, RTree};

use super::trait_::RStarRTree;

pub struct Index {
    rtree: RTree<CachedEnvelope<Polygon>>,
}

impl Index {
    fn create(cell_geoms: PolygonArray) -> Self {
        let rtree = cell_geoms.create_rstar_rtree();

        return Index { rtree: rtree };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use geo;
    use geo::Dimension;
    use geoarrow::array::PolygonArray;

    #[test]
    fn create_from_polygon_array() {
        let polygon1 = geo::polygon!(x: 30, y: 20);
        let polygon2 = geo::polygon!(x: 10, y: -50);
        let array: PolygonArray = (vec![polygon, polygon2].as_slice(), Dimension::XY).into();

        let index = Index::create(array);
    }
}
