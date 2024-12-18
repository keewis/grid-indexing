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

#[cfg(test)]
mod tests {
    use super::*;

    use geoarrow::array::PointArray;
    use geo;
    use geo::Dimension;

    #[test]
    fn create_from_point_array() {
        let point1 = geo::point!(x: 30, y: 20);
        let point2 = geo::point!(x: 10, y: -50);
        let array: PointArray = (vec![point1, point2].as_slice(), Dimension::XY).into();

        let index = Index::create(array);
    }
}
