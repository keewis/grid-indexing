use geo::Polygon;
use geoarrow::{
    array::PolygonArray, scalar::Polygon as GeoArrowPolygon, trait_::ArrayAccessor,
    trait_::NativeScalar,
};
use pyo3::prelude::*;
use rstar::{primitives::CachedEnvelope, RTree, RTreeObject};

trait RStarRTree<T: RTreeObject> {
    fn create_rstar_rtree(self) -> RTree<T>;
}

impl RStarRTree<CachedEnvelope<Polygon>> for PolygonArray {
    fn create_rstar_rtree(self) -> RTree<CachedEnvelope<Polygon>> {
        let cells: Vec<_> = self
            .iter()
            .flatten()
            .map(|cell| {
                let geom: Polygon = cell.to_geo();
                CachedEnvelope::<Polygon>::new(geom.into())
            })
            .collect();

        return RTree::bulk_load_with_params(cells);
    }
}

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
