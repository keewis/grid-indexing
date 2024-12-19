use geo::Polygon;
use geoarrow::{
    array::PolygonArray,
    trait_::{ArrayAccessor, NativeScalar},
};
use rstar::{primitives::CachedEnvelope, RTree, RTreeObject};

pub trait RStarRTree<T: RTreeObject> {
    fn create_rstar_rtree(self) -> RTree<T>;
}

impl RStarRTree<CachedEnvelope<Polygon>> for PolygonArray<8> {
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
