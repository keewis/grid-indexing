use geo::Polygon;
use geoarrow::array::PolygonArray;
use pyo3::prelude::*;
use pyo3_geoarrow::PyNativeArray;
use rstar::{primitives::CachedEnvelope, RTree};

use super::trait_::RStarRTree;

#[pyclass]
pub struct Index {
    tree: RTree<CachedEnvelope<Polygon>>,
}

impl Index {
    pub fn create(cell_geoms: PolygonArray<8>) -> Self {
        let rtree = cell_geoms.create_rstar_rtree();

        Index { tree: rtree }
    }
}

#[pymethods]
impl Index {
    #[new]
    pub fn new(py: Python, cell_geoms: PyNativeArray) -> PyResult<Self> {
        let polygons: PolygonArray<8> = (cell_geoms
            .into_inner()
            .into_inner()
            .as_any()
            .downcast_ref::<PolygonArray<8>>()
            .unwrap()
            .clone());

        Ok(Index::create(polygons))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use geo::{LineString, Polygon};
    use geoarrow::array::{PolygonArray, PolygonBuilder};
    use geoarrow::datatypes::Dimension;

    #[test]
    fn create_from_polygon_array() {
        let polygon1 = Polygon::new(
            LineString::from(vec![
                (-5.0, 0.0),
                (-10.0, 5.0),
                (-10.0, 10.0),
                (-5.0, 15.0),
                (5.0, 15.0),
                (10.0, 10.0),
                (10.0, 5.0),
                (5.0, 0.0),
            ]),
            vec![],
        );
        let polygon2 = Polygon::new(
            LineString::from(vec![
                (-90.0, -45.0),
                (-90.0, 45.0),
                (90.0, 45.0),
                (90.0, -45.0),
            ]),
            vec![],
        );

        let mut builder = PolygonBuilder::new(Dimension::XY);
        let _ = builder.push_polygon(Some(&polygon1));
        let _ = builder.push_polygon(Some(&polygon2));
        let array: PolygonArray = builder.finish();

        let index = Index::create(array);
    }
}
