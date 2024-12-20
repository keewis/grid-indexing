use arrow_buffer::OffsetBuffer;
use geo::Polygon;
use geoarrow::array::{metadata::ArrayMetadata, CoordBuffer, InterleavedCoordBuffer, PolygonArray};
use geoarrow::datatypes::Dimension;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use rstar::{primitives::CachedEnvelope, RTree};

use super::trait_::RStarRTree;

#[pyclass]
pub struct Index {
    tree: RTree<CachedEnvelope<Polygon>>,
}

impl Index {
    pub fn create(cell_geoms: PolygonArray) -> Self {
        let rtree = cell_geoms.create_rstar_rtree();

        Index { tree: rtree }
    }
}

#[pymethods]
impl Index {
    #[new]
    pub fn new<'py>(
        coords: PyReadonlyArray2<'py, f64>,
        geometry_offsets: PyReadonlyArray1<'py, usize>,
        ring_offsets: PyReadonlyArray1<'py, usize>,
    ) -> PyResult<Self> {
        let coord_buffer = CoordBuffer::Interleaved(InterleavedCoordBuffer::new(
            coords.as_array().flatten().to_vec().into(),
            Dimension::XY,
        ));
        let geom_offset_buffer = OffsetBuffer::from_lengths(geometry_offsets.as_array().to_vec());
        let ring_offset_buffer = OffsetBuffer::from_lengths(ring_offsets.as_array().to_vec());

        let polygons = PolygonArray::try_new(
            coord_buffer,
            geom_offset_buffer,
            ring_offset_buffer,
            None,
            ArrayMetadata::from_authority_code("epsg:4326".to_string()).into(),
        );

        match polygons {
            Err(error) => Err(PyOSError::new_err(error.to_string())),
            Ok(p) => Ok(Index::create(p)),
        }
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

        let mut builder = PolygonBuilder::new();
        let _ = builder.push_polygon(Some(&polygon1));
        let _ = builder.push_polygon(Some(&polygon2));
        let array: PolygonArray<8> = builder.finish();

        let index = Index::create(array);
    }
}
