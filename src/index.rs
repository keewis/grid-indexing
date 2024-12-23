use std::ops::Deref;

use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use geo::{Intersects, Polygon};
use geoarrow::array::{metadata::ArrayMetadata, CoordBuffer, InterleavedCoordBuffer, PolygonArray};
use geoarrow::datatypes::Dimension;
use geoarrow::error::GeoArrowError;
use geoarrow::trait_::{ArrayAccessor, NativeScalar};
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use rstar::{primitives::CachedEnvelope, RTree, RTreeObject};

#[derive(Debug)]
pub struct NumberedCell {
    index: usize,
    envelope: CachedEnvelope<Polygon>,
}

impl NumberedCell {
    pub fn new(idx: usize, geom: Polygon) -> Self {
        NumberedCell {
            index: idx,
            envelope: CachedEnvelope::<Polygon>::new(geom),
        }
    }

    pub fn geometry(&self) -> &Polygon {
        self.envelope.deref()
    }
}

impl RTreeObject for NumberedCell {
    type Envelope = <CachedEnvelope<Polygon> as RTreeObject>::Envelope;

    fn envelope(&self) -> Self::Envelope {
        self.envelope.envelope()
    }
}

#[pyclass]
pub struct Index {
    tree: RTree<NumberedCell>,
}

trait FromPyArray {
    fn from_pyarray(
        coords: PyReadonlyArray2<'_, f64>,
        geometry_offsets: Vec<i32>,
        ring_offsets: Vec<i32>,
    ) -> Result<PolygonArray, GeoArrowError>;
}

impl FromPyArray for PolygonArray {
    fn from_pyarray(
        coords: PyReadonlyArray2<'_, f64>,
        geometry_offsets: Vec<i32>,
        ring_offsets: Vec<i32>,
    ) -> Result<PolygonArray, GeoArrowError> {
        let coord_buffer = CoordBuffer::Interleaved(InterleavedCoordBuffer::new(
            ScalarBuffer::from(coords.as_array().flatten().to_vec()),
            Dimension::XY,
        ));

        let geom_offset_buffer = OffsetBuffer::new(ScalarBuffer::from(geometry_offsets));
        let ring_offset_buffer = OffsetBuffer::new(ScalarBuffer::from(ring_offsets));

        PolygonArray::try_new(
            coord_buffer,
            geom_offset_buffer,
            ring_offset_buffer,
            None,
            ArrayMetadata::from_authority_code("epsg:4326".to_string()).into(),
        )
    }
}

impl Index {
    pub fn create(cell_geoms: PolygonArray) -> Self {
        let cells: Vec<_> = cell_geoms
            .iter()
            .flatten()
            .enumerate()
            .map(|c| NumberedCell::new(c.0, c.1.to_geo()))
            .collect();

        Index {
            tree: RTree::bulk_load_with_params(cells),
        }
    }

    fn query_overlaps_one(&self, cell: Polygon) -> Vec<usize> {
        let bbox = cell.envelope();

        self.tree
            .locate_in_envelope_intersecting(&bbox)
            .filter(|candidate| cell.intersects(candidate.geometry()))
            .map(|match_| match_.index)
            .collect()
    }

    pub fn overlaps(&self, cells: PolygonArray) -> Vec<i32> {
        // steps
        // 1. for each polygon, compute (cached) envelopes
        // 2. query the tree using the envelopes
        // 3. filter using the geo predicates
        // 4. assemble into a sparse array
        let results: Vec<_> = cells
            .iter()
            .flatten()
            .map(|cell| self.query_overlaps_one(cell.to_geo()))
            .collect();
        println!("result: {:?}", results);

        Vec::<i32>::new()
    }
}

#[pymethods]
impl Index {
    #[new]
    pub fn new(
        coords: PyReadonlyArray2<'_, f64>,
        geometry_offsets: Vec<i32>,
        ring_offsets: Vec<i32>,
    ) -> PyResult<Self> {
        let polygons = PolygonArray::from_pyarray(coords, geometry_offsets, ring_offsets);

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

        let _index = Index::create(array);
    }
}
