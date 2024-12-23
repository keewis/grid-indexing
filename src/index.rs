use std::ops::Deref;

use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use geo::{Intersects, Polygon};
use geoarrow::array::{
    metadata::ArrayMetadata, ArrayBase, CoordBuffer, InterleavedCoordBuffer, PolygonArray,
};
use geoarrow::datatypes::Dimension;
use geoarrow::error::GeoArrowError;
use geoarrow::trait_::{ArrayAccessor, NativeScalar};
use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
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

fn index_pointer<T>(array: &[Vec<T>]) -> Vec<usize> {
    let first_element: Vec<Vec<T>> = vec![vec![]];

    first_element
        .iter()
        .chain(array.iter())
        .scan(0, |cur, el| {
            *cur += el.len();
            Some(*cur)
        })
        .collect()
}

trait AsSparse {
    fn into_sparse(self, shape: (usize, usize)) -> PyResult<PyObject>;
}

impl AsSparse for Vec<Vec<usize>> {
    fn into_sparse(self, shape: (usize, usize)) -> PyResult<PyObject> {
        let counts = index_pointer(&self);
        let indices: Vec<usize> = self.into_iter().flatten().collect();
        let data = [true].repeat(indices.len());

        Python::with_gil(|py| {
            let arg = (
                PyArray1::from_vec(py, data),
                PyArray1::from_iter(py, indices.into_iter().map(|v| v as i64)),
                PyArray1::from_iter(py, counts.into_iter().map(|v| v as i64)),
            );
            let sparse = PyModule::import(py, "sparse")?;
            let args = (arg,);
            let kwargs = [
                ("shape", shape.into_pyobject(py)?.as_any()),
                ("compressed_axes", vec![0].into_pyobject(py)?.as_any()),
            ]
            .into_py_dict(py)?;

            Ok(sparse.getattr("GCXS")?.call(args, Some(&kwargs))?.unbind())
        })
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

    pub fn overlaps(&self, cells: &PolygonArray) -> Vec<Vec<usize>> {
        // steps
        // 1. for each polygon, compute (cached) envelopes
        // 2. query the tree using the envelopes
        // 3. filter using the geo predicates
        // 4. assemble into a sparse array
        cells
            .iter()
            .flatten()
            .map(|cell| self.query_overlaps_one(cell.to_geo()))
            .collect()
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

    pub fn query_overlap(
        &self,
        coords: PyReadonlyArray2<'_, f64>,
        geom_offsets: Vec<i32>,
        ring_offsets: Vec<i32>,
    ) -> PyResult<Py<PyAny>> {
        let cells = PolygonArray::from_pyarray(coords, geom_offsets, ring_offsets);

        match cells {
            Ok(c) => self.overlaps(&c).into_sparse((c.len(), self.tree.size())),
            Err(error) => Err(PyOSError::new_err(error.to_string())),
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
