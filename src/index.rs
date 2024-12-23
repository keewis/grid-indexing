use std::ops::Deref;

use geo::{Intersects, Polygon};
use geoarrow::array::{ArrayBase, PolygonArray};
use geoarrow::trait_::{ArrayAccessor, NativeScalar};
use numpy::PyArray1;
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3_arrow::PyArray;
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
                PyArray1::from_vec(py, indices),
                PyArray1::from_vec(py, counts),
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

trait AsPolygonArray {
    fn into_polygon_array(self) -> PyResult<PolygonArray>;
}

impl AsPolygonArray for PyArray {
    fn into_polygon_array(self) -> PyResult<PolygonArray> {
        let (array, field) = self.into_inner();

        let polygons = PolygonArray::try_from((array.as_ref(), field.as_ref()));

        match polygons {
            Ok(p) => Ok(p),
            Err(error) => Err(PyOSError::new_err(error.to_string())),
        }
    }
}

#[pymethods]
impl Index {
    #[new]
    pub fn new(source_cells: PyArray) -> PyResult<Self> {
        let polygons = source_cells.into_polygon_array();

        polygons.map(Index::create)
    }

    pub fn query_overlap(&self, target_cells: PyArray) -> PyResult<Py<PyAny>> {
        let polygons = target_cells.into_polygon_array()?;

        self.overlaps(&polygons)
            .into_sparse((polygons.len(), self.tree.size()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use geo::{LineString, Polygon};
    use geoarrow::array::PolygonBuilder;
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

        let _index = Index::create(array);
    }
}
