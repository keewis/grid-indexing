use bincode::{deserialize, serialize};
use std::mem;
use std::ops::Deref;

use geo::{CoordsIter, Polygon, Relate};
use geoarrow::array::{ArrayBase, PolygonArray};
use geoarrow::trait_::{ArrayAccessor, NativeScalar};
use pyo3::exceptions::PyRuntimeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBytes, PyType};
use pyo3_arrow::PyArray;
use rstar::{primitives::CachedEnvelope, ParentNode, RTree, RTreeNode, RTreeObject};
use serde::{Deserialize, Serialize};

use super::trait_::{AsPolygonArray, AsSparse};

#[derive(Serialize, Deserialize, Debug)]
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

    pub fn num_bytes(&self) -> usize {
        let mut nbytes = mem::size_of_val(self);
        nbytes += (*self.envelope).coords_count() * 2 * mem::size_of::<f64>();

        nbytes
    }
}

impl RTreeObject for NumberedCell {
    type Envelope = <CachedEnvelope<Polygon> as RTreeObject>::Envelope;

    fn envelope(&self) -> Self::Envelope {
        self.envelope.envelope()
    }
}

struct LeafReference<'a> {
    reference: &'a NumberedCell,
}

struct ParentNodeReference<'a> {
    reference: &'a ParentNode<NumberedCell>,
}

enum NodeReference<'a> {
    Node(ParentNodeReference<'a>),
    Leaf(LeafReference<'a>),
}

fn estimate_tree_size(tree: &RTree<NumberedCell>) -> usize {
    let mut nbytes = mem::size_of_val(tree);

    let mut to_visit: Vec<NodeReference> = vec![NodeReference::Node(ParentNodeReference {
        reference: tree.root(),
    })];
    while let Some(item) = to_visit.pop() {
        // Iteration:
        // - pop the next item
        // - if the popped item was a parent node, extend the queue
        // - return the popped item
        match item {
            NodeReference::Node(parent) => {
                to_visit.extend(
                    parent
                        .reference
                        .children()
                        .iter()
                        .map(|n| match n {
                            RTreeNode::Parent(p) => {
                                NodeReference::Node(ParentNodeReference { reference: p })
                            }
                            RTreeNode::Leaf(l) => {
                                NodeReference::Leaf(LeafReference { reference: l })
                            }
                        })
                        .collect::<Vec<NodeReference>>(),
                );

                nbytes += mem::size_of_val(parent.reference);
            }
            NodeReference::Leaf(leaf) => {
                nbytes += leaf.reference.num_bytes();
            }
        };
    }

    nbytes
}

#[derive(Serialize, Deserialize, Debug)]
#[pyclass]
#[pyo3(module = "grid_indexing")]
pub struct Index {
    tree: RTree<NumberedCell>,
    num_bytes: usize,
}

impl Index {
    fn from_tree(tree: RTree<NumberedCell>) -> Self {
        let nbytes = estimate_tree_size(&tree);
        Index {
            tree,
            num_bytes: nbytes,
        }
    }

    pub fn create(cell_geoms: PolygonArray) -> Self {
        let cells: Vec<_> = cell_geoms
            .iter()
            .flatten()
            .enumerate()
            .map(|c| NumberedCell::new(c.0, c.1.to_geo()))
            .collect();

        Self::from_tree(RTree::bulk_load_with_params(cells))
    }

    fn overlaps_one(&self, cell: Polygon) -> Vec<usize> {
        let bbox = cell.envelope();

        self.tree
            .locate_in_envelope_intersecting(&bbox)
            .filter(|candidate| {
                let relate = cell.relate(candidate.geometry());
                // We're looking for anything that fully covers / is covered by / intersects
                // (no touching)
                relate.is_intersects() && !relate.is_touches()
            })
            .map(|match_| match_.index)
            .collect()
    }

    pub fn overlaps(&self, cells: &PolygonArray) -> Vec<Vec<usize>> {
        cells
            .iter()
            .flatten()
            .map(|cell| self.overlaps_one(cell.to_geo()))
            .collect()
    }
}

#[pyfunction]
pub fn create_empty() -> Index {
    Index::from_tree(RTree::new())
}

#[pymethods]
impl Index {
    #[new]
    pub fn new(source_cells: PyArray) -> PyResult<Self> {
        let polygons = source_cells.into_polygon_array();

        polygons.map(Index::create)
    }

    pub fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        // Deserialize the data contained in the PyBytes object
        // and update the struct with the deserialized values.
        *self = deserialize(state).map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

        Ok(())
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        // Serialize the struct and return a PyBytes object
        // containing the serialized data.
        let serialized = serialize(&self).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let bytes = PyBytes::new(py, &serialized);
        Ok(bytes)
    }

    pub fn __reduce__(&self, py: Python) -> PyResult<(PyObject, PyObject, PyObject)> {
        let create = py.import("grid_indexing")?.getattr("create_empty")?;
        let args = ();
        let state = self.__getstate__(py)?;

        Ok((
            create.into_pyobject(py)?.unbind().into_any(),
            args.into_pyobject(py)?.unbind().into_any(),
            state.into_pyobject(py)?.unbind().into_any(),
        ))
    }

    #[getter]
    pub fn nbytes(&self) -> PyResult<usize> {
        Ok(self.num_bytes)
    }

    #[classmethod]
    pub fn from_shapely(_cls: &Bound<'_, PyType>, geoms: &Bound<PyAny>) -> PyResult<Self> {
        let array = Python::with_gil(|py| {
            let geoarrow = PyModule::import(py, "geoarrow.rust.core")?;
            let crs = intern!(py, "epsg:4326");

            let kwargs = [("crs", crs)].into_py_dict(py)?;

            let pyobj = geoarrow
                .getattr("from_shapely")?
                .call((geoms,), Some(&kwargs))?;

            PyArray::extract_bound(&pyobj)
        })?;

        Self::new(array)
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
