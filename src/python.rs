use super::index::CellRTree;
use super::trait_::{AsPolygonArray, AsSparse};
use bincode::{deserialize, serialize};
use geoarrow::array::ArrayBase;
use pyo3::exceptions::PyRuntimeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBytes, PyType};
use pyo3_arrow::PyArray;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
#[pyclass]
#[pyo3(module = "grid_indexing")]
pub struct RTree {
    tree: CellRTree,
}

#[pyfunction]
pub fn create_empty() -> RTree {
    RTree {
        tree: CellRTree::empty(),
    }
}

#[pymethods]
impl RTree {
    #[new]
    pub fn new(source_cells: PyArray) -> PyResult<Self> {
        let polygons = source_cells.into_polygon_array();

        polygons.map(|arr| RTree {
            tree: CellRTree::create(arr),
        })
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

        self.tree
            .overlaps(&polygons)
            .into_sparse((polygons.len(), self.tree.size()))
    }
}
