use super::index::CellRTree;
use super::trait_::{AsPolygonArray, AsSparse};
use bincode::{deserialize, serialize};
use geoarrow::array::ArrayBase;
use numpy::{PyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBytes, PyTuple, PyType};
use pyo3_arrow::PyArray;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
#[pyclass]
#[pyo3(module = "grid_indexing")]
pub struct RTree {
    tree: CellRTree,
    shape: Vec<usize>,
}

#[pyfunction]
pub fn create_empty() -> RTree {
    RTree {
        tree: CellRTree::empty(),
        shape: vec![],
    }
}

#[pymethods]
impl RTree {
    #[new]
    #[pyo3(signature=(source_cells, shape=None))]
    pub fn new(source_cells: PyArray, shape: Option<&Bound<PyTuple>>) -> PyResult<Self> {
        let polygons = source_cells.into_polygon_array()?;
        let shape_: Vec<usize> = match shape {
            None => vec![polygons.len()],
            Some(shape) => shape.extract()?,
        };

        Ok(RTree {
            tree: CellRTree::create(polygons),
            shape: shape_,
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
    pub fn from_shapely(_cls: &Bound<'_, PyType>, geoms: &Bound<'_, PyAny>) -> PyResult<Self> {
        Python::with_gil(|py| {
            let geoarrow = PyModule::import(py, "geoarrow.rust.core")?;
            let crs = intern!(py, "epsg:4326");

            let kwargs = [("crs", crs)].into_py_dict(py)?;
            let pyarray = geoms.downcast::<PyArrayDyn<PyObject>>()?;
            let shape: Bound<PyTuple> = PyTuple::new(py, pyarray.shape())?;

            let pyobj = geoarrow
                .getattr("from_shapely")?
                .call((pyarray,), Some(&kwargs))?;

            let array = PyArray::extract_bound(&pyobj)?;

            Self::new(array, Some(&shape))
        })
    }

    #[pyo3(signature=(target_cells, shape=None))]
    pub fn query_overlap(
        &self,
        target_cells: PyArray,
        shape: Option<&Bound<PyTuple>>,
    ) -> PyResult<Py<PyAny>> {
        let polygons = target_cells.into_polygon_array()?;
        let intermediate_shape = (polygons.len(), self.tree.size());

        let target_shape = match shape {
            None => vec![polygons.len()],
            Some(shape) => shape.extract()?,
        };
        let final_shape: Vec<&usize> = target_shape.iter().chain(self.shape.iter()).collect();

        self.tree
            .overlaps(&polygons)
            .into_sparse(intermediate_shape, final_shape)
    }
}
