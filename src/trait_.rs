use geoarrow::array::PolygonArray;
use numpy::PyArray1;
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3_arrow::PyArray;

pub trait AsSparse {
    fn into_sparse(self, shape: (usize, usize), final_shape: Vec<&usize>) -> PyResult<PyObject>;
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

impl AsSparse for Vec<Vec<usize>> {
    fn into_sparse(self, shape: (usize, usize), final_shape: Vec<&usize>) -> PyResult<PyObject> {
        let counts: Vec<i64> = index_pointer(&self).into_iter().map(|v| v as i64).collect();
        let indices: Vec<i64> = self.into_iter().flatten().map(|v| v as i64).collect();
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

            Ok(sparse
                .getattr("GCXS")?
                .call(args, Some(&kwargs))?
                .getattr("reshape")?
                .call1((final_shape,))?
                .unbind())
        })
    }
}

pub trait AsPolygonArray {
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
