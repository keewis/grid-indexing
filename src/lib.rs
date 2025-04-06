use pyo3::prelude::*;

mod index;
mod python;
mod rtreeobject;
mod trait_;

use self::python::{create_empty, RTree};

/// A Python module implemented in Rust.
#[pymodule]
fn grid_indexing(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RTree>()?;
    m.add_function(wrap_pyfunction!(create_empty, m)?)?;

    Ok(())
}
