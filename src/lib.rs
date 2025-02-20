use pyo3::prelude::*;

mod index;
mod trait_;

use self::index::{create_empty, Index};

/// A Python module implemented in Rust.
#[pymodule]
fn grid_indexing(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Index>()?;
    m.add_function(wrap_pyfunction!(create_empty, m)?)?;

    Ok(())
}
