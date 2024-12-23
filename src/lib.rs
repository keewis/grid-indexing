use pyo3::prelude::*;

mod index;

use self::index::Index;

/// A Python module implemented in Rust.
#[pymodule]
fn grid_indexing(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Index>()?;
    Ok(())
}
