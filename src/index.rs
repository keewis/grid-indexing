use std::ops::Deref;

use geo::{Intersects, Polygon};
use geoarrow::array::{ArrayBase, PolygonArray};
use geoarrow::trait_::{ArrayAccessor, NativeScalar};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyType};
use pyo3_arrow::PyArray;
use rstar::{primitives::CachedEnvelope, RTree, RTreeObject};

use super::trait_::{AsPolygonArray, AsSparse};

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

    fn overlaps_one(&self, cell: Polygon) -> Vec<usize> {
        let bbox = cell.envelope();

        self.tree
            .locate_in_envelope_intersecting(&bbox)
            .filter(|candidate| cell.intersects(candidate.geometry()))
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

#[pymethods]
impl Index {
    #[new]
    pub fn new(source_cells: PyArray) -> PyResult<Self> {
        let polygons = source_cells.into_polygon_array();

        polygons.map(Index::create)
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
