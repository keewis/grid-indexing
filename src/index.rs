use geo::{Polygon, Relate};
use geoarrow::array::PolygonArray;
use geoarrow::trait_::{ArrayAccessor, NativeScalar};
use rstar::{RTree, RTreeObject};
use serde::{Deserialize, Serialize};

use super::rtreeobject::NumberedCell;

#[derive(Serialize, Deserialize, Debug)]
pub struct CellRTree {
    tree: RTree<NumberedCell>,
}

impl CellRTree {
    pub fn create(cell_geoms: PolygonArray) -> Self {
        let cells: Vec<_> = cell_geoms
            .iter()
            .flatten()
            .enumerate()
            .map(|c| NumberedCell::new(c.0, c.1.to_geo()))
            .collect();

        CellRTree {
            tree: RTree::bulk_load_with_params(cells),
        }
    }

    pub fn empty() -> Self {
        CellRTree { tree: RTree::new() }
    }

    pub fn size(&self) -> usize {
        self.tree.size()
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
            .map(|match_| match_.index())
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

#[cfg(test)]
mod tests {
    use super::*;

    use geo::{coord, Coord, LineString, Polygon, Rect};
    use geoarrow::array::PolygonBuilder;
    use geoarrow::datatypes::Dimension;

    fn bbox(ll: Coord, ur: Coord) -> Option<Polygon> {
        Some(Rect::new(ll, ur).to_polygon())
    }

    #[test]
    fn test_create_from_polygon_array() {
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

        let index = CellRTree::create(array);

        assert_eq!(index.tree.size(), 2);
    }

    #[test]
    fn test_empty() {
        let index = CellRTree::empty();
        assert_eq!(index.tree.size(), 0);
    }

    #[test]
    fn test_size() {
        assert_eq!(CellRTree::empty().size(), 0);

        let array1 = PolygonArray::from((
            vec![bbox(coord! {x: 0.0, y: 0.0}, coord! {x: 1.0, y: 1.0})],
            Dimension::XY,
        ));
        assert_eq!(CellRTree::create(array1).size(), 1);

        let array2 = PolygonArray::from((
            vec![
                bbox(coord! {x:0.0, y: 0.0}, coord! {x: 1.0, y: 1.0}),
                bbox(coord! {x:1.0, y: 0.0}, coord! {x: 2.0, y: 1.0}),
                bbox(coord! {x:0.0, y: 1.0}, coord! {x: 1.0, y: 2.0}),
                bbox(coord! {x:1.0, y: 1.0}, coord! {x: 2.0, y: 2.0}),
            ],
            Dimension::XY,
        ));
        assert_eq!(CellRTree::create(array2).size(), 4);
    }
}
