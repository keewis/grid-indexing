use geo::{Polygon, Relate};
use geo_traits::to_geo::ToGeoGeometry;
use geoarrow_array::array::PolygonArray;
use geoarrow_array::GeoArrowArrayAccessor;
use rstar::{RTree, RTreeObject};
use serde::{Deserialize, Serialize};

use super::rtreeobject::NumberedCell;

#[derive(Serialize, Deserialize, Debug)]
pub struct CellRTree {
    tree: RTree<NumberedCell>,
}

impl CellRTree {
    pub fn create(cell_geoms: &PolygonArray) -> Self {
        let cells: Vec<_> = cell_geoms
            .iter()
            .flatten()
            .enumerate()
            .map(|c| NumberedCell::new(c.0, c.1.unwrap().to_geometry().try_into().unwrap()))
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
            .map(|cell| self.overlaps_one(cell.unwrap().to_geometry().try_into().unwrap()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use geo::{coord, Coord, LineString, Polygon, Rect};
    use geoarrow_array::builder::PolygonBuilder;
    use geoarrow_schema::{Dimension, PolygonType};

    fn bbox(ll: Coord, ur: Coord) -> Polygon {
        Rect::new(ll, ur).to_polygon()
    }

    fn polygon(exterior: Vec<(f64, f64)>) -> Polygon {
        Polygon::new(LineString::from(exterior), vec![])
    }

    fn normalize_result(result: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        result
            .into_iter()
            .map(|mut v| {
                v.sort();
                v
            })
            .collect::<Vec<_>>()
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

        let mut builder = PolygonBuilder::new(PolygonType::new(Dimension::XY, Default::default()));
        let _ = builder.push_polygon(Some(&polygon1));
        let _ = builder.push_polygon(Some(&polygon2));
        let array: PolygonArray = builder.finish();

        let index = CellRTree::create(&array);

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

        let polygons = vec![bbox(coord! {x: 0.0, y: 0.0}, coord! {x: 1.0, y: 1.0})];
        let array1 = PolygonBuilder::from_polygons(
            &polygons,
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();
        assert_eq!(CellRTree::create(&array1).size(), 1);

        let polygons = vec![
            bbox(coord! {x: 0.0, y: 0.0}, coord! {x: 1.0, y: 1.0}),
            bbox(coord! {x: 1.0, y: 0.0}, coord! {x: 2.0, y: 1.0}),
            bbox(coord! {x: 0.0, y: 1.0}, coord! {x: 1.0, y: 2.0}),
            bbox(coord! {x: 1.0, y: 1.0}, coord! {x: 2.0, y: 2.0}),
        ];
        let array2 = PolygonBuilder::from_polygons(
            &polygons,
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();
        assert_eq!(CellRTree::create(&array2).size(), 4);
    }

    /// check the basic functionality of the overlap search
    #[test]
    fn test_overlaps_rectilinear() {
        let polygons = vec![
            bbox(coord! {x: 0.0, y: 0.0}, coord! {x: 1.0, y: 1.0}),
            bbox(coord! {x: 1.0, y: 0.0}, coord! {x: 2.0, y: 1.0}),
            bbox(coord! {x: 0.0, y: 1.0}, coord! {x: 1.0, y: 2.0}),
            bbox(coord! {x: 1.0, y: 1.0}, coord! {x: 2.0, y: 2.0}),
        ];
        let source = PolygonBuilder::from_polygons(
            &polygons,
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let index = CellRTree::create(&source);

        let polygons = vec![
            bbox(coord! {x: 0.2, y: 0.0}, coord! {x: 1.5, y: 1.0}),
            bbox(coord! {x: 0.6, y: 1.2}, coord! {x: 0.9, y: 1.8}),
            bbox(coord! {x: 0.3, y: 0.2}, coord! {x: 1.3, y: 1.6}),
            bbox(coord! {x: 2.1, y: 2.3}, coord! {x: 2.7, y: 3.1}),
        ];
        let target = PolygonBuilder::from_polygons(
            &polygons,
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let actual = index.overlaps(&target);
        let expected: Vec<Vec<usize>> = vec![vec![0, 1], vec![2], vec![0, 1, 2, 3], vec![]];
        assert_eq!(normalize_result(actual), expected);
    }

    /// check touches
    #[test]
    fn test_overlaps_touches() {
        let polygons = vec![
            bbox(coord! {x: 0.0, y: 0.0}, coord! {x: 1.0, y: 1.0}),
            bbox(coord! {x: 1.0, y: 0.0}, coord! {x: 2.0, y: 1.0}),
        ];
        let source = PolygonBuilder::from_polygons(
            &polygons,
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let index = CellRTree::create(&source);

        let polygons = vec![
            bbox(coord! {x: 0.0, y: 1.0}, coord! {x: 1.0, y: 2.0}),
            bbox(coord! {x: 2.0, y: 0.0}, coord! {x: 3.0, y: 1.0}),
        ];
        let target = PolygonBuilder::from_polygons(
            &polygons,
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let actual = index.overlaps(&target);
        let expected: Vec<Vec<usize>> = vec![vec![], vec![]];
        assert_eq!(normalize_result(actual), expected);
    }

    /// check that the additional filter works properly
    #[test]
    fn test_overlaps_tilted() {
        let polygons = vec![
            polygon(vec![
                (0.0, 0.0),
                (1.0, 0.0),
                (1.5, 1.0),
                (0.5, 1.0),
                (0.0, 0.0),
            ]),
            polygon(vec![
                (1.0, 0.0),
                (2.0, 0.0),
                (2.5, 1.0),
                (1.5, 1.0),
                (1.0, 0.0),
            ]),
        ];
        let source = PolygonBuilder::from_polygons(
            &polygons,
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let index = CellRTree::create(&source);

        let polygons = vec![
            bbox(coord! {x: -1.0, y: 0.8}, coord! {x: 0.2, y: 1.5}),
            bbox(coord! {x: 2.4, y: 0.9}, coord! {x: 3.0, y: 2.0}),
        ];
        let target = PolygonBuilder::from_polygons(
            &polygons,
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let actual = index.overlaps(&target);
        let expected: Vec<Vec<usize>> = vec![vec![], vec![1]];
        assert_eq!(normalize_result(actual), expected);
    }
}
