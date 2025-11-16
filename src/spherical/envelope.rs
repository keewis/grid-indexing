use crate::spherical::point::{max_inline, SphericalPoint};
use num_traits::{Bounded, One, Zero};
use rstar::{Envelope, Point, RTreeObject};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, PartialOrd, Deserialize, Serialize)]
struct SphericalAABB {
    lower: SphericalPoint,
    upper: SphericalPoint,
}

impl SphericalAABB {
    /// Returns the spherical AABB encompassing a single point.
    pub fn from_point(p: SphericalPoint) -> Self {
        Self {
            lower: p.clone(),
            upper: p,
        }
    }

    /// Returns the spherical AABB's lower corner.
    ///
    /// This is the point contained within the AABB with the smallest coordinate value in each
    /// dimension.
    pub fn lower(&self) -> SphericalPoint {
        self.lower.clone()
    }

    /// Returns the spherical AABB's upper corner.
    ///
    /// This is the point contained within the AABB with the largest coordinate value in each
    /// dimension.
    pub fn upper(&self) -> SphericalPoint {
        self.upper.clone()
    }

    /// Creates a new spherical AABB encompassing two points.
    pub fn from_corners(p1: SphericalPoint, p2: SphericalPoint) -> Self {
        Self {
            lower: p1.min_point(&p2),
            upper: p1.max_point(&p2),
        }
    }

    /// Creates a new AABB encompassing a collection of points.
    pub fn from_points<'a, I>(i: I) -> Self
    where
        I: IntoIterator<Item = &'a SphericalPoint> + 'a,
    {
        i.into_iter().fold(
            Self {
                lower: SphericalPoint::from_value(<SphericalPoint as Point>::Scalar::max_value()),
                upper: SphericalPoint::from_value(<SphericalPoint as Point>::Scalar::min_value()),
            },
            |aabb, p| Self {
                lower: aabb.lower.min_point(p),
                upper: aabb.upper.max_point(p),
            },
        )
    }

    /// Returns the point within this AABB closest to a given point.
    ///
    /// If `point` is contained within the AABB, `point` will be returned.
    pub fn min_point(&self, point: &SphericalPoint) -> SphericalPoint {
        self.upper.min_point(&self.lower.max_point(point))
    }

    /// Returns the squared distance to the spherical AABB's [min_point](SphericalAABB::min_point)
    pub fn distance_2(&self, point: &SphericalPoint) -> <SphericalPoint as Point>::Scalar {
        if self.contains_point(point) {
            Zero::zero()
        } else {
            self.min_point(point).distance_2(point)
        }
    }
}

impl Envelope for SphericalAABB {
    type Point = SphericalPoint;

    fn new_empty() -> Self {
        let max = <SphericalPoint as Point>::Scalar::max_value();
        let min = <SphericalPoint as Point>::Scalar::min_value();
        Self {
            lower: SphericalPoint::from_value(max),
            upper: SphericalPoint::from_value(min),
        }
    }

    fn contains_point(&self, point: &SphericalPoint) -> bool {
        self.min_point(point)
            .all_component_wise(point, |l, r| l == r)
    }

    fn contains_envelope(&self, other: &Self) -> bool {
        self.lower.all_component_wise(&other.lower, |l, r| l <= r)
            && self.upper.all_component_wise(&other.upper, |l, r| l >= r)
    }

    fn merge(&mut self, other: &Self) {
        self.lower = self.lower.min_point(&other.lower);
        self.upper = self.upper.max_point(&other.upper);
    }

    fn merged(&self, other: &Self) -> Self {
        SphericalAABB {
            lower: self.lower.min_point(&other.lower),
            upper: self.upper.max_point(&other.upper),
        }
    }

    fn intersects(&self, other: &Self) -> bool {
        self.lower.all_component_wise(&other.upper, |l, r| l <= r)
            && self.upper.all_component_wise(&other.lower, |l, r| l >= r)
    }

    fn area(&self) -> <SphericalPoint as Point>::Scalar {
        let zero = <SphericalPoint as Point>::Scalar::zero();
        let one = <SphericalPoint as Point>::Scalar::one();
        let diag = self.upper.sub(&self.lower);
        diag.fold(one, |acc, cur| max_inline(cur, zero) * acc)
    }

    fn distance_2(&self, point: &SphericalPoint) -> <SphericalPoint as Point>::Scalar {
        self.min_point(point).distance_2(point)
    }

    fn min_max_dist_2(&self, point: &SphericalPoint) -> <SphericalPoint as Point>::Scalar {
        let l = self.lower.sub(point);
        let u = self.upper.sub(point);
        let mut max_diff = (Zero::zero(), Zero::zero(), 0); // diff, min, index
        let mut result = SphericalPoint::new();

        for i in 0..SphericalPoint::DIMENSIONS {
            let mut min = l.nth(i);
            let mut max = u.nth(i);
            max = max * max;
            min = min * min;
            if max < min {
                core::mem::swap(&mut min, &mut max);
            }

            let diff = max - min;
            *result.nth_mut(i) = max;

            if diff >= max_diff.0 {
                max_diff = (diff, min, i);
            }
        }

        *result.nth_mut(max_diff.2) = max_diff.1;
        result.fold(Zero::zero(), |acc, curr| acc + curr)
    }

    fn center(&self) -> Self::Point {
        let one = <Self::Point as Point>::Scalar::one();
        let two = one + one;
        self.lower.component_wise(&self.upper, |x, y| (x + y) / two)
    }

    fn intersection_area(&self, other: &Self) -> <Self::Point as Point>::Scalar {
        SphericalAABB {
            lower: self.lower.max_point(&other.lower),
            upper: self.upper.min_point(&other.upper),
        }
        .area()
    }

    fn perimeter_value(&self) -> <SphericalPoint as Point>::Scalar {
        let diag = self.upper.sub(&self.lower);
        let zero = <SphericalPoint as Point>::Scalar::zero();
        max_inline(diag.fold(zero, |acc, value| acc + value), zero)
    }

    fn sort_envelopes<T: RTreeObject<Envelope = Self>>(axis: usize, envelopes: &mut [T]) {
        envelopes.sort_unstable_by(|l, r| {
            l.envelope()
                .lower
                .nth(axis)
                .partial_cmp(&r.envelope().lower.nth(axis))
                .unwrap()
        });
    }

    fn partition_envelopes<T: RTreeObject<Envelope = Self>>(
        axis: usize,
        envelopes: &mut [T],
        selection_size: usize,
    ) {
        envelopes.select_nth_unstable_by(selection_size, |l, r| {
            l.envelope()
                .lower
                .nth(axis)
                .partial_cmp(&r.envelope().lower.nth(axis))
                .unwrap()
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_point() {
        let p = SphericalPoint::create(10.0, 10.0);
        let actual = SphericalAABB::from_point(p);

        let expected = SphericalAABB {
            lower: SphericalPoint::create(10.0, 10.0),
            upper: SphericalPoint::create(10.0, 10.0),
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_lower() {
        let aabb = SphericalAABB {
            lower: SphericalPoint::create(0.0, 0.0),
            upper: SphericalPoint::create(10.0, 10.0),
        };

        let actual = aabb.lower();
        let expected = SphericalPoint::create(0.0, 0.0);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_upper() {
        let aabb = SphericalAABB {
            lower: SphericalPoint::create(0.0, 0.0),
            upper: SphericalPoint::create(10.0, 10.0),
        };

        let actual = aabb.upper();
        let expected = SphericalPoint::create(10.0, 10.0);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_from_corners() {
        let p1 = SphericalPoint::create(10.0, 0.0);
        let p2 = SphericalPoint::create(0.0, 10.0);

        let actual = SphericalAABB::from_corners(p1, p2);
        let expected = SphericalAABB {
            lower: SphericalPoint::create(0.0, 0.0),
            upper: SphericalPoint::create(10.0, 10.0),
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_from_corners_central() {
        let p1 = SphericalPoint::create(10.0, 10.0);
        let p2 = SphericalPoint::create(350.0, -20.0);

        let actual = SphericalAABB::from_corners(p1, p2);
        let expected = SphericalAABB {
            lower: SphericalPoint::create(350.0, -20.0),
            upper: SphericalPoint::create(10.0, 10.0),
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_from_points() {
        let points = vec![
            SphericalPoint::create(0.0, 0.0),
            SphericalPoint::create(5.0, 0.0),
            SphericalPoint::create(10.0, 0.0),
            SphericalPoint::create(0.0, 5.0),
            SphericalPoint::create(5.0, 5.0),
            SphericalPoint::create(10.0, 5.0),
            SphericalPoint::create(0.0, 10.0),
            SphericalPoint::create(5.0, 10.0),
            SphericalPoint::create(10.0, 10.0),
        ];

        let actual = SphericalAABB::from_points(&points);
        let expected = SphericalAABB {
            lower: SphericalPoint::create(0.0, 0.0),
            upper: SphericalPoint::create(10.0, 10.0),
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_min_point() {
        let aabb = SphericalAABB {
            lower: SphericalPoint::create(350.0, -20.0),
            upper: SphericalPoint::create(10.0, 10.0),
        };

        let point = SphericalPoint::create(15.0, 0.0);
        let expected = SphericalPoint::create(10.0, 0.0);

        let actual = aabb.min_point(&point);

        assert_eq!(actual, expected);

        let point = SphericalPoint::create(320.0, -40.0);
        let expected = SphericalPoint::create(350.0, -20.0);

        let actual = aabb.min_point(&point);

        assert_eq!(actual, expected);

        let point = SphericalPoint::create(359.0, -10.0);
        let expected = SphericalPoint::create(359.0, -10.0);

        let actual = aabb.min_point(&point);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_contains_point() {
        let aabb = SphericalAABB {
            lower: SphericalPoint::create(-10.0, -20.0),
            upper: SphericalPoint::create(10.0, 10.0),
        };

        let point = SphericalPoint::create(-5.0, 5.0);
        let actual = aabb.contains_point(&point);

        assert!(actual);

        let point = SphericalPoint::create(345.0, 5.0);
        let actual = aabb.contains_point(&point);

        assert!(!actual);

        let point = SphericalPoint::create(5.0, -45.0);
        let actual = aabb.contains_point(&point);

        assert!(!actual);
    }
}
