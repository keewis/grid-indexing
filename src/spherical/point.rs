use num_traits::Zero;
use rstar::{Point, RTreeNum};
use serde::{Deserialize, Serialize};

#[derive(Clone, PartialEq, PartialOrd, Debug, Deserialize, Serialize)]
pub struct SphericalPoint {
    lon: f64,
    lat: f64,
}

impl Point for SphericalPoint {
    type Scalar = f64;
    const DIMENSIONS: usize = 2;

    fn generate(mut generator: impl FnMut(usize) -> Self::Scalar) -> Self {
        Self {
            lon: generator(0).rem_euclid(360.0),
            lat: generator(1),
        }
    }

    #[inline]
    fn nth(&self, index: usize) -> Self::Scalar {
        match index {
            0 => self.lon,
            1 => self.lat,
            _ => unreachable!(),
        }
    }

    #[inline]
    fn nth_mut(&mut self, index: usize) -> &mut Self::Scalar {
        match index {
            0 => &mut self.lon,
            1 => &mut self.lat,
            _ => unreachable!(),
        }
    }
}

impl SphericalPoint {
    /// Returns a new Point with all components set to zero.
    pub fn new() -> Self {
        Self::from_value(Zero::zero())
    }

    pub fn create(lon: f64, lat: f64) -> Self {
        Self {
            lon: lon.rem_euclid(360.0),
            lat,
        }
    }

    /// Applies `f` to each pair of components of `self` and `other`.
    pub fn component_wise(
        &self,
        other: &Self,
        mut f: impl FnMut(<Self as Point>::Scalar, <Self as Point>::Scalar) -> <Self as Point>::Scalar,
    ) -> Self {
        Self::generate(|i| f(self.nth(i), other.nth(i)))
    }

    /// Returns whether all pairs of components of `self` and `other` pass test closure `f`. Short circuits if any result is false.
    pub fn all_component_wise(
        &self,
        other: &Self,
        mut f: impl FnMut(<Self as Point>::Scalar, <Self as Point>::Scalar) -> bool,
    ) -> bool {
        (0..Self::DIMENSIONS).all(|i| f(self.nth(i), other.nth(i)))
    }

    /// Returns the dot product of `self` and `rhs`.
    pub fn dot(&self, rhs: &Self) -> <Self as Point>::Scalar {
        self.component_wise(rhs, |l, r| l * r)
            .fold(Zero::zero(), |acc, val| acc + val)
    }

    /// Folds (aka reduces or injects) the Point component wise using `f` and returns the result.
    /// fold() takes two arguments: an initial value, and a closure with two arguments: an 'accumulator', and the value of the current component.
    /// The closure returns the value that the accumulator should have for the next iteration.
    ///
    /// The `start_value` is the value the accumulator will have on the first call of the closure.
    ///
    /// After applying the closure to every component of the Point, fold() returns the accumulator.
    pub fn fold<T>(&self, start_value: T, mut f: impl FnMut(T, <Self as Point>::Scalar) -> T) -> T {
        (0..Self::DIMENSIONS).fold(start_value, |accumulated, i| f(accumulated, self.nth(i)))
    }

    /// Returns a Point with every component set to `value`.
    pub fn from_value(value: <Self as Point>::Scalar) -> Self {
        Self::generate(|_| value)
    }

    /// Returns the Point that is the south-west corner of the box defined by `self` and `other`.
    pub fn min_point(&self, other: &Self) -> Self {
        // For spherical coords: choose the westernmost longitude using the smaller
        // angle between the two points. With that, we define l1 as west of l2
        // if the angle from l1 to l2 is less than 180 degree.
        let min_lat = self.lat.min(other.lat);
        let min_lon = if (other.lon - self.lon).rem_euclid(360.0) <= 180.0 {
            self.lon
        } else {
            other.lon
        };

        Self::create(min_lon, min_lat)
    }

    /// Returns the Point that is the north-east corner of the box defined by `self` and `other`.
    pub fn max_point(&self, other: &Self) -> Self {
        // For spherical coords: choose the easternmost longitude using the smaller
        // angle between the two points. With that, we define l1 as east of l2
        // if the angle from l1 to l2 is greater than 180 degree.
        let max_lat = self.lat.max(other.lat);
        let max_lon = if (other.lon - self.lon).rem_euclid(360.0) > 180.0 {
            self.lon
        } else {
            other.lon
        };

        Self::create(max_lon, max_lat)
    }

    /// Returns the squared length of this Point as if it was a vector.
    pub fn length_2(&self) -> <Self as Point>::Scalar {
        self.fold(Zero::zero(), |acc, cur| cur * cur + acc)
    }

    /// Subtracts `other` from `self` component wise.
    pub fn sub(&self, other: &Self) -> Self {
        self.component_wise(other, |l, r| l - r)
    }

    /// Adds `other` to `self` component wise.
    pub fn add(&self, other: &Self) -> Self {
        self.component_wise(other, |l, r| l + r)
    }

    /// Multiplies `self` with `scalar` component wise.
    pub fn mul(&self, scalar: <Self as Point>::Scalar) -> Self {
        self.map(|coordinate| coordinate * scalar)
    }

    /// Applies `f` to `self` component wise.
    pub fn map(
        &self,
        mut f: impl FnMut(<Self as Point>::Scalar) -> <Self as Point>::Scalar,
    ) -> Self {
        Self::generate(|i| f(self.nth(i)))
    }

    /// Returns the squared distance between `self` and `other`.
    pub fn distance_2(&self, other: &Self) -> <Self as Point>::Scalar {
        self.sub(other).length_2()
    }
}

#[inline]
pub fn min_inline<S>(a: S, b: S) -> S
where
    S: RTreeNum,
{
    if a < b {
        a
    } else {
        b
    }
}

#[inline]
pub fn max_inline<S>(a: S, b: S) -> S
where
    S: RTreeNum,
{
    if a > b {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let actual = SphericalPoint::new();

        assert_eq!(actual.lon, 0f64);
        assert_eq!(actual.lat, 0f64);
    }

    #[test]
    fn test_create() {
        let actual = SphericalPoint::create(2.0, 6.0);

        assert_eq!(actual.lon, 2.0);
        assert_eq!(actual.lat, 6.0);
    }

    #[test]
    fn test_component_wise() {
        let p1 = SphericalPoint::create(10.0, 10.0);
        let p2 = SphericalPoint::create(20.0, 5.0);

        let actual = p1.component_wise(&p2, |a, b| a - b);
        let expected = SphericalPoint::create(-10.0, 5.0);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_all_component_wise() {
        let p1 = SphericalPoint::create(10.0, 10.0);
        let p2 = SphericalPoint::create(5.0, 15.0);

        let actual = p1.all_component_wise(&p2, |a, b| (a - b).abs() < 10.0);

        assert!(actual);
    }

    #[test]
    fn test_min_point() {
        let p1 = SphericalPoint {
            lon: 10.0,
            lat: 0.0,
        };
        let p2 = SphericalPoint {
            lon: 0.0,
            lat: 10.0,
        };

        let actual = p1.min_point(&p2);
        let expected = SphericalPoint { lon: 0.0, lat: 0.0 };

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_min_point2() {
        let p1 = SphericalPoint {
            lon: 350.0,
            lat: 0.0,
        };
        let p2 = SphericalPoint {
            lon: 0.0,
            lat: 10.0,
        };

        let actual = p1.min_point(&p2);
        let expected = SphericalPoint {
            lon: 350.0,
            lat: 0.0,
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_max_point1() {
        let p1 = SphericalPoint {
            lon: 10.0,
            lat: 0.0,
        };
        let p2 = SphericalPoint {
            lon: 0.0,
            lat: 10.0,
        };

        let actual = p1.max_point(&p2);
        let expected = SphericalPoint {
            lon: 10.0,
            lat: 10.0,
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_max_point2() {
        let p1 = SphericalPoint {
            lon: 350.0,
            lat: 0.0,
        };
        let p2 = SphericalPoint {
            lon: 0.0,
            lat: 10.0,
        };

        let actual = p1.max_point(&p2);
        let expected = SphericalPoint {
            lon: 0.0,
            lat: 10.0,
        };

        assert_eq!(actual, expected);
    }
}
