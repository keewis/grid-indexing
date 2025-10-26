use num_traits::Zero;
use rstar::{Point, RTreeNum};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, PartialEq, Debug, Deserialize, Serialize)]
struct SphericalPoint {
    lon: f64,
    lat: f64,
}

impl Point for SphericalPoint {
    type Scalar = f64;
    const DIMENSIONS: usize = 2;

    fn generate(mut generator: impl FnMut(usize) -> Self::Scalar) -> Self {
        Self {
            lon: generator(0),
            lat: generator(1),
        }
    }

    fn nth(&self, index: usize) -> Self::Scalar {
        match index {
            0 => self.lon,
            1 => self.lat,
            _ => unreachable!(),
        }
    }

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
    fn new() -> Self {
        Self::from_value(Zero::zero())
    }

    /// Applies `f` to each pair of components of `self` and `other`.
    fn component_wise(
        &self,
        other: &Self,
        mut f: impl FnMut(<Self as Point>::Scalar, <Self as Point>::Scalar) -> <Self as Point>::Scalar,
    ) -> Self {
        Self::generate(|i| f(self.nth(i), other.nth(i)))
    }

    /// Returns whether all pairs of components of `self` and `other` pass test closure `f`. Short circuits if any result is false.
    fn all_component_wise(
        &self,
        other: &Self,
        mut f: impl FnMut(<Self as Point>::Scalar, <Self as Point>::Scalar) -> bool,
    ) -> bool {
        (0..Self::DIMENSIONS).all(|i| f(self.nth(i), other.nth(i)))
    }

    /// Returns the dot product of `self` and `rhs`.
    fn dot(&self, rhs: &Self) -> <Self as Point>::Scalar {
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
    fn fold<T>(&self, start_value: T, mut f: impl FnMut(T, <Self as Point>::Scalar) -> T) -> T {
        (0..Self::DIMENSIONS).fold(start_value, |accumulated, i| f(accumulated, self.nth(i)))
    }

    /// Returns a Point with every component set to `value`.
    fn from_value(value: <Self as Point>::Scalar) -> Self {
        Self::generate(|_| value)
    }

    /// Returns a Point with each component set to the smallest of each component pair of `self` and `other`.
    fn min_point(&self, other: &Self) -> Self {
        self.component_wise(other, min_inline)
    }

    /// Returns a Point with each component set to the biggest of each component pair of `self` and `other`.
    fn max_point(&self, other: &Self) -> Self {
        self.component_wise(other, max_inline)
    }

    /// Returns the squared length of this Point as if it was a vector.
    fn length_2(&self) -> <Self as Point>::Scalar {
        self.fold(Zero::zero(), |acc, cur| cur * cur + acc)
    }

    /// Subtracts `other` from `self` component wise.
    fn sub(&self, other: &Self) -> Self {
        self.component_wise(other, |l, r| l - r)
    }

    /// Adds `other` to `self` component wise.
    fn add(&self, other: &Self) -> Self {
        self.component_wise(other, |l, r| l + r)
    }

    /// Multiplies `self` with `scalar` component wise.
    fn mul(&self, scalar: <Self as Point>::Scalar) -> Self {
        self.map(|coordinate| coordinate * scalar)
    }

    /// Applies `f` to `self` component wise.
    fn map(&self, mut f: impl FnMut(<Self as Point>::Scalar) -> <Self as Point>::Scalar) -> Self {
        Self::generate(|i| f(self.nth(i)))
    }

    /// Returns the squared distance between `self` and `other`.
    fn distance_2(&self, other: &Self) -> <Self as Point>::Scalar {
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
