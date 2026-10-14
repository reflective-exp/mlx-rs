//! Implements bindings for the sorting ops.

use crate::{Array, Stream, error::Result, utils::guard::Guarded};

/// Which side of a run of equal values [`search_sorted`] inserts at.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SearchSide {
    /// Insert before equal values (the first suitable index).
    #[default]
    Left,

    /// Insert after equal values (the last suitable index).
    Right,
}

impl SearchSide {
    fn as_c_str(self) -> &'static std::ffi::CStr {
        match self {
            Self::Left => c"left",
            Self::Right => c"right",
        }
    }
}

/// Find the indices at which `values` would be inserted into `sorted` to keep it sorted.
///
/// The result has the shape of `values` and dtype `uint32`. `sorted` must be one-dimensional and
/// sorted in ascending order; this is not checked, and an unsorted input gives unspecified results.
///
/// # Params
///
/// - `sorted`: one-dimensional input sorted in ascending order
/// - `values`: values to locate
/// - `side`: where to insert among equal values -- defaults to [`SearchSide::Left`]
///
/// # Example
///
/// ```rust
/// use mlx_rs::{array, ops::*};
///
/// let sorted = array!([1, 2, 2, 3]);
/// let left = search_sorted(&sorted, array!([2]), None).unwrap();
/// let right = search_sorted(&sorted, array!([2]), SearchSide::Right).unwrap();
/// assert_eq!(left.as_slice::<u32>(), &[1]);
/// assert_eq!(right.as_slice::<u32>(), &[3]);
/// ```
pub fn search_sorted(
    sorted: impl AsRef<Array>,
    values: impl AsRef<Array>,
    side: impl Into<Option<SearchSide>>,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let side = side.into().unwrap_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_searchsorted(
            res,
            sorted.as_ref().as_ptr(),
            values.as_ref().as_ptr(),
            side.as_c_str().as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Returns a sorted copy of the array. Returns an error if the arguments are invalid.
///
/// # Params
///
/// - `array`: input array
/// - `axis`: axis to sort over
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let axis = 0;
/// let result = sort_axis(&a, axis);
/// ```
pub fn sort_axis(a: impl AsRef<Array>, axis: i32) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_sort_axis(res, a.as_ref().as_ptr(), axis, stream.as_ref().as_ptr())
    })
}

/// Returns a sorted copy of the flattened array. Returns an error if the arguments are invalid.
///
/// # Params
///
/// - `array`: input array
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let result = sort(&a);
/// ```
pub fn sort(a: impl AsRef<Array>) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_sort(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Returns the indices that sort the array. Returns an error if the arguments are invalid.
///
/// # Params
///
/// - `a`: The array to sort.
/// - `axis`: axis to sort over
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let axis = 0;
/// let result = argsort_axis(&a, axis);
/// ```
pub fn argsort_axis(a: impl AsRef<Array>, axis: i32) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_argsort_axis(res, a.as_ref().as_ptr(), axis, stream.as_ref().as_ptr())
    })
}

/// Returns the indices that sort the flattened array. Returns an error if the arguments are
/// invalid.
///
/// # Params
///
/// - `a`: The array to sort.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let result = argsort(&a);
/// ```
pub fn argsort(a: impl AsRef<Array>) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_argsort(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Returns a partitioned copy of the array such that the smaller `kth` elements are first.
/// Returns an error if the arguments are invalid.
///
/// The ordering of the elements in partitions is undefined.
///
/// # Params
///
/// - `array`: input array
/// - `kth`: Element at the `kth` index will be in its sorted position in the output. All elements
///   before the kth index will be less or equal to the `kth` element and all elements after will be
///   greater or equal to the `kth` element in the output.
/// - `axis`: axis to partition over
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let kth = 1;
/// let axis = 0;
/// let result = partition_axis(&a, kth, axis);
/// ```
pub fn partition_axis(a: impl AsRef<Array>, kth: i32, axis: i32) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_partition_axis(
            res,
            a.as_ref().as_ptr(),
            kth,
            axis,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Returns a partitioned copy of the flattened array such that the smaller `kth` elements are
/// first. Returns an error if the arguments are invalid.
///
/// The ordering of the elements in partitions is undefined.
///
/// # Params
///
/// - `array`: input array
/// - `kth`: Element at the `kth` index will be in its sorted position in the output. All elements
///   before the kth index will be less or equal to the `kth` element and all elements after will be
///   greater or equal to the `kth` element in the output.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let kth = 1;
/// let result = partition(&a, kth);
/// ```
pub fn partition(a: impl AsRef<Array>, kth: i32) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_partition(res, a.as_ref().as_ptr(), kth, stream.as_ref().as_ptr())
    })
}

/// Returns the indices that partition the array. Returns an error if the arguments are invalid.
///
/// The ordering of the elements within a partition in given by the indices is undefined.
///
/// # Params
///
/// - `a`: The array to sort.
/// - `kth`: element index at the `kth` position in the output will give the sorted position.  All
///   indices before the`kth` position will be of elements less than or equal to the element at the
///   `kth` index and all indices after will be elemenents greater than or equal to the element at
///   the `kth` position.
/// - `axis`: axis to partition over
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let kth = 1;
/// let axis = 0;
/// let result = argpartition_axis(&a, kth, axis);
/// ```
pub fn argpartition_axis(a: impl AsRef<Array>, kth: i32, axis: i32) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_argpartition_axis(
            res,
            a.as_ref().as_ptr(),
            kth,
            axis,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Returns the indices that partition the flattened array. Returns an error if the arguments are
/// invalid.
///
/// The ordering of the elements within a partition in given by the indices is undefined.
///
/// # Params
///
/// - `a`: The array to sort.
/// - `kth`: element index at the `kth` position in the output will give the sorted position.  All
///   indices before the`kth` position will be of elements less than or equal to the element at the
///   `kth` index and all indices after will be elemenents greater than or equal to the element at
///   the `kth` position.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let kth = 1;
/// let result = argpartition(&a, kth);
/// ```
pub fn argpartition(a: impl AsRef<Array>, kth: i32) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_argpartition(res, a.as_ref().as_ptr(), kth, stream.as_ref().as_ptr())
    })
}

#[cfg(test)]
mod tests {
    use super::{SearchSide, search_sorted};
    use crate::{Array, array};

    #[test]
    fn search_sorted_sides_differ_on_ties() {
        let sorted = array!([1.0f32, 2.0, 2.0, 2.0, 5.0]);
        let values = array!([[0.0f32, 2.0], [3.0, 9.0]]);

        let left = search_sorted(&sorted, &values, None).unwrap();
        assert_eq!(left.shape(), &[2, 2]);
        assert_eq!(left.as_slice::<u32>(), &[0, 1, 4, 5]);

        let right = search_sorted(&sorted, &values, SearchSide::Right).unwrap();
        assert_eq!(right.as_slice::<u32>(), &[0, 4, 4, 5]);
    }

    #[test]
    fn search_sorted_rejects_multidimensional_sequence() {
        let sorted = array!([[1, 2], [3, 4]]);
        assert!(search_sorted(&sorted, array!([2]), None).is_err());
    }

    #[test]
    fn test_sort_with_invalid_axis() {
        let a = Array::from_slice(&[1, 2, 3, 4, 5], &[5]);
        let axis = 1;
        let result = super::sort_axis(&a, axis);
        assert!(result.is_err());
    }

    #[test]
    fn test_partition_with_invalid_axis() {
        let a = Array::from_slice(&[1, 2, 3, 4, 5], &[5]);
        let kth = 2;
        let axis = 1;
        let result = super::partition_axis(&a, kth, axis);
        assert!(result.is_err());
    }

    #[test]
    fn test_partition_with_invalid_kth() {
        let a = Array::from_slice(&[1, 2, 3, 4, 5], &[5]);
        let kth = 5;
        let axis = 0;
        let result = super::partition_axis(&a, kth, axis);
        assert!(result.is_err());
    }

    #[test]
    fn test_partition_all_with_invalid_kth() {
        let a = Array::from_slice(&[1, 2, 3, 4, 5], &[5]);
        let kth = 5;
        let result = super::partition(&a, kth);
        assert!(result.is_err());
    }
}
