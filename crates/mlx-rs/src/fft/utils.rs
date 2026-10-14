use smallvec::SmallVec;

use crate::constants::DEFAULT_STACK_VEC_LEN;
use crate::error::{Exception, Result};
use crate::utils::resolve_index_unchecked;
use crate::{Array, Axes};

#[inline]
pub(super) fn resolve_size_and_axis_unchecked(
    a: &Array,
    n: Option<i32>,
    axis: Option<i32>,
) -> (i32, i32) {
    let axis = axis.unwrap_or(-1);
    let n = n.unwrap_or_else(|| {
        let axis_index = resolve_index_unchecked(axis, a.ndim());
        a.shape()[axis_index]
    });
    (n, axis)
}

type SizesAndAxes = (
    SmallVec<[i32; DEFAULT_STACK_VEC_LEN]>,
    SmallVec<[i32; DEFAULT_STACK_VEC_LEN]>,
);

/// Default axes of the two-dimensional transforms.
const AXES_2D: [i32; 2] = [-2, -1];

/// Resolve the transform lengths and axes of an n-dimensional transform.
///
/// [`Axes::All`] selects every input axis. When `lengths` is given there must be exactly one
/// length per selected axis; otherwise the lengths are the input sizes along the selected axes.
pub(super) fn resolve_lengths_and_axes(
    shape: &[i32],
    lengths: Option<&[i32]>,
    axes: &Axes,
) -> Result<SizesAndAxes> {
    match axes {
        Axes::All => {
            let ndim = shape.len() as i32;
            let all: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = (-ndim..0).collect();
            resolve_explicit(shape, lengths, &all)
        }
        Axes::Axis(axis) => resolve_explicit(shape, lengths, std::slice::from_ref(axis)),
        Axes::Axes(axes) => resolve_explicit(shape, lengths, axes),
    }
}

/// Like [`resolve_lengths_and_axes`], but [`Axes::All`] selects the last two axes.
pub(super) fn resolve_lengths_and_axes_2d(
    shape: &[i32],
    lengths: Option<&[i32]>,
    axes: &Axes,
) -> Result<SizesAndAxes> {
    match axes {
        Axes::All => resolve_explicit(shape, lengths, &AXES_2D),
        _ => resolve_lengths_and_axes(shape, lengths, axes),
    }
}

fn resolve_explicit(shape: &[i32], lengths: Option<&[i32]>, axes: &[i32]) -> Result<SizesAndAxes> {
    let ndim = shape.len() as i32;
    let mut sizes = SmallVec::with_capacity(axes.len());
    for &axis in axes {
        if axis < -ndim || axis >= ndim {
            return Err(Exception::custom(format!(
                "FFT axis {axis} is out of bounds for an array with {ndim} dimensions"
            )));
        }
        sizes.push(shape[resolve_index_unchecked(axis, shape.len())]);
    }

    if let Some(lengths) = lengths {
        if lengths.len() != axes.len() {
            return Err(Exception::custom(format!(
                "FFT lengths and axes must have the same length, got {} lengths for {} axes",
                lengths.len(),
                axes.len()
            )));
        }
        sizes = SmallVec::from_slice(lengths);
    }

    Ok((sizes, SmallVec::from_slice(axes)))
}

/// Real transforms treat the last selected axis as the real axis, so at least one is required.
pub(super) fn require_real_axis(axes: &[i32]) -> Result<()> {
    if axes.is_empty() {
        return Err(Exception::custom(
            "real FFTs need at least one axis to treat as the real axis",
        ));
    }
    Ok(())
}

/// The output length along the real axis of an inverse real transform whose input has `n`
/// entries there: `(n - 1) * 2`.
pub(super) fn resolve_inverse_real_length(n: i32) -> Result<i32> {
    n.checked_sub(1)
        .and_then(|n| n.checked_mul(2))
        .filter(|&len| len > 0)
        .ok_or_else(|| {
            Exception::custom(format!(
                "inverse real FFT needs at least 2 entries along the real axis, got {n}"
            ))
        })
}
