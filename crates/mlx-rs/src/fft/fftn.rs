use crate::Stream;
use crate::array::Array;
use crate::error::Result;
use crate::utils::guard::Guarded;

use super::utils::{
    resolve_lengths_and_axes, resolve_lengths_and_axes_2d, resolve_size_and_axis_unchecked,
};
use super::{FftNorm, FftnOptions, resolve_norm};

/// One dimensional discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `n`: Size of the transformed axis. The corresponding axis in the input is truncated or padded
///   with zeros to match `n`. The default value is `a.shape[axis]`.
/// - `axis`: Axis along which to perform the FFT. The default is -1.
pub fn fft(
    a: impl AsRef<Array>,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    norm: impl Into<Option<FftNorm>>,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let a = a.as_ref();
    let (n, axis) = resolve_size_and_axis_unchecked(a, n.into(), axis.into());
    let norm = resolve_norm(norm.into());
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fft_fft(res, a.as_ptr(), n, axis, norm, stream.as_ref().as_ptr())
    })
}

/// Two dimensional discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `options`: Transform lengths, axes and normalization. The corresponding axes in the input are
///   truncated or padded with zeros to match `lengths`, which default to the sizes of `a` along
///   `axes`. [`Axes::All`](crate::Axes::All) (the default) selects the last two axes, `[-2, -1]`.
pub fn fft2(a: impl AsRef<Array>, options: FftnOptions) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let a = a.as_ref();
    let (s, axes) =
        resolve_lengths_and_axes_2d(a.shape(), options.lengths.as_deref(), &options.axes)?;

    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();
    let norm = resolve_norm(Some(options.norm));

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fft_fft2(
            res,
            a.as_ptr(),
            s_ptr,
            num_s,
            axes_ptr,
            num_axes,
            norm,
            stream.as_ref().as_ptr(),
        )
    })
}

/// n-dimensional discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `options`: Transform lengths, axes and normalization. The default transforms all axes at
///   their input sizes.
pub fn fftn(a: impl AsRef<Array>, options: FftnOptions) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let a = a.as_ref();
    let (s, axes) = resolve_lengths_and_axes(a.shape(), options.lengths.as_deref(), &options.axes)?;
    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();
    let norm = resolve_norm(Some(options.norm));

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fft_fftn(
            res,
            a.as_ptr(),
            s_ptr,
            num_s,
            axes_ptr,
            num_axes,
            norm,
            stream.as_ref().as_ptr(),
        )
    })
}

/// One dimensional inverse discrete Fourier Transform.
///
/// # Params
///
/// - `a`: Input array.
/// - `n`: Size of the transformed axis. The corresponding axis in the input is truncated or padded
///   with zeros to match `n`. The default value is `a.shape[axis]` if not specified.
/// - `axis`: Axis along which to perform the FFT. The default is `-1` if not specified.
pub fn ifft(
    a: impl AsRef<Array>,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    norm: impl Into<Option<FftNorm>>,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let a = a.as_ref();
    let (n, axis) = resolve_size_and_axis_unchecked(a, n.into(), axis.into());
    let norm = resolve_norm(norm.into());

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fft_ifft(res, a.as_ptr(), n, axis, norm, stream.as_ref().as_ptr())
    })
}

/// Two dimensional inverse discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `options`: Transform lengths, axes and normalization. The corresponding axes in the input are
///   truncated or padded with zeros to match `lengths`, which default to the sizes of `a` along
///   `axes`. [`Axes::All`](crate::Axes::All) (the default) selects the last two axes, `[-2, -1]`.
pub fn ifft2(a: impl AsRef<Array>, options: FftnOptions) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let a = a.as_ref();
    let (s, axes) =
        resolve_lengths_and_axes_2d(a.shape(), options.lengths.as_deref(), &options.axes)?;

    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();
    let norm = resolve_norm(Some(options.norm));

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fft_ifft2(
            res,
            a.as_ptr(),
            s_ptr,
            num_s,
            axes_ptr,
            num_axes,
            norm,
            stream.as_ref().as_ptr(),
        )
    })
}

/// n-dimensional inverse discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `options`: Transform lengths, axes and normalization. The default transforms all axes at
///   their input sizes.
pub fn ifftn(a: impl AsRef<Array>, options: FftnOptions) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let a = a.as_ref();
    let (s, axes) = resolve_lengths_and_axes(a.shape(), options.lengths.as_deref(), &options.axes)?;
    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();
    let norm = resolve_norm(Some(options.norm));

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fft_ifftn(
            res,
            a.as_ptr(),
            s_ptr,
            num_s,
            axes_ptr,
            num_axes,
            norm,
            stream.as_ref().as_ptr(),
        )
    })
}

#[cfg(test)]
mod tests {
    use crate::{Array, Dtype, complex64, fft::*};

    #[test]
    fn test_fft() {
        const FFT_DATA: &[f32] = &[1.0, 2.0, 3.0, 4.0];
        const FFT_SHAPE: &[i32] = &[4];
        const FFT_EXPECTED: &[complex64; 4] = &[
            complex64::new(10.0, 0.0),
            complex64::new(-2.0, 2.0),
            complex64::new(-2.0, 0.0),
            complex64::new(-2.0, -2.0),
        ];

        let array = Array::from_slice(FFT_DATA, FFT_SHAPE);
        let fft = fft(&array, None, None, None).unwrap();

        assert_eq!(fft.dtype(), Dtype::Complex64);
        assert_eq!(fft.as_slice::<complex64>(), FFT_EXPECTED);

        let ifft = ifft(&fft, None, None, None).unwrap();

        assert_eq!(ifft.dtype(), Dtype::Complex64);
        assert_eq!(
            ifft.as_slice::<complex64>(),
            FFT_DATA
                .iter()
                .map(|&x| complex64::new(x, 0.0))
                .collect::<Vec<_>>()
        );

        // The original array is not modified and valid
        let data: &[f32] = array.as_slice();
        assert_eq!(data, FFT_DATA);
    }

    #[test]
    fn test_fft_norm_ortho() {
        // Orthonormal normalization scales the forward transform by 1/sqrt(n).
        let array = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
        let fft = fft(&array, None, None, FftNorm::Ortho).unwrap();

        let expected = &[
            complex64::new(5.0, 0.0),
            complex64::new(-1.0, 1.0),
            complex64::new(-1.0, 0.0),
            complex64::new(-1.0, -1.0),
        ];
        assert_eq!(fft.as_slice::<complex64>(), expected);
    }

    #[test]
    fn test_fft2() {
        const FFT2_DATA: &[f32] = &[1.0, 1.0, 1.0, 1.0];
        const FFT2_SHAPE: &[i32] = &[2, 2];
        const FFT2_EXPECTED: &[complex64; 4] = &[
            complex64::new(4.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
        ];

        let array = Array::from_slice(FFT2_DATA, FFT2_SHAPE);
        let fft2 = fft2(&array, FftnOptions::default()).unwrap();

        assert_eq!(fft2.dtype(), Dtype::Complex64);
        // A multi-axis FFT returns a strided (not row-major) view.
        assert_eq!(
            fft2.contiguous().unwrap().as_slice::<complex64>(),
            FFT2_EXPECTED
        );

        let ifft2 = ifft2(&fft2, FftnOptions::default()).unwrap();

        assert_eq!(ifft2.dtype(), Dtype::Complex64);
        assert_eq!(
            ifft2.as_slice::<complex64>(),
            FFT2_DATA
                .iter()
                .map(|&x| complex64::new(x, 0.0))
                .collect::<Vec<_>>()
        );

        // test that previous array is not modified and valid
        let data: &[f32] = array.as_slice();
        assert_eq!(data, FFT2_DATA);
    }

    #[test]
    fn test_fftn() {
        const FFTN_DATA: &[f32] = &[1.0; 8];
        const FFTN_SHAPE: &[i32] = &[2, 2, 2];
        const FFTN_EXPECTED: &[complex64; 8] = &[
            complex64::new(8.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
        ];

        let array = Array::from_slice(FFTN_DATA, FFTN_SHAPE);
        let fftn = fftn(&array, FftnOptions::default()).unwrap();

        assert_eq!(fftn.dtype(), Dtype::Complex64);
        // A multi-axis FFT returns a strided (not row-major) view.
        assert_eq!(
            fftn.contiguous().unwrap().as_slice::<complex64>(),
            FFTN_EXPECTED
        );

        let ifftn = ifftn(
            &fftn,
            FftnOptions {
                lengths: Some(FFTN_SHAPE.to_vec()),
                axes: [0, 1, 2].into(),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(ifftn.dtype(), Dtype::Complex64);
        assert_eq!(
            ifftn.contiguous().unwrap().as_slice::<complex64>(),
            FFTN_DATA
                .iter()
                .map(|&x| complex64::new(x, 0.0))
                .collect::<Vec<_>>()
        );

        // test that previous array is not modified and valid
        let data: &[f32] = array.as_slice();
        assert_eq!(data, FFTN_DATA);
    }

    #[test]
    fn test_fftn_rejects_mismatched_lengths_and_axes() {
        let array = Array::ones::<f32>(&[2, 2, 2]).unwrap();

        // An all-axis transform needs one length per input dimension.
        let all_axes = FftnOptions {
            lengths: Some(vec![2, 2]),
            ..Default::default()
        };
        assert!(fftn(&array, all_axes).is_err());

        let partial = FftnOptions {
            lengths: Some(vec![2]),
            axes: [0, 1].into(),
            ..Default::default()
        };
        assert!(fftn(&array, partial).is_err());

        let out_of_bounds = FftnOptions {
            axes: 3.into(),
            ..Default::default()
        };
        assert!(fftn(&array, out_of_bounds).is_err());
    }
}
