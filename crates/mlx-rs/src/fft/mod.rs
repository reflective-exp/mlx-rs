//! Fast Fourier Transform (FFT) and its inverse (IFFT) for one, two, and `N` dimensions.
//!
//! Every function returns a `Result` instead of panicking on invalid input. The one-dimensional
//! transforms take the transform length, axis and normalization as arguments; the two- and
//! `N`-dimensional transforms take [`FftnOptions`].
//!
//! Like all `mlx-rs` functions, they run on the default stream; wrap calls in
//! [`crate::with_stream`] or [`crate::with_device`] to choose another. GPU support is not yet
//! implemented.
//!
//! # Examples
//!
//! ## One dimension
//!
//! ```rust
//! use mlx_rs::{Dtype, Array, complex64, fft::*};
//!
//! let src = [1.0f32, 2.0, 3.0, 4.0];
//! let mut array = Array::from_slice(&src[..], &[4]);
//!
//! let mut fft_result = fft(&array, 4, 0, None).unwrap();
//! assert_eq!(fft_result.dtype(), Dtype::Complex64);
//!
//! let expected = &[
//!     complex64::new(10.0, 0.0),
//!     complex64::new(-2.0, 2.0),
//!     complex64::new(-2.0, 0.0),
//!     complex64::new(-2.0, -2.0),
//! ];
//! assert_eq!(fft_result.as_slice::<complex64>(), &expected[..]);
//!
//! let mut ifft_result = ifft(&fft_result, 4, 0, None).unwrap();
//! assert_eq!(ifft_result.dtype(), Dtype::Complex64);
//!
//! let expected = &[
//!    complex64::new(1.0, 0.0),
//!    complex64::new(2.0, 0.0),
//!    complex64::new(3.0, 0.0),
//!    complex64::new(4.0, 0.0),
//! ];
//! assert_eq!(ifft_result.as_slice::<complex64>(), &expected[..]);
//!
//! let mut rfft_result = rfft(&array, 4, 0, None).unwrap();
//! assert_eq!(rfft_result.dtype(), Dtype::Complex64);
//!
//! let expected = &[
//!    complex64::new(10.0, 0.0),
//!    complex64::new(-2.0, 2.0),
//!    complex64::new(-2.0, 0.0),
//! ];
//! assert_eq!(rfft_result.as_slice::<complex64>(), &expected[..]);
//!
//! let mut irfft_result = irfft(&rfft_result, 4, 0, None).unwrap();
//! assert_eq!(irfft_result.dtype(), Dtype::Float32);
//! assert_eq!(irfft_result.as_slice::<f32>(), &src[..]);
//!
//! // The original array is not modified
//! let data: &[f32] = array.as_slice();
//! assert_eq!(data, &src[..]);
//! ```
//!
//! ## Two dimensions
//!
//! ```rust
//! use mlx_rs::{Dtype, Array, complex64, fft::*};
//!
//! let src = [1.0f32, 1.0, 1.0, 1.0];
//! let mut array = Array::from_slice(&src[..], &[2, 2]);
//!
//! let mut fft2_result = fft2(&array, FftnOptions::default()).unwrap();
//! assert_eq!(fft2_result.dtype(), Dtype::Complex64);
//! let expected = &[
//!     complex64::new(4.0, 0.0),
//!     complex64::new(0.0, 0.0),
//!     complex64::new(0.0, 0.0),
//!     complex64::new(0.0, 0.0),
//! ];
//! // Multi-axis FFTs can return a strided view; make it contiguous before borrowing a slice.
//! let fft2_result = fft2_result.contiguous().unwrap();
//! assert_eq!(fft2_result.as_slice::<complex64>(), &expected[..]);
//!
//! let mut ifft2_result = ifft2(&fft2_result, FftnOptions::default()).unwrap();
//! assert_eq!(ifft2_result.dtype(), Dtype::Complex64);
//!
//! let expected = &[
//!    complex64::new(1.0, 0.0),
//!    complex64::new(1.0, 0.0),
//!    complex64::new(1.0, 0.0),
//!    complex64::new(1.0, 0.0),
//! ];
//! assert_eq!(ifft2_result.as_slice::<complex64>(), &expected[..]);
//!
//! let mut rfft2_result = rfft2(&array, FftnOptions::default()).unwrap();
//! assert_eq!(rfft2_result.dtype(), Dtype::Complex64);
//!
//! let expected = &[
//!     complex64::new(4.0, 0.0),
//!     complex64::new(0.0, 0.0),
//!     complex64::new(0.0, 0.0),
//!     complex64::new(0.0, 0.0),
//! ];
//! let rfft2_result = rfft2_result.contiguous().unwrap();
//! assert_eq!(rfft2_result.as_slice::<complex64>(), &expected[..]);
//!
//! let mut irfft2_result = irfft2(&rfft2_result, FftnOptions::default()).unwrap();
//! assert_eq!(irfft2_result.dtype(), Dtype::Float32);
//! assert_eq!(irfft2_result.as_slice::<f32>(), &src[..]);
//!
//! // The original array is not modified
//! let data: &[f32] = array.as_slice();
//! assert_eq!(data, &[1.0, 1.0, 1.0, 1.0]);
//! ```
//!
//! ## `N` dimensions
//!
//! ```rust
//! use mlx_rs::{Dtype, Array, complex64, fft::*};
//!
//! let mut array = Array::ones::<f32>(&[2, 2, 2]).unwrap();
//! let mut fftn_result = fftn(&array, FftnOptions::default()).unwrap();
//! assert_eq!(fftn_result.dtype(), Dtype::Complex64);
//!
//! let mut expected = [complex64::new(0.0, 0.0); 8];
//! expected[0] = complex64::new(8.0, 0.0);
//! // Multi-axis FFTs can return a strided view; make it contiguous before borrowing a slice.
//! let fftn_result = fftn_result.contiguous().unwrap();
//! assert_eq!(fftn_result.as_slice::<complex64>(), &expected[..]);
//!
//! let mut ifftn_result = ifftn(&fftn_result, FftnOptions::default()).unwrap();
//! assert_eq!(ifftn_result.dtype(), Dtype::Complex64);
//!
//! let expected = [complex64::new(1.0, 0.0); 8];
//! let ifftn_result = ifftn_result.contiguous().unwrap();
//! assert_eq!(ifftn_result.as_slice::<complex64>(), &expected[..]);
//!
//! let mut rfftn_result = rfftn(&array, FftnOptions::default()).unwrap();
//! assert_eq!(rfftn_result.dtype(), Dtype::Complex64);
//!
//! let mut expected = [complex64::new(0.0, 0.0); 8];
//! expected[0] = complex64::new(8.0, 0.0);
//! let rfftn_result = rfftn_result.contiguous().unwrap();
//! assert_eq!(rfftn_result.as_slice::<complex64>(), &expected[..]);
//!
//! let mut irfftn_result = irfftn(&rfftn_result, FftnOptions::default()).unwrap();
//! assert_eq!(irfftn_result.dtype(), Dtype::Float32);
//!
//! let expected = [1.0; 8];
//! let irfftn_result = irfftn_result.contiguous().unwrap();
//! assert_eq!(irfftn_result.as_slice::<f32>(), &expected[..]);
//!
//! // The original array is not modified
//! let data: &[f32] = array.as_slice();
//! assert_eq!(data, &[1.0; 8]);
//! ```

mod fftn;
mod freq;
mod options;
mod rfftn;
mod shift;
mod utils;

pub use self::{fftn::*, freq::*, options::FftnOptions, rfftn::*, shift::*};

/// Normalization mode applied to an FFT.
///
/// Mirrors numpy's `norm` argument. The default is [`FftNorm::Backward`], which matches the
/// behavior of the pre-0.32 FFT functions (no scaling on the forward transform, `1/n` on the
/// inverse).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FftNorm {
    /// No scaling on the forward transform, `1/n` on the inverse. This is the default.
    #[default]
    Backward,
    /// `1/sqrt(n)` scaling on both the forward and inverse transforms.
    Ortho,
    /// `1/n` scaling on the forward transform, none on the inverse.
    Forward,
}

impl From<FftNorm> for mlx_sys::mlx_fft_norm {
    fn from(norm: FftNorm) -> Self {
        match norm {
            FftNorm::Backward => mlx_sys::mlx_fft_norm__MLX_FFT_NORM_BACKWARD,
            FftNorm::Ortho => mlx_sys::mlx_fft_norm__MLX_FFT_NORM_ORTHO,
            FftNorm::Forward => mlx_sys::mlx_fft_norm__MLX_FFT_NORM_FORWARD,
        }
    }
}

/// Resolve an optional [`FftNorm`] to the C enum, defaulting to [`FftNorm::Backward`].
fn resolve_norm(norm: Option<FftNorm>) -> mlx_sys::mlx_fft_norm {
    norm.unwrap_or_default().into()
}

/* -------------------------------------------------------------------------- */
/*                              Helper functions                              */
/* -------------------------------------------------------------------------- */
