use mlx_internal_macros::{default_device, generate_macro};

use crate::Stream;
use crate::array::Array;
use crate::error::Result;
use crate::utils::guard::Guarded;

/// Return the Discrete Fourier Transform sample frequencies.
///
/// The returned array contains the frequency bin centers in cycles per unit of the sample spacing,
/// ordered as `[0, 1, ..., n/2 - 1, -n/2, ..., -1] / (d * n)`.
///
/// # Params
///
/// - `n`: Size of the FFT window. Must be greater than 0.
/// - `d`: Sample spacing. The default is `1.0`. Must be non-zero.
///
/// # Example
///
/// ```rust
/// use mlx_rs::fft::*;
///
/// let freqs = fftfreq(4, None).unwrap();
/// assert_eq!(freqs.as_slice::<f32>(), &[0.0, 0.25, -0.5, -0.25]);
/// ```
#[generate_macro(customize(root = "$crate::fft"))]
#[default_device]
pub fn fftfreq_device(
    n: i32,
    #[optional] d: impl Into<Option<f64>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let d = d.into().unwrap_or(1.0);
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fft_fftfreq(res, n, d, stream.as_ref().as_ptr())
    })
}

/// Return the Discrete Fourier Transform sample frequencies for [`crate::fft::rfft`].
///
/// Unlike [`fftfreq`], the returned array only contains the `n / 2 + 1` non-negative frequencies,
/// matching the output length of a real-input FFT.
///
/// # Params
///
/// - `n`: Size of the FFT window. Must be greater than 0.
/// - `d`: Sample spacing. The default is `1.0`. Must be non-zero.
///
/// # Example
///
/// ```rust
/// use mlx_rs::fft::*;
///
/// let freqs = rfftfreq(4, None).unwrap();
/// assert_eq!(freqs.as_slice::<f32>(), &[0.0, 0.25, 0.5]);
/// ```
#[generate_macro(customize(root = "$crate::fft"))]
#[default_device]
pub fn rfftfreq_device(
    n: i32,
    #[optional] d: impl Into<Option<f64>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let d = d.into().unwrap_or(1.0);
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fft_rfftfreq(res, n, d, stream.as_ref().as_ptr())
    })
}

#[cfg(test)]
mod tests {
    use crate::fft::*;

    #[test]
    fn test_fftfreq() {
        let freqs = fftfreq(4, None).unwrap();
        assert_eq!(freqs.shape(), &[4]);
        assert_eq!(freqs.as_slice::<f32>(), &[0.0, 0.25, -0.5, -0.25]);

        let odd = fftfreq(5, None).unwrap();
        assert_eq!(odd.shape(), &[5]);

        let scaled = fftfreq(4, 0.5).unwrap();
        assert_eq!(scaled.as_slice::<f32>(), &[0.0, 0.5, -1.0, -0.5]);
    }

    #[test]
    fn test_fftfreq_invalid() {
        assert!(fftfreq(0, None).is_err());
        assert!(fftfreq(4, 0.0).is_err());
    }

    #[test]
    fn test_rfftfreq() {
        let freqs = rfftfreq(4, None).unwrap();
        assert_eq!(freqs.shape(), &[3]);
        assert_eq!(freqs.as_slice::<f32>(), &[0.0, 0.25, 0.5]);

        let scaled = rfftfreq(4, 0.5).unwrap();
        assert_eq!(scaled.as_slice::<f32>(), &[0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_rfftfreq_invalid() {
        assert!(rfftfreq(0, None).is_err());
        assert!(rfftfreq(4, 0.0).is_err());
    }
}
