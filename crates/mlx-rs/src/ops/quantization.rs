use std::ffi::CStr;

use crate::error::Result;
use crate::utils::guard::Guarded;
use crate::utils::{VectorArray, optional_dtype, optional_int};
use crate::{Array, Dtype, Stream};

const DEFAULT_GROUP_SIZE: i32 = 64;
const DEFAULT_BITS: i32 = 4;

/// The quantization scheme used by [`quantize`], [`dequantize`], [`quantized_matmul`],
/// [`gather_qmm`], [`qqmm`] and [`gather_qqmm`].
///
/// See the [MLX
/// documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
/// for details on each mode. The `mx`/`nv` modes require a fixed group size (32 for `mxfp4`/`mxfp8`,
/// 16 for `nvfp4`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantizationMode {
    /// Affine quantization with per-group scales and biases. The default.
    #[default]
    Affine,

    /// Microscaling 4-bit floating point (E2M1). Requires a group size of 32.
    Mxfp4,

    /// Microscaling 8-bit floating point (E4M3). Requires a group size of 32.
    Mxfp8,

    /// NVIDIA 4-bit floating point. Requires a group size of 16.
    Nvfp4,
}

impl From<QuantizationMode> for &'static CStr {
    /// The null-terminated C string MLX expects for this mode.
    fn from(mode: QuantizationMode) -> Self {
        match mode {
            QuantizationMode::Affine => c"affine",
            QuantizationMode::Mxfp4 => c"mxfp4",
            QuantizationMode::Mxfp8 => c"mxfp8",
            QuantizationMode::Nvfp4 => c"nvfp4",
        }
    }
}

/// Resolve an optional [`QuantizationMode`] to its C string, defaulting to
/// [`QuantizationMode::Affine`].
fn resolve_mode(mode: Option<QuantizationMode>) -> &'static CStr {
    mode.unwrap_or_default().into()
}

/// Quantize the matrix `w` using `bits` bits per element.
///
/// Note, every `group_size` elements in a row of `w` are quantized together. Hence, number of
/// columns of `w` should be divisible by `group_size`. In particular, the rows of `w` are divided
/// into groups of size `group_size` which are quantized together.
///
/// > `quantized` currently only supports 2D inputs with dimensions which are multiples of 32
///
/// For details, please see [this
/// documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
///
/// # Params
///
/// - `w`: The input matrix
/// - `group_size`: The size of the group in `w` that shares a scale and bias. (default: `64`)
/// - `bits`: The number of bits occupied by each element of w in the returned quantized matrix.
///   (default: 4)
pub fn quantize<'a>(
    w: impl AsRef<Array>,
    group_size: impl Into<Option<i32>>,
    bits: impl Into<Option<i32>>,
    mode: impl Into<Option<QuantizationMode>>,
    global_scale: impl Into<Option<&'a Array>>,
) -> Result<(Array, Array, Array)> {
    let stream = Stream::thread_local_or_default();
    let group_size = optional_int(group_size.into(), DEFAULT_GROUP_SIZE);
    let bits = optional_int(bits.into(), DEFAULT_BITS);
    let mode = resolve_mode(mode.into());
    let global_scale = global_scale.into();

    let result = VectorArray::try_from_op(|res| unsafe {
        mlx_sys::mlx_quantize(
            res,
            w.as_ref().as_ptr(),
            group_size,
            bits,
            mode.as_ptr(),
            // An empty array signals "no global scale" (used by non-affine modes only).
            global_scale
                .map(|a| a.as_ptr())
                .unwrap_or_else(crate::utils::empty_array_ptr),
            stream.as_ref().as_ptr(),
        )
    })?;

    let arrays: Vec<Array> = result.try_into_values()?;
    if arrays.len() != 3 {
        return Err(crate::error::Exception::custom(format!(
            "Expected 3 arrays from quantize, got {}",
            arrays.len()
        )));
    }
    let mut iter = arrays.into_iter();
    Ok((
        iter.next().unwrap(),
        iter.next().unwrap(),
        iter.next().unwrap(),
    ))
}

/// Perform the matrix multiplication with the quantized matrix `w`. The quantization uses one
/// floating point scale and bias per `group_size` of elements. Each element in `w` takes `bits`
/// bits and is packed in an unsigned 32 bit integer.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul<'a>(
    x: impl AsRef<Array>,
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    biases: impl Into<Option<&'a Array>>,
    transpose: impl Into<Option<bool>>,
    group_size: impl Into<Option<i32>>,
    bits: impl Into<Option<i32>>,
    mode: impl Into<Option<QuantizationMode>>,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let transpose = transpose.into().unwrap_or(false);
    let group_size = optional_int(group_size.into(), DEFAULT_GROUP_SIZE);
    let bits = optional_int(bits.into(), DEFAULT_BITS);
    let mode = resolve_mode(mode.into());
    let biases = biases.into();

    <Array as Guarded>::try_from_op(|res| unsafe {
        mlx_sys::mlx_quantized_matmul(
            res,
            x.as_ref().as_ptr(),
            w.as_ref().as_ptr(),
            scales.as_ref().as_ptr(),
            biases
                .map(|a| a.as_ptr())
                .unwrap_or_else(crate::utils::empty_array_ptr),
            transpose,
            group_size,
            bits,
            mode.as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Dequantize the matrix `w` using the provided `scales` and `biases` and the `group_size` and
/// `bits` configuration.
///
/// For details, please see [this
/// documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.dequantize.html)
///
/// # Params
///
/// - `output_dtype`: The dtype of the dequantized output. When `None`, MLX picks the output dtype.
#[allow(clippy::too_many_arguments)]
pub fn dequantize<'a>(
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    biases: impl Into<Option<&'a Array>>,
    group_size: impl Into<Option<i32>>,
    bits: impl Into<Option<i32>>,
    mode: impl Into<Option<QuantizationMode>>,
    global_scale: impl Into<Option<&'a Array>>,
    output_dtype: impl Into<Option<Dtype>>,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let group_size = optional_int(group_size.into(), DEFAULT_GROUP_SIZE);
    let bits = optional_int(bits.into(), DEFAULT_BITS);
    let mode = resolve_mode(mode.into());
    let biases = biases.into();
    let global_scale = global_scale.into();
    let output_dtype = optional_dtype(output_dtype.into());

    <Array as Guarded>::try_from_op(|res| unsafe {
        mlx_sys::mlx_dequantize(
            res,
            w.as_ref().as_ptr(),
            scales.as_ref().as_ptr(),
            biases
                .map(|a| a.as_ptr())
                .unwrap_or_else(crate::utils::empty_array_ptr),
            group_size,
            bits,
            mode.as_ptr(),
            // An empty array signals "no global scale" (used by non-affine modes only).
            global_scale
                .map(|a| a.as_ptr())
                .unwrap_or_else(crate::utils::empty_array_ptr),
            output_dtype,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Perform quantized matrix multiplication with gathered indices.
///
/// This combines the functionality of `gather_mm` and `quantized_matmul`, allowing
/// matrix multiplication with quantized weights and index gathering along batch dimensions.
///
/// # Params
///
/// - `x`: Input array
/// - `w`: Quantized weight matrix
/// - `scales`: Quantization scales
/// - `biases`: Optional quantization biases (required for affine mode)
/// - `lhs_indices`: Optional indices to gather from `x`'s batch dimensions
/// - `rhs_indices`: Optional indices to gather from `w`'s batch dimensions
/// - `transpose`: If true, transpose the weight matrix (default: true)
/// - `group_size`: The quantization group size (default: 64)
/// - `bits`: The number of bits per element (default: 4)
/// - `sorted_indices`: If true, indicates the indices are sorted (default: false)
#[allow(clippy::too_many_arguments)]
pub fn gather_qmm<'b, 'lhs, 'rhs>(
    x: impl AsRef<Array>,
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    biases: impl Into<Option<&'b Array>>,
    lhs_indices: impl Into<Option<&'lhs Array>>,
    rhs_indices: impl Into<Option<&'rhs Array>>,
    transpose: impl Into<Option<bool>>,
    group_size: impl Into<Option<i32>>,
    bits: impl Into<Option<i32>>,
    mode: impl Into<Option<QuantizationMode>>,
    sorted_indices: impl Into<Option<bool>>,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let transpose = transpose.into().unwrap_or(true);
    let group_size = optional_int(group_size.into(), DEFAULT_GROUP_SIZE);
    let bits = optional_int(bits.into(), DEFAULT_BITS);
    let mode = resolve_mode(mode.into());
    let sorted = sorted_indices.into().unwrap_or(false);

    unsafe {
        let biases_ptr = biases
            .into()
            .map(|a| a.as_ptr())
            .unwrap_or_else(crate::utils::empty_array_ptr);
        let lhs_ptr = lhs_indices
            .into()
            .map(|i| i.as_ptr())
            .unwrap_or_else(crate::utils::empty_array_ptr);
        let rhs_ptr = rhs_indices
            .into()
            .map(|i| i.as_ptr())
            .unwrap_or_else(crate::utils::empty_array_ptr);

        <Array as Guarded>::try_from_op(|res| {
            mlx_sys::mlx_gather_qmm(
                res,
                x.as_ref().as_ptr(),
                w.as_ref().as_ptr(),
                scales.as_ref().as_ptr(),
                biases_ptr,
                lhs_ptr,
                rhs_ptr,
                transpose,
                group_size,
                bits,
                mode.as_ptr(),
                sorted,
                stream.as_ref().as_ptr(),
            )
        })
    }
}

/// Quantization settings for [`qqmm`] and [`gather_qqmm`].
///
/// `group_size` and `bits` default to what `mode` requires (16 and 4 for `nvfp4`, 32 and 8 for
/// `mxfp8`). Global scales apply to `nvfp4` only and must be given together.
#[derive(Debug, Clone, Copy, Default)]
pub struct QqmmOptions<'a> {
    /// The quantization group size.
    pub group_size: Option<i32>,

    /// The number of bits per element.
    pub bits: Option<i32>,

    /// The quantization mode. Defaults to [`QuantizationMode::Nvfp4`].
    pub mode: Option<QuantizationMode>,

    /// Global scale used to quantize `x`.
    pub global_scale_x: Option<&'a Array>,

    /// Global scale used to quantize `w`.
    pub global_scale_w: Option<&'a Array>,
}

impl QqmmOptions<'_> {
    // MLX picks mode-specific defaults when `has_value` is false, so the fallback value is unused.
    fn group_size(&self) -> mlx_sys::mlx_optional_int {
        optional_int(self.group_size, 0)
    }

    fn bits(&self) -> mlx_sys::mlx_optional_int {
        optional_int(self.bits, 0)
    }

    fn mode(&self) -> &'static CStr {
        self.mode.unwrap_or(QuantizationMode::Nvfp4).into()
    }
}

/// Quantized matrix multiplication with quantization of both inputs.
///
/// Computes `x @ w.T`, quantizing `x` on the fly, and `w` too when `w_scales` is `None`.
/// Supports the `nvfp4` and `mxfp8` modes.
///
/// GPU only (Metal or CUDA).
///
/// # Params
///
/// - `x`: Input matrix to be dynamically quantized
/// - `w`: Weight matrix, quantized or not
/// - `w_scales`: Scales for `w` if it is already quantized
/// - `options`: See [`QqmmOptions`]
pub fn qqmm<'a>(
    x: impl AsRef<Array>,
    w: impl AsRef<Array>,
    w_scales: impl Into<Option<&'a Array>>,
    options: QqmmOptions<'_>,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let mode = options.mode();

    <Array as Guarded>::try_from_op(|res| unsafe {
        mlx_sys::mlx_qqmm(
            res,
            x.as_ref().as_ptr(),
            w.as_ref().as_ptr(),
            w_scales
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or_else(crate::utils::empty_array_ptr),
            options.group_size(),
            options.bits(),
            mode.as_ptr(),
            options
                .global_scale_x
                .map(|a| a.as_ptr())
                .unwrap_or_else(crate::utils::empty_array_ptr),
            options
                .global_scale_w
                .map(|a| a.as_ptr())
                .unwrap_or_else(crate::utils::empty_array_ptr),
            stream.as_ref().as_ptr(),
        )
    })
}

/// [`qqmm`] on matrices gathered from the batch dimensions, like
/// [`gather_mm`](crate::ops::gather_mm).
///
/// GPU only (Metal or CUDA).
///
/// # Params
///
/// - `x`: Input array to be dynamically quantized
/// - `w`: Weight array, quantized or not
/// - `w_scales`: Scales for `w` if it is already quantized
/// - `lhs_indices`: Indices into the batch dimensions of `x`. Defaults to all of them.
/// - `rhs_indices`: Indices into the batch dimensions of `w`. Defaults to all of them.
/// - `sorted_indices`: Whether `rhs_indices` is sorted (default: false)
/// - `options`: See [`QqmmOptions`]
pub fn gather_qqmm<'a>(
    x: impl AsRef<Array>,
    w: impl AsRef<Array>,
    w_scales: impl Into<Option<&'a Array>>,
    lhs_indices: impl Into<Option<&'a Array>>,
    rhs_indices: impl Into<Option<&'a Array>>,
    sorted_indices: impl Into<Option<bool>>,
    options: QqmmOptions<'_>,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let mode = options.mode();
    let sorted = sorted_indices.into().unwrap_or(false);

    <Array as Guarded>::try_from_op(|res| unsafe {
        mlx_sys::mlx_gather_qqmm(
            res,
            x.as_ref().as_ptr(),
            w.as_ref().as_ptr(),
            w_scales
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or_else(crate::utils::empty_array_ptr),
            lhs_indices
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or_else(crate::utils::empty_array_ptr),
            rhs_indices
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or_else(crate::utils::empty_array_ptr),
            options.group_size(),
            options.bits(),
            mode.as_ptr(),
            options
                .global_scale_x
                .map(|a| a.as_ptr())
                .unwrap_or_else(crate::utils::empty_array_ptr),
            options
                .global_scale_w
                .map(|a| a.as_ptr())
                .unwrap_or_else(crate::utils::empty_array_ptr),
            sorted,
            stream.as_ref().as_ptr(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::QuantizationMode;
    use crate::{
        Array,
        ops::{dequantize, expand_dims, quantize, quantized_matmul},
        random,
    };
    use std::ffi::CStr;

    #[test]
    fn test_quantization_mode_c_str() {
        let cases = [
            (QuantizationMode::Affine, c"affine"),
            (QuantizationMode::Mxfp4, c"mxfp4"),
            (QuantizationMode::Mxfp8, c"mxfp8"),
            (QuantizationMode::Nvfp4, c"nvfp4"),
        ];
        for (mode, expected) in cases {
            let c: &CStr = mode.into();
            assert_eq!(c, expected);
        }
        assert_eq!(QuantizationMode::default(), QuantizationMode::Affine);
    }

    #[test]
    fn test_quantize_dequantize() {
        let x1 = Array::ones::<f32>(&[128, 1]).unwrap();
        let x2 = expand_dims(Array::arange::<_, f32>(0, 512, None).unwrap(), 0).unwrap();
        let x = x1 * x2;

        for i in [2, 4, 8].iter() {
            let el_per_int = 32 / i;
            let (x_q, scales, biases) = quantize(&x, 128, *i, None, None).unwrap();
            assert_eq!(x_q.shape(), [128, 512 / el_per_int]);
            assert_eq!(scales.shape(), [128, 4]);
            assert_eq!(biases.shape(), [128, 4]);

            let x_hat = dequantize(&x_q, &scales, &biases, 128, *i, None, None, None).unwrap();
            let max_diff = ((&x - &x_hat).abs().unwrap().max(None).unwrap()).item::<f32>();
            assert!(max_diff <= 127.0 / (1 << i) as f32);
        }
    }

    #[test]
    fn test_dequantize_output_dtype() {
        use crate::Dtype;

        let x1 = Array::ones::<f32>(&[128, 1]).unwrap();
        let x2 = expand_dims(Array::arange::<_, f32>(0, 512, None).unwrap(), 0).unwrap();
        let x = x1 * x2;

        let (x_q, scales, biases) = quantize(&x, 128, 4, None, None).unwrap();

        // Without an output dtype, dequantize returns float32.
        let default_hat = dequantize(&x_q, &scales, &biases, 128, 4, None, None, None).unwrap();
        assert_eq!(default_hat.dtype(), Dtype::Float32);

        // Requesting float16 as the output dtype is honored.
        let f16_hat = dequantize(
            &x_q,
            &scales,
            &biases,
            128,
            4,
            None,
            None,
            Some(Dtype::Float16),
        )
        .unwrap();
        assert_eq!(f16_hat.dtype(), Dtype::Float16);
    }

    // Test adapted from Python test `test_quantized.py/test_qmm`
    #[test]
    fn test_quantized_matmul() {
        random::seed(0).unwrap();

        let group_size = 64;
        let bits = 4;
        let m = 32;
        let n = 128;
        let k = 128;

        let scale = 1.0 / (k as f32).sqrt();
        let x = random::normal::<f32>(&[m, k], None, None, None).unwrap() * scale;
        let w = random::normal::<f32>(&[k, n], None, None, None).unwrap() * scale;

        let (w_q, scales, biases) = quantize(&w, group_size, bits, None, None).unwrap();
        let w_hat = dequantize(&w_q, &scales, &biases, group_size, bits, None, None, None).unwrap();

        // Test with biases
        let y_q =
            quantized_matmul(&x, &w_q, &scales, &biases, false, group_size, bits, None).unwrap();
        let y_hat = x.matmul(&w_hat).unwrap();

        assert_eq!(y_q.shape(), y_hat.shape());
        let max_diff = ((&y_q - &y_hat).abs().unwrap().max(None).unwrap()).item::<f32>();
        assert!(max_diff < 1e-3, "max_diff: {}", max_diff);
    }

    // Test adapted from Python test `test_quantized.py/test_gather_qmm`
    #[test]
    fn test_gather_qmm() {
        use crate::ops::{gather_mm, gather_qmm, swap_axes};

        random::seed(0).unwrap();

        let group_size = 64;
        let bits = 4;

        // Helper to quantize with transpose option
        fn quantize_with_transpose(
            w: &Array,
            transpose: bool,
            group_size: i32,
            bits: i32,
        ) -> (Array, Array, Array, Array) {
            let (w_q, scales, biases) = quantize(w, group_size, bits, None, None).unwrap();
            let mut w_hat =
                dequantize(&w_q, &scales, &biases, group_size, bits, None, None, None).unwrap();
            if transpose {
                w_hat = swap_axes(&w_hat, -1, -2).unwrap();
            }
            (w_hat, w_q, scales, biases)
        }

        // Test case 1: batch_A=(1,), lhs_indices=(0,), batch_B=(3,), rhs_indices=(2, 1)
        let m = 32;
        let n = 64;
        let k = 64;

        let x = random::normal::<f32>(&[1, m, k], None, None, None).unwrap();
        let w = random::normal::<f32>(&[3, n, k], None, None, None).unwrap(); // transpose=true shape
        let (w_hat, w_q, scales, biases) = quantize_with_transpose(&w, true, group_size, bits);

        let lhs_indices = Array::from_slice(&[0u32], &[1]);
        let rhs_indices = Array::from_slice(&[2u32, 1], &[2]);

        // Compare gather_mm on dequantized weights vs gather_qmm
        let c1 = gather_mm(&x, &w_hat, &lhs_indices, &rhs_indices, None).unwrap();
        let c2 = gather_qmm(
            &x,
            &w_q,
            &scales,
            &biases,
            &lhs_indices,
            &rhs_indices,
            true,
            group_size,
            bits,
            None, // mode
            None, // sorted_indices
        )
        .unwrap();
        assert!(
            c1.all_close(&c2, 1e-4, 1e-4, None).unwrap().item::<bool>(),
            "gather_qmm test case 1 failed"
        );

        // Test case 2: batch_A=(5,), lhs_indices=(0, 2), batch_B=(3,), rhs_indices=(2, 1)
        let x = random::normal::<f32>(&[5, m, k], None, None, None).unwrap();
        let lhs_indices = Array::from_slice(&[0u32, 2], &[2]);

        let c1 = gather_mm(&x, &w_hat, &lhs_indices, &rhs_indices, None).unwrap();
        let c2 = gather_qmm(
            &x,
            &w_q,
            &scales,
            &biases,
            &lhs_indices,
            &rhs_indices,
            true,
            group_size,
            bits,
            None, // mode
            None, // sorted_indices
        )
        .unwrap();
        assert!(
            c1.all_close(&c2, 1e-4, 1e-4, None).unwrap().item::<bool>(),
            "gather_qmm test case 2 failed"
        );
    }

    // `quantize` cannot return the two-array fp formats yet, so these tests let MLX quantize
    // `w` on the fly rather than following the Python tests.
    #[test]
    fn test_qqmm_approximates_matmul() {
        use super::{QqmmOptions, qqmm};

        random::seed(0).unwrap();
        let x = random::normal::<f32>(&[32, 128], None, None, None).unwrap();
        let w = random::normal::<f32>(&[64, 128], None, None, None).unwrap();
        let expected = x.matmul(w.swap_axes(-1, -2).unwrap()).unwrap();

        for mode in [QuantizationMode::Nvfp4, QuantizationMode::Mxfp8] {
            let options = QqmmOptions {
                mode: Some(mode),
                ..Default::default()
            };
            let y = qqmm(&x, &w, None, options).unwrap();
            assert_eq!(y.shape(), expected.shape());

            let err = (&y - &expected).abs().unwrap().mean(None).unwrap();
            let scale = expected.abs().unwrap().mean(None).unwrap();
            let rel = err.item::<f32>() / scale.item::<f32>();
            assert!(rel < 0.25, "{mode:?}: relative error {rel}");
        }
    }

    #[test]
    fn test_gather_qqmm_matches_qqmm() {
        use super::{QqmmOptions, gather_qqmm, qqmm};
        use crate::ops::{indexing::IndexOp, stack};

        random::seed(0).unwrap();
        let (m, n, k) = (32, 64, 128);
        let x = random::normal::<f32>(&[5, m, k], None, None, None).unwrap();
        let w = random::normal::<f32>(&[3, n, k], None, None, None).unwrap();
        let lhs = [0u32, 2];
        let rhs = [2u32, 1];
        let lhs_indices = Array::from_slice(&lhs, &[2]);
        let rhs_indices = Array::from_slice(&rhs, &[2]);

        for mode in [QuantizationMode::Nvfp4, QuantizationMode::Mxfp8] {
            let options = QqmmOptions {
                mode: Some(mode),
                ..Default::default()
            };
            let y = gather_qqmm(&x, &w, None, &lhs_indices, &rhs_indices, None, options).unwrap();
            assert_eq!(y.shape(), [2, m, n]);

            let expected: Vec<Array> = lhs
                .iter()
                .zip(rhs)
                .map(|(&l, r)| qqmm(x.index(l as i32), w.index(r as i32), None, options).unwrap())
                .collect();
            let expected = stack(&expected, 0).unwrap();
            assert!(
                y.all_close(&expected, 1e-3, 1e-3, None)
                    .unwrap()
                    .item::<bool>(),
                "{mode:?}: gather_qqmm differs from qqmm"
            );
        }
    }

    #[test]
    fn test_qqmm_rejects_batched_inputs() {
        use super::{QqmmOptions, qqmm};

        let x = random::normal::<f32>(&[32, 128], None, None, None).unwrap();
        let w = random::normal::<f32>(&[2, 64, 128], None, None, None).unwrap();
        assert!(qqmm(&x, &w, None, QqmmOptions::default()).is_err());
    }
}
