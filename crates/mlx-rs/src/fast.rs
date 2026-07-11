//! Fast implementations of commonly used multi-op functions.

use std::ffi::{CStr, CString};

use crate::error::{Exception, Result};
use crate::utils::guard::Guarded;
use crate::utils::{IntoOption, SUCCESS, VectorArray};
use crate::{Array, Dtype, Stream};
use mlx_internal_macros::{default_device, generate_macro};

/// Optimized implementation of `NN.RoPE`.
#[allow(clippy::too_many_arguments)]
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn rope_device<'a>(
    #[named] array: impl AsRef<Array>,
    #[named] dimensions: i32,
    #[named] traditional: bool,
    #[optional] base: impl Into<Option<f32>>,
    #[named] scale: f32,
    #[named] offset: i32,
    #[optional] freqs: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let base = base.into();
    let base = mlx_sys::mlx_optional_float {
        value: base.unwrap_or(0.0),
        has_value: base.is_some(),
    };
    let freqs = freqs.into();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rope(
            res,
            array.as_ref().as_ptr(),
            dimensions,
            traditional,
            base,
            scale,
            offset,
            freqs
                .map(|a| a.as_ptr())
                .unwrap_or_else(crate::utils::empty_array_ptr),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Optimized implementation of `NN.RoPE` with dynamic (array) offset.
///
/// This variant allows specifying the offset as an array, enabling different
/// offsets for different positions in the input.
///
/// # Params
///
/// - `array`: Input array
/// - `dimensions`: The feature dimensions to apply rope to
/// - `traditional`: If true, uses the traditional rope implementation
/// - `base`: The base used to compute angular frequency for each dimension
/// - `scale`: The scale to apply to the positions
/// - `offset`: An array of position offsets
/// - `freqs`: Optional precomputed frequencies
/// - `stream`: Stream to evaluate on
#[allow(clippy::too_many_arguments)]
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn rope_dynamic_device<'a>(
    #[named] array: impl AsRef<Array>,
    #[named] dimensions: i32,
    #[named] traditional: bool,
    #[optional] base: impl Into<Option<f32>>,
    #[named] scale: f32,
    #[named] offset: impl AsRef<Array>,
    #[optional] freqs: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let base = base.into();
    let base = mlx_sys::mlx_optional_float {
        value: base.unwrap_or(0.0),
        has_value: base.is_some(),
    };
    let freqs = freqs.into();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rope_dynamic(
            res,
            array.as_ref().as_ptr(),
            dimensions,
            traditional,
            base,
            scale,
            offset.as_ref().as_ptr(),
            freqs
                .map(|a| a.as_ptr())
                .unwrap_or_else(crate::utils::empty_array_ptr),
            stream.as_ref().as_ptr(),
        )
    })
}

const DEFAULT_MASK_MODE: &CStr = c"";
const CAUSAL_MASK_MODE: &CStr = c"causal";

/// Mask modes for scaled dot product attention.
#[derive(Debug)]
pub enum ScaledDotProductAttentionMask<'a> {
    /// A single mask array
    Array(&'a Array),

    /// Causal masking (no explicit mask array needed)
    Causal,
}

impl<'a> From<&'a Array> for ScaledDotProductAttentionMask<'a> {
    fn from(mask: &'a Array) -> Self {
        ScaledDotProductAttentionMask::Array(mask)
    }
}

impl<'a> IntoOption<ScaledDotProductAttentionMask<'a>> for &'a Array {
    fn into_option(self) -> Option<ScaledDotProductAttentionMask<'a>> {
        Some(ScaledDotProductAttentionMask::Array(self))
    }
}

impl ScaledDotProductAttentionMask<'_> {
    fn as_mode_and_mask(&self) -> (&'static CStr, mlx_sys::mlx_array) {
        match self {
            ScaledDotProductAttentionMask::Array(mask) => (DEFAULT_MASK_MODE, mask.as_ptr()),
            ScaledDotProductAttentionMask::Causal => {
                (CAUSAL_MASK_MODE, unsafe { mlx_sys::mlx_array_new() })
            }
        }
    }
}

/// A fast implementation of multi-head attention: `O = softmax(Q @ K.T, dim=-1) @ V`
///
/// Supports [Multi-Head Attention](https://arxiv.org/abs/1706.03762), [Grouped Query Attention](https://arxiv.org/abs/2305.13245), and [Multi-Query Attention](https://arxiv.org/abs/1911.02150).
///
/// This function will dispatch to an optimized Metal kernel when the query sequence length is 1. It handles other cases with regular MLX operations.
///
/// > Note: The softmax operation is performed in float32 precision regardless of input precision (float16 or float32).
///
/// > Note: For Grouped Query Attention and Multi-Query Attention, the input arrays for `key` and `value` should not be pre-tiled to match the `query` array.
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn scaled_dot_product_attention_device<'a>(
    queries: impl AsRef<Array>,
    keys: impl AsRef<Array>,
    values: impl AsRef<Array>,
    scale: f32,
    #[optional] mask: impl IntoOption<ScaledDotProductAttentionMask<'a>>,
    #[optional] sinks: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let (mask_mode, mask_arr) = mask.into_option().map_or_else(
        || (DEFAULT_MASK_MODE, unsafe { mlx_sys::mlx_array_new() }),
        |m| m.as_mode_and_mask(),
    );

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_scaled_dot_product_attention(
            res,
            queries.as_ref().as_ptr(),
            keys.as_ref().as_ptr(),
            values.as_ref().as_ptr(),
            scale,
            mask_mode.as_ptr(),
            mask_arr,
            sinks
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or_else(crate::utils::empty_array_ptr),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Root Mean Square normalization (RMS norm).
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// # Params
///
/// - x: input array
/// - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional with the same size as the last axis of `x`.
/// - eps: A small additive constant for numerical stability
/// - stream: stream or device to evaluate on
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn rms_norm_device(
    x: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    eps: f32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rms_norm(
            res,
            x.as_ref().as_ptr(),
            weight.as_ref().as_ptr(),
            eps,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Layer normalization.
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// # Params
///
/// - x: input array
/// - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional
///   with the same size as the last axis of `x`.  If not given no scaling will occur.
/// - bias: An additive offset to be added to the result. The `bias` should be one-dimensional
///   with the same size as the last axis of `x`.  It not given no offset will occur.
/// - eps: A small additive constant for numerical stability
/// - stream: stream or device to evaluate on
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn layer_norm_device<'a>(
    #[named] x: impl AsRef<Array>,
    #[optional] weight: impl Into<Option<&'a Array>>,
    #[optional] bias: impl Into<Option<&'a Array>>,
    #[named] eps: f32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_layer_norm(
            res,
            x.as_ref().as_ptr(),
            weight
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or_else(crate::utils::empty_array_ptr),
            bias.into()
                .map(|a| a.as_ptr())
                .unwrap_or_else(crate::utils::empty_array_ptr),
            eps,
            stream.as_ref().as_ptr(),
        )
    })
}

fn cstring(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| Exception::custom("string contains an interior nul byte"))
}

/// Raw exception message left by the most recent failing C call, or `fallback`
/// if none was recorded.
fn last_mlx_error(fallback: &str) -> Exception {
    match crate::error::get_and_clear_last_mlx_error() {
        Some(e) => Exception::custom(e.what),
        None => Exception::custom(fallback),
    }
}

/// An owned `mlx_vector_string` that frees its contents on drop.
///
/// MLX copies each appended string into its own storage, so the source `&str`
/// values only need to outlive [`try_from_iter`](VectorString::try_from_iter).
struct VectorString {
    c_vec: mlx_sys::mlx_vector_string,
}

impl VectorString {
    fn try_from_iter<'a>(iter: impl IntoIterator<Item = &'a str>) -> Result<Self> {
        let this = VectorString {
            c_vec: unsafe { mlx_sys::mlx_vector_string_new() },
        };
        for s in iter {
            let s = cstring(s)?;
            let status = unsafe { mlx_sys::mlx_vector_string_append_value(this.c_vec, s.as_ptr()) };
            if status != SUCCESS {
                return Err(last_mlx_error("failed to build vector of strings"));
            }
        }
        Ok(this)
    }

    fn as_ptr(&self) -> mlx_sys::mlx_vector_string {
        self.c_vec
    }
}

impl Drop for VectorString {
    fn drop(&mut self) {
        let status = unsafe { mlx_sys::mlx_vector_string_free(self.c_vec) };
        debug_assert_eq!(status, SUCCESS);
    }
}

/// A compiled custom Metal kernel.
///
/// This is a safe wrapper around the `mlx_fast_metal_kernel_*` C API. Build one
/// with [`MetalKernel::builder`], then run it with [`MetalKernel::apply`] once
/// per call, supplying a [`MetalKernelConfig`] that describes the launch grid,
/// output arrays, and template arguments.
///
/// The Metal source is compiled lazily on the first [`apply`](MetalKernel::apply),
/// so source-compilation errors surface there rather than at build time.
///
/// # Example
///
/// ```no_run
/// use mlx_rs::{Dtype, fast::{MetalKernel, MetalKernelConfig}};
///
/// let input = mlx_rs::random::normal::<f32>(&[4, 16], None, None, None).unwrap();
/// let kernel = MetalKernel::builder(
///     "myexp",
///     "uint elem = thread_position_in_grid.x;\
///      T tmp = inp[elem];\
///      out[elem] = metal::exp(tmp);",
/// )
/// .input_names(["inp"])
/// .output_names(["out"])
/// .build()
/// .unwrap();
///
/// let config = MetalKernelConfig::new()
///     .template_arg_dtype("T", Dtype::Float32).unwrap()
///     .grid(input.size() as i32, 1, 1).unwrap()
///     .thread_group(256, 1, 1).unwrap()
///     .output_arg(input.shape(), Dtype::Float32).unwrap();
///
/// let outputs = kernel.apply([&input], &config).unwrap();
/// ```
#[derive(Debug)]
pub struct MetalKernel {
    c_kernel: mlx_sys::mlx_fast_metal_kernel,
}

impl MetalKernel {
    /// Starts building a kernel named `name` from the given Metal `source`.
    ///
    /// The `source` is the body of the Metal kernel function; MLX generates the
    /// surrounding function signature from the input and output names. See
    /// [`MetalKernelBuilder`] for the remaining options.
    pub fn builder<'a>(name: &'a str, source: &'a str) -> MetalKernelBuilder<'a> {
        MetalKernelBuilder::new(name, source)
    }

    /// Runs the kernel on the default stream. See [`apply_device`](MetalKernel::apply_device).
    pub fn apply(
        &self,
        inputs: impl IntoIterator<Item = impl AsRef<Array>>,
        config: &MetalKernelConfig,
    ) -> Result<Vec<Array>> {
        self.apply_device(inputs, config, Stream::task_local_or_default())
    }

    /// Runs the kernel on `stream`, returning one [`Array`] per output declared
    /// on `config` via [`MetalKernelConfig::output_arg`].
    pub fn apply_device(
        &self,
        inputs: impl IntoIterator<Item = impl AsRef<Array>>,
        config: &MetalKernelConfig,
        stream: impl AsRef<Stream>,
    ) -> Result<Vec<Array>> {
        let inputs = VectorArray::try_from_iter(inputs.into_iter())?;
        Vec::<Array>::try_from_op(|res| unsafe {
            mlx_sys::mlx_fast_metal_kernel_apply(
                res,
                self.c_kernel,
                inputs.as_ptr(),
                config.c_config,
                stream.as_ref().as_ptr(),
            )
        })
    }
}

impl Drop for MetalKernel {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_fast_metal_kernel_free(self.c_kernel);
        }
    }
}

// SAFETY: A `MetalKernel` owns a compiled-kernel handle that is created once in
// `build` and thereafter only read — passed as an argument to
// `mlx_fast_metal_kernel_apply`, which does not mutate it. It is therefore safe
// to share and move a kernel across threads, letting callers cache one in a
// `static` and reuse it for every call.
unsafe impl Send for MetalKernel {}
unsafe impl Sync for MetalKernel {}

/// Builder for a [`MetalKernel`].
///
/// Created by [`MetalKernel::builder`]. `ensure_row_contiguous` defaults to
/// `true` and `atomic_outputs` to `false`, matching MLX's own defaults.
#[derive(Debug)]
pub struct MetalKernelBuilder<'a> {
    name: &'a str,
    source: &'a str,
    header: &'a str,
    input_names: Vec<&'a str>,
    output_names: Vec<&'a str>,
    ensure_row_contiguous: bool,
    atomic_outputs: bool,
}

impl<'a> MetalKernelBuilder<'a> {
    fn new(name: &'a str, source: &'a str) -> Self {
        Self {
            name,
            source,
            header: "",
            input_names: Vec::new(),
            output_names: Vec::new(),
            ensure_row_contiguous: true,
            atomic_outputs: false,
        }
    }

    /// Sets the names of the input arrays, in the order they are passed to
    /// [`MetalKernel::apply`]. Each name becomes a buffer parameter in the
    /// generated kernel.
    pub fn input_names(mut self, names: impl IntoIterator<Item = &'a str>) -> Self {
        self.input_names = names.into_iter().collect();
        self
    }

    /// Sets the names of the output arrays, in the order they are declared on
    /// the [`MetalKernelConfig`] via [`output_arg`](MetalKernelConfig::output_arg).
    pub fn output_names(mut self, names: impl IntoIterator<Item = &'a str>) -> Self {
        self.output_names = names.into_iter().collect();
        self
    }

    /// Sets source placed outside the kernel function body, e.g. helper
    /// functions or `#include`s. Empty by default.
    pub fn header(mut self, header: &'a str) -> Self {
        self.header = header;
        self
    }

    /// When `true` (the default), inputs are made row-contiguous before the
    /// kernel runs.
    pub fn ensure_row_contiguous(mut self, ensure_row_contiguous: bool) -> Self {
        self.ensure_row_contiguous = ensure_row_contiguous;
        self
    }

    /// When `true`, outputs are passed as atomic buffers. Defaults to `false`.
    pub fn atomic_outputs(mut self, atomic_outputs: bool) -> Self {
        self.atomic_outputs = atomic_outputs;
        self
    }

    /// Builds the [`MetalKernel`].
    pub fn build(self) -> Result<MetalKernel> {
        crate::error::INIT_ERR_HANDLER
            .with(|init| init.call_once(crate::error::setup_mlx_error_handler));

        let name = cstring(self.name)?;
        let source = cstring(self.source)?;
        let header = cstring(self.header)?;
        let input_names = VectorString::try_from_iter(self.input_names)?;
        let output_names = VectorString::try_from_iter(self.output_names)?;

        let c_kernel = unsafe {
            mlx_sys::mlx_fast_metal_kernel_new(
                name.as_ptr(),
                input_names.as_ptr(),
                output_names.as_ptr(),
                source.as_ptr(),
                header.as_ptr(),
                self.ensure_row_contiguous,
                self.atomic_outputs,
            )
        };

        if c_kernel.ctx.is_null() {
            return Err(last_mlx_error("failed to build metal kernel"));
        }
        Ok(MetalKernel { c_kernel })
    }
}

/// Per-call configuration for a [`MetalKernel`].
///
/// Describes the launch grid, thread group, output arrays, and template
/// arguments for a single [`MetalKernel::apply`]. Output arrays are produced in
/// the order they are added with [`output_arg`](MetalKernelConfig::output_arg).
///
/// The setter methods consume and return `self`, so they can be chained with
/// `?`.
#[derive(Debug)]
pub struct MetalKernelConfig {
    c_config: mlx_sys::mlx_fast_metal_kernel_config,
}

impl MetalKernelConfig {
    /// Creates an empty configuration.
    pub fn new() -> Self {
        MetalKernelConfig {
            c_config: unsafe { mlx_sys::mlx_fast_metal_kernel_config_new() },
        }
    }

    /// Declares an output array with the given `shape` and `dtype`. The kernel
    /// produces one [`Array`] per call to this method, in call order.
    pub fn output_arg(self, shape: &[i32], dtype: Dtype) -> Result<Self> {
        <()>::try_from_op(|_| unsafe {
            mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
                self.c_config,
                shape.as_ptr(),
                shape.len(),
                dtype.into(),
            )
        })?;
        Ok(self)
    }

    /// Sets the launch grid (total number of threads in each dimension).
    pub fn grid(self, x: i32, y: i32, z: i32) -> Result<Self> {
        <()>::try_from_op(|_| unsafe {
            mlx_sys::mlx_fast_metal_kernel_config_set_grid(self.c_config, x, y, z)
        })?;
        Ok(self)
    }

    /// Sets the thread group (threads per group in each dimension).
    pub fn thread_group(self, x: i32, y: i32, z: i32) -> Result<Self> {
        <()>::try_from_op(|_| unsafe {
            mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(self.c_config, x, y, z)
        })?;
        Ok(self)
    }

    /// Sets the value used to initialize the output arrays before the kernel runs.
    pub fn init_value(self, value: f32) -> Result<Self> {
        <()>::try_from_op(|_| unsafe {
            mlx_sys::mlx_fast_metal_kernel_config_set_init_value(self.c_config, value)
        })?;
        Ok(self)
    }

    /// When `true`, MLX prints the generated kernel source when it is compiled.
    pub fn verbose(self, verbose: bool) -> Result<Self> {
        <()>::try_from_op(|_| unsafe {
            mlx_sys::mlx_fast_metal_kernel_config_set_verbose(self.c_config, verbose)
        })?;
        Ok(self)
    }

    /// Binds a `dtype` template argument named `name` in the kernel source.
    pub fn template_arg_dtype(self, name: &str, dtype: Dtype) -> Result<Self> {
        let name = cstring(name)?;
        <()>::try_from_op(|_| unsafe {
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                self.c_config,
                name.as_ptr(),
                dtype.into(),
            )
        })?;
        Ok(self)
    }

    /// Binds an integer template argument named `name` in the kernel source.
    pub fn template_arg_int(self, name: &str, value: i32) -> Result<Self> {
        let name = cstring(name)?;
        <()>::try_from_op(|_| unsafe {
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                self.c_config,
                name.as_ptr(),
                value,
            )
        })?;
        Ok(self)
    }

    /// Binds a boolean template argument named `name` in the kernel source.
    pub fn template_arg_bool(self, name: &str, value: bool) -> Result<Self> {
        let name = cstring(name)?;
        <()>::try_from_op(|_| unsafe {
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_bool(
                self.c_config,
                name.as_ptr(),
                value,
            )
        })?;
        Ok(self)
    }
}

impl Default for MetalKernelConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for MetalKernelConfig {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_fast_metal_kernel_config_free(self.c_config);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ops::indexing::{ArrayIndexOp, IndexOp},
        random::normal,
    };
    use float_eq::assert_float_eq;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_rope() {
        crate::random::seed(71).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let result = rope(a, 8, false, 10000., 1.0, 0, None).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            0.456_253_77,
            abs <= 0.009_125_075
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            116.800_964,
            abs <= 2.336_019_3
        );
    }

    // Test adapted from Python test_fast.py/test_rope - the Python test accepts both
    // int offset and array offset, which in C/Rust are separate functions
    #[test]
    fn test_rope_dynamic() {
        crate::random::seed(71).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        // Test with array offset - should produce similar results to int offset of 3
        let offset = crate::Array::from_int(3);
        let result = rope_dynamic(&a, 8, false, 10000., 1.0, &offset, None).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);

        // Compare with regular rope using int offset=3
        let result_int_offset = rope(&a, 8, false, 10000., 1.0, 3, None).unwrap();
        assert_eq!(result_int_offset.shape(), [2, 8, 16]);

        // The results should be close
        let diff = &result - &result_int_offset;
        let max_diff = diff.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(max_diff < 1e-5, "Max difference was {}", max_diff);
    }

    #[test]
    fn test_rms_norm() {
        crate::random::seed(103).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let weight = Array::ones::<f32>(&[16]).unwrap();
        let result = rms_norm(a, weight, 1e-5).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            0.872_938_75,
            abs <= 0.017_458_774
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            223.472_32,
            abs <= 4.469_446
        );
    }

    #[test]
    pub fn test_layer_norm_affine() {
        crate::random::seed(635).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let weight = Array::ones::<f32>(&[16]).unwrap();
        let bias = Array::zeros::<f32>(&[16]).unwrap();
        let result = layer_norm(a, &weight, &bias, 1e-5).unwrap();
        let result = result.index((ArrayIndexOp::Ellipsis, 0));
        assert_eq!(result.shape(), [2, 8]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            0.290_990_38,
            abs <= 0.005_819_807_8
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            4.655_846,
            abs <= 0.093_116_924
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_fast_sdpa() {
        // This test just makes sure that `scaled_dot_product_attention` is callable
        // in the various cases, based on the Python test `test_fast_sdpa`.

        let Dk = 64;
        let scale = 1.0 / (Dk as f32).sqrt();
        for seq_len in [63, 129, 400] {
            for dtype in [crate::Dtype::Float32, crate::Dtype::Float16] {
                let B = 2;
                let H = 24;
                let q = normal::<f32>(&[B, H, seq_len, Dk], None, None, None)
                    .unwrap()
                    .as_dtype(dtype, None)
                    .unwrap();
                let k = normal::<f32>(&[B, H, seq_len, Dk], None, None, None)
                    .unwrap()
                    .as_dtype(dtype, None)
                    .unwrap();
                let v = normal::<f32>(&[B, H, seq_len, Dk], None, None, None)
                    .unwrap()
                    .as_dtype(dtype, None)
                    .unwrap();

                let result = scaled_dot_product_attention(q, k, v, scale, None, None).unwrap();
                assert_eq!(result.shape(), [B, H, seq_len, Dk]);
                assert_eq!(result.dtype(), dtype);
            }
        }
    }

    // Test adapted from Python test `test_fast_sdpa.py/test_sdpa_attention_sinks`
    #[test]
    fn test_fast_sdpa_with_sinks() {
        let b = 2;
        let n_q = 8;
        let t_q = 128;
        let t_kv = 128;
        let d = 64;

        let q = normal::<f32>(&[b, n_q, t_q, d], None, None, None).unwrap();
        let k = normal::<f32>(&[b, n_q, t_kv, d], None, None, None).unwrap();
        let v = normal::<f32>(&[b, n_q, t_kv, d], None, None, None).unwrap();
        let scale = (d as f32).powf(-0.5);

        // Test with sinks parameter
        let sinks = normal::<f32>(&[n_q], None, None, None).unwrap() * 10.0;

        let result = scaled_dot_product_attention(&q, &k, &v, scale, None, &sinks).unwrap();
        assert_eq!(result.shape(), &[b, n_q, t_q, d]);
    }

    // Adapted from the C worked example `example-metal-kernel.c`: an
    // element-wise `exp` kernel with a `T` dtype template argument.
    const EXP_SOURCE: &str = "uint elem = thread_position_in_grid.x;\
                              T tmp = inp[elem];\
                              out[elem] = metal::exp(tmp);";

    #[test]
    fn test_metal_kernel_exp() {
        crate::random::seed(42).unwrap();
        let input = normal::<f32>(&[4, 16], None, None, None).unwrap();

        let kernel = MetalKernel::builder("myexp", EXP_SOURCE)
            .input_names(["inp"])
            .output_names(["out"])
            .build()
            .unwrap();

        let config = MetalKernelConfig::new()
            .template_arg_dtype("T", Dtype::Float32)
            .unwrap()
            .grid(input.size() as i32, 1, 1)
            .unwrap()
            .thread_group(256, 1, 1)
            .unwrap()
            .output_arg(input.shape(), Dtype::Float32)
            .unwrap();

        let outputs = kernel.apply([&input], &config).unwrap();
        assert_eq!(outputs.len(), 1);

        let out = &outputs[0];
        assert_eq!(out.shape(), [4, 16]);
        assert_eq!(out.dtype(), Dtype::Float32);

        let expected = input.exp().unwrap();
        let max_diff = (out - &expected)
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(max_diff < 1e-5, "max diff was {}", max_diff);
    }

    #[test]
    fn test_metal_kernel_init_value() {
        // A kernel that writes nothing leaves every output element at the
        // configured init value.
        let input = Array::zeros::<f32>(&[8]).unwrap();

        let kernel = MetalKernel::builder("noop", "")
            .input_names(["inp"])
            .output_names(["out"])
            .build()
            .unwrap();

        let config = MetalKernelConfig::new()
            .init_value(3.0)
            .unwrap()
            .grid(1, 1, 1)
            .unwrap()
            .thread_group(1, 1, 1)
            .unwrap()
            .output_arg(input.shape(), Dtype::Float32)
            .unwrap();

        let outputs = kernel.apply([&input], &config).unwrap();
        let out = &outputs[0];
        assert_eq!(out.shape(), [8]);
        assert_eq!(out.sum(None).unwrap().item::<f32>(), 24.0);
    }

    #[test]
    fn test_metal_kernel_compile_error_surfaces() {
        // Invalid Metal source compiles lazily, so the error appears when the
        // output is evaluated rather than at `apply`.
        let input = Array::zeros::<f32>(&[4]).unwrap();
        let kernel = MetalKernel::builder("bad", "this is not valid metal;")
            .input_names(["inp"])
            .output_names(["out"])
            .build()
            .unwrap();

        let config = MetalKernelConfig::new()
            .grid(4, 1, 1)
            .unwrap()
            .thread_group(1, 1, 1)
            .unwrap()
            .output_arg(input.shape(), Dtype::Float32)
            .unwrap();

        let outputs = kernel.apply([&input], &config).unwrap();
        assert!(outputs[0].eval().is_err());
    }
}
