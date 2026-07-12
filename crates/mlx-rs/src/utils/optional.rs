//! Conversions from Rust `Option` values to the C `mlx_optional_*` structs used by FFI calls.

use crate::Dtype;

/// Convert an optional [`Dtype`] to the C `mlx_optional_dtype`. When `None`, MLX picks the dtype
/// (the pre-0.32 behavior).
pub(crate) fn optional_dtype(value: Option<Dtype>) -> mlx_sys::mlx_optional_dtype {
    match value {
        Some(dtype) => mlx_sys::mlx_optional_dtype {
            value: dtype.into(),
            has_value: true,
        },
        None => mlx_sys::mlx_optional_dtype {
            value: mlx_sys::mlx_dtype__MLX_FLOAT32, // ignored when has_value is false
            has_value: false,
        },
    }
}

/// Convert an optional `bool` flag to the C `mlx_optional_bool`. When `None`, MLX decides (the
/// pre-0.32 behavior).
pub(crate) fn optional_bool(value: Option<bool>) -> mlx_sys::mlx_optional_bool {
    match value {
        Some(value) => mlx_sys::mlx_optional_bool {
            value,
            has_value: true,
        },
        None => mlx_sys::mlx_optional_bool {
            value: false, // ignored when has_value is false
            has_value: false,
        },
    }
}

/// Convert an optional `i32` to the C `mlx_optional_int`, substituting `default` when `None`.
pub(crate) fn optional_int(value: Option<i32>, default: i32) -> mlx_sys::mlx_optional_int {
    mlx_sys::mlx_optional_int {
        value: value.unwrap_or(default),
        has_value: value.is_some(),
    }
}
