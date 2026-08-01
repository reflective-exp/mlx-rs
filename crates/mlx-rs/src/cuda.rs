//! CUDA backend controls.

use crate::error::Result;
use crate::utils::guard::Guarded;

/// Whether MLX was built with the CUDA backend and a CUDA device is usable.
///
/// Always `false` on Apple silicon, where [`crate::metal`] is the GPU backend.
pub fn is_available() -> Result<bool> {
    bool::try_from_op(|res| unsafe { mlx_sys::mlx_cuda_is_available(res) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_available() {
        // mlx-rs targets Apple silicon, where the CUDA backend is never built in.
        assert!(!is_available().unwrap());
    }
}
