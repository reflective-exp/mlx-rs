//! Device memory management.
//!
//! MLX keeps a cache of previously allocated buffers to speed up future
//! allocations. These functions let you inspect current usage, bound how much
//! memory MLX may use, and reclaim cached buffers — the main levers for
//! reducing peak memory on Apple Silicon.
//!
//! ```rust
//! use mlx_rs::{Array, memory};
//!
//! // Cap the buffer cache and reclaim what is already cached.
//! let _previous = memory::set_cache_limit(0).unwrap();
//! memory::clear_cache().unwrap();
//! ```

use crate::error::Result;
use crate::utils::guard::Guarded;

/// Get the actively used memory in bytes.
///
/// Note, this will not always match memory use reported by the system because
/// it does not include cached memory buffers.
pub fn get_active_memory() -> Result<usize> {
    usize::try_from_op(|res| unsafe { mlx_sys::mlx_get_active_memory(res) })
}

/// Get the cache size in bytes.
///
/// The cache includes memory not currently used that has not been returned to
/// the system allocator.
pub fn get_cache_memory() -> Result<usize> {
    usize::try_from_op(|res| unsafe { mlx_sys::mlx_get_cache_memory(res) })
}

/// Get the peak amount of active memory in bytes.
///
/// The maximum memory used is recorded from the beginning of the program
/// execution or since the last call to [`reset_peak_memory`].
pub fn get_peak_memory() -> Result<usize> {
    usize::try_from_op(|res| unsafe { mlx_sys::mlx_get_peak_memory(res) })
}

/// Reset the peak memory to zero.
pub fn reset_peak_memory() -> Result<()> {
    <() as Guarded>::try_from_op(|_| unsafe { mlx_sys::mlx_reset_peak_memory() })
}

/// Get the current memory limit in bytes.
pub fn get_memory_limit() -> Result<usize> {
    usize::try_from_op(|res| unsafe { mlx_sys::mlx_get_memory_limit(res) })
}

/// Set the memory limit in bytes, returning the previous limit.
///
/// Calls to allocation will wait on scheduled tasks if the limit is exceeded.
/// If there are no more scheduled tasks an error will be raised if the limit is
/// exceeded. The memory limit defaults to 1.5 times the maximum recommended
/// working set size reported by the device.
pub fn set_memory_limit(limit: usize) -> Result<usize> {
    usize::try_from_op(|res| unsafe { mlx_sys::mlx_set_memory_limit(res, limit) })
}

/// Set the free cache limit in bytes, returning the previous limit.
///
/// If using more than the given limit, free memory will be reclaimed from the
/// cache on the next allocation. Setting a limit of `0` disables the cache.
pub fn set_cache_limit(limit: usize) -> Result<usize> {
    usize::try_from_op(|res| unsafe { mlx_sys::mlx_set_cache_limit(res, limit) })
}

/// Set the wired memory limit in bytes, returning the previous limit.
///
/// Memory up to the limit is kept resident (wired) so it is not paged out,
/// which can reduce latency for weights reused across calls. The limit must be
/// less than the maximum recommended working set size reported by the device.
pub fn set_wired_limit(limit: usize) -> Result<usize> {
    usize::try_from_op(|res| unsafe { mlx_sys::mlx_set_wired_limit(res, limit) })
}

/// Clear the memory cache, returning cached buffers to the system allocator.
pub fn clear_cache() -> Result<()> {
    <() as Guarded>::try_from_op(|_| unsafe { mlx_sys::mlx_clear_cache() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Array;
    use crate::transforms::eval;

    #[test]
    fn test_getters_return_ok() {
        assert!(get_active_memory().is_ok());
        assert!(get_cache_memory().is_ok());
        assert!(get_peak_memory().is_ok());
        assert!(get_memory_limit().is_ok());
    }

    #[test]
    fn test_memory_limit_round_trip() {
        let original = get_memory_limit().unwrap();

        let previous = set_memory_limit(original).unwrap();
        assert_eq!(previous, original);
        assert_eq!(get_memory_limit().unwrap(), original);

        // Restore.
        set_memory_limit(original).unwrap();
    }

    #[test]
    fn test_set_cache_limit_disables_cache() {
        let original = set_cache_limit(0).unwrap();
        clear_cache().unwrap();
        assert_eq!(get_cache_memory().unwrap(), 0);

        // Restore.
        set_cache_limit(original).unwrap();
    }

    #[test]
    fn test_peak_memory_tracks_allocation() {
        reset_peak_memory().unwrap();
        assert_eq!(get_peak_memory().unwrap(), 0);

        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
        let b = a.multiply(&a).unwrap();
        eval([&b]).unwrap();

        assert!(get_peak_memory().unwrap() > 0);
    }
}
