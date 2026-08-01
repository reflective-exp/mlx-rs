//! Metal backend controls: availability, shader library location, and GPU trace capture.

use std::ffi::{CStr, CString};

use crate::error::{Exception, Result, ensure_error_handler, last_mlx_error_or};
use crate::utils::SUCCESS;
use crate::utils::guard::Guarded;

/// Whether MLX was built with the Metal backend and a Metal device is usable.
pub fn is_available() -> Result<bool> {
    bool::try_from_op(|res| unsafe { mlx_sys::mlx_metal_is_available(res) })
}

/// The path MLX loads its compiled Metal shader library (`mlx.metallib`) from.
///
/// Empty unless it has been overridden with [`set_metallib_path`], in which case MLX falls back to
/// looking next to the loaded binary.
pub fn metallib_path() -> Result<String> {
    ensure_error_handler();
    unsafe {
        let mut ptr: *const std::os::raw::c_char = std::ptr::null();
        let status = mlx_sys::mlx_metal_get_metallib_path(&mut ptr as *mut _);
        if status != SUCCESS {
            return Err(last_mlx_error_or("Failed to get metallib path"));
        }
        Ok(CStr::from_ptr(ptr).to_string_lossy().into_owned())
    }
}

/// Override the path MLX loads its compiled Metal shader library from.
///
/// This is process-global and must be set before the Metal device is first used.
pub fn set_metallib_path(path: &str) -> Result<()> {
    let path = CString::new(path).map_err(|_| Exception::from("Invalid metallib path"))?;
    <()>::try_from_op(|_res| unsafe { mlx_sys::mlx_metal_set_metallib_path(path.as_ptr()) })
}

/// Start capturing a Metal GPU trace to `path`, which should end in `.gputrace`.
///
/// The process must have been launched with `MTL_CAPTURE_ENABLED=1`; otherwise Metal refuses to
/// start the capture and this returns an error. Pair with [`stop_capture`].
pub fn start_capture(path: &str) -> Result<()> {
    let path = CString::new(path).map_err(|_| Exception::from("Invalid capture path"))?;
    <()>::try_from_op(|_res| unsafe { mlx_sys::mlx_metal_start_capture(path.as_ptr()) })
}

/// Stop the capture started by [`start_capture`] and flush the trace to its destination.
pub fn stop_capture() -> Result<()> {
    <()>::try_from_op(|_res| unsafe { mlx_sys::mlx_metal_stop_capture() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_available() {
        assert!(is_available().unwrap());
    }

    #[test]
    fn test_metallib_path_roundtrip() {
        set_metallib_path("/tmp/some-metallib-path").unwrap();
        assert_eq!(metallib_path().unwrap(), "/tmp/some-metallib-path");
    }

    #[test]
    fn test_capture() {
        let path = std::env::temp_dir().join("mlx-rs-capture.gputrace");
        let _ = std::fs::remove_dir_all(&path);
        let path = path.to_str().unwrap();

        // Capture only starts when the process was launched with MTL_CAPTURE_ENABLED=1,
        // so accept either outcome and check that each is well formed.
        match start_capture(path) {
            Ok(()) => {
                stop_capture().unwrap();
                assert!(std::path::Path::new(path).exists());
            }
            Err(e) => assert!(e.what().contains("start_capture")),
        }
    }
}
