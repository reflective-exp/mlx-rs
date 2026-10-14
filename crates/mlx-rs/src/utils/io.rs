use crate::error::{Exception, IoError, ensure_error_handler, last_mlx_error_or};
use crate::utils::SUCCESS;
use crate::{Array, Stream};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::null_mut;

use super::Guarded;

pub(crate) struct SafeTensors {
    pub(crate) c_data: mlx_sys::mlx_map_string_to_array,
    pub(crate) c_metadata: mlx_sys::mlx_map_string_to_string,
}

impl Drop for SafeTensors {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_map_string_to_string_free(self.c_metadata);
            mlx_sys::mlx_map_string_to_array_free(self.c_data);
        }
    }
}

impl SafeTensors {
    pub(crate) fn load_device(path: &Path, stream: impl AsRef<Stream>) -> Result<Self, IoError> {
        if !path.is_file() {
            return Err(IoError::NotFile);
        }

        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or(IoError::UnsupportedFormat)?;

        if extension != "safetensors" {
            return Err(IoError::UnsupportedFormat);
        }

        let path_str = path.to_str().ok_or(IoError::InvalidUtf8)?;
        let filepath = CString::new(path_str)?;

        SafeTensors::try_from_op(|(res_0, res_1)| unsafe {
            mlx_sys::mlx_load_safetensors(res_0, res_1, filepath.as_ptr(), stream.as_ref().as_ptr())
        })
        .map_err(Into::into)
    }

    pub(crate) fn data(&self) -> Result<HashMap<String, Array>, Exception> {
        ensure_error_handler();
        let mut map = HashMap::new();
        unsafe {
            let iterator =
                ArrayMapIterator(mlx_sys::mlx_map_string_to_array_iterator_new(self.c_data));

            loop {
                let mut key_ptr: *const ::std::os::raw::c_char = null_mut();
                let mut value = mlx_sys::mlx_array_new();
                let status = mlx_sys::mlx_map_string_to_array_iterator_next(
                    &mut key_ptr as *mut *const _,
                    &mut value,
                    iterator.0,
                );

                if status != SUCCESS {
                    mlx_sys::mlx_array_free(value);
                }
                match status {
                    SUCCESS => {
                        let key = CStr::from_ptr(key_ptr).to_string_lossy().into_owned();
                        let array = Array::from_ptr(value);
                        map.insert(key, array);
                    }
                    ITERATOR_END => break,
                    _ => return Err(last_mlx_error_or("Failed to read safetensors arrays")),
                }
            }
        }

        Ok(map)
    }

    pub(crate) fn metadata(&self) -> Result<HashMap<String, String>, Exception> {
        ensure_error_handler();

        let mut map = HashMap::new();
        unsafe {
            let iterator = StringMapIterator(mlx_sys::mlx_map_string_to_string_iterator_new(
                self.c_metadata,
            ));

            let mut key: *const ::std::os::raw::c_char = null_mut();
            let mut value: *const ::std::os::raw::c_char = null_mut();
            loop {
                let status = mlx_sys::mlx_map_string_to_string_iterator_next(
                    &mut key as *mut *const _,
                    &mut value as *mut *const _,
                    iterator.0,
                );

                match status {
                    SUCCESS => {
                        let key = CStr::from_ptr(key).to_string_lossy().into_owned();
                        let value = CStr::from_ptr(value).to_string_lossy().into_owned();
                        map.insert(key, value);
                    }
                    ITERATOR_END => break,
                    _ => return Err(last_mlx_error_or("Failed to read safetensors metadata")),
                }
            }
        }

        Ok(map)
    }
}

/// mlx-c map iterator status after the last entry.
const ITERATOR_END: i32 = 2;

/// Frees the iterator on every exit path, including early error returns.
struct ArrayMapIterator(mlx_sys::mlx_map_string_to_array_iterator);

impl Drop for ArrayMapIterator {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_map_string_to_array_iterator_free(self.0) };
    }
}

/// Frees the iterator on every exit path, including early error returns.
struct StringMapIterator(mlx_sys::mlx_map_string_to_string_iterator);

impl Drop for StringMapIterator {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_map_string_to_string_iterator_free(self.0) };
    }
}
