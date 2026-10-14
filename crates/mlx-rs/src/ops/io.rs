use crate::error::{IoError, ensure_error_handler, last_mlx_error_or};
use crate::utils::SUCCESS;
use crate::utils::guard::Guarded;
use crate::utils::io::SafeTensors;
use crate::{Array, Stream};
use std::collections::HashMap;
use std::ffi::CString;
use std::path::Path;

/// Arrays loaded from a `safetensors` file paired with its `[String:String]` metadata.
type SafetensorsWithMetadata = (HashMap<String, Array>, HashMap<String, String>);

fn check_file_extension(path: &Path, expected: &str) -> Result<(), IoError> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext == expected => Ok(()),
        _ => Err(IoError::UnsupportedFormat),
    }
}

impl Array {
    /// Load array from a binary file in `.npy` format.
    ///
    /// # Params
    ///
    /// - path: path of file to load
    pub fn load_numpy(path: impl AsRef<Path>) -> Result<Array, IoError> {
        let stream = Stream::thread_local_or_cpu();
        let path = path.as_ref();
        if !path.is_file() {
            return Err(IoError::NotFile);
        }
        let c_path = CString::new(path.to_str().ok_or(IoError::InvalidUtf8)?)?;
        check_file_extension(path, "npy")?;

        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_load(res, c_path.as_ptr(), stream.as_ref().as_ptr())
        })
        .map_err(Into::into)
    }

    /// Load dictionary of ``MLXArray`` from a `safetensors` file.
    ///
    /// # Params
    ///
    /// - path: path of file to load
    ///
    pub fn load_safetensors(path: impl AsRef<Path>) -> Result<HashMap<String, Array>, IoError> {
        let stream = Stream::thread_local_or_cpu();
        let safetensors = SafeTensors::load_device(path.as_ref(), stream)?;
        let data = safetensors.data()?;
        Ok(data)
    }

    /// Load dictionary of ``MLXArray`` and metadata `[String:String]` from a `safetensors` file.
    ///
    /// # Params
    ///
    /// - path: path of file to load
    pub fn load_safetensors_with_metadata(
        path: impl AsRef<Path>,
    ) -> Result<SafetensorsWithMetadata, IoError> {
        let stream = Stream::thread_local_or_cpu();
        let safetensors = SafeTensors::load_device(path.as_ref(), stream)?;
        let data = safetensors.data()?;
        let metadata = safetensors.metadata()?;

        Ok((data, metadata))
    }

    /// Save array to a binary file in `.npy`format.
    ///
    /// # Params
    ///
    /// - array: array to save
    /// - url: URL of file to load
    pub fn save_numpy(&self, path: impl AsRef<Path>) -> Result<(), IoError> {
        let path = path.as_ref();
        check_file_extension(path, "npy")?;
        let c_path = CString::new(path.to_str().ok_or(IoError::InvalidUtf8)?)?;

        unsafe { mlx_sys::mlx_save(c_path.as_ptr(), self.as_ptr()) };

        Ok(())
    }

    /// Save dictionary of arrays in `safetensors` format.
    ///
    /// # Params
    ///
    /// - arrays: arrays to save
    /// - metadata: metadata to save
    /// - path: path of file to save
    pub fn save_safetensors<'a, I, S, V>(
        arrays: I,
        metadata: impl Into<Option<&'a HashMap<String, String>>>,
        path: impl AsRef<Path>,
    ) -> Result<(), IoError>
    where
        I: IntoIterator<Item = (S, V)>,
        S: AsRef<str>,
        V: AsRef<Array>,
    {
        ensure_error_handler();

        let path = path.as_ref();

        check_file_extension(path, "safetensors")?;

        // The guards free both maps on every return, including `?` exits.
        let c_arrays = ArrayMap(unsafe { mlx_sys::mlx_map_string_to_array_new() });
        for (key, array) in arrays.into_iter() {
            let key = CString::new(key.as_ref())?;
            let status = unsafe {
                mlx_sys::mlx_map_string_to_array_insert(
                    c_arrays.0,
                    key.as_ptr(),
                    array.as_ref().as_ptr(),
                )
            };
            if status != SUCCESS {
                return Err(last_mlx_error_or("Failed to collect safetensors arrays").into());
            }
        }

        let default_metadata = HashMap::new();
        let metadata_ref = metadata.into().unwrap_or(&default_metadata);

        let c_metadata = StringMap(unsafe { mlx_sys::mlx_map_string_to_string_new() });
        for (key, value) in metadata_ref.iter() {
            let key = CString::new(key.as_str())?;
            let value = CString::new(value.as_str())?;
            let status = unsafe {
                mlx_sys::mlx_map_string_to_string_insert(c_metadata.0, key.as_ptr(), value.as_ptr())
            };
            if status != SUCCESS {
                return Err(last_mlx_error_or("Failed to collect safetensors metadata").into());
            }
        }

        let c_path = CString::new(path.to_str().ok_or(IoError::InvalidUtf8)?)?;

        let status =
            unsafe { mlx_sys::mlx_save_safetensors(c_path.as_ptr(), c_arrays.0, c_metadata.0) };
        if status != SUCCESS {
            return Err(last_mlx_error_or("Failed to save safetensors").into());
        }

        Ok(())
    }
}

struct ArrayMap(mlx_sys::mlx_map_string_to_array);

impl Drop for ArrayMap {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_map_string_to_array_free(self.0) };
    }
}

struct StringMap(mlx_sys::mlx_map_string_to_string);

impl Drop for StringMap {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_map_string_to_string_free(self.0) };
    }
}

#[cfg(test)]
mod tests {
    use crate::Array;

    #[test]
    fn test_save_arrays() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("test.safetensors");

        let mut arrays = std::collections::HashMap::new();
        arrays.insert("foo".to_string(), Array::ones::<i32>(&[1, 2]).unwrap());
        arrays.insert("bar".to_string(), Array::zeros::<i32>(&[2, 1]).unwrap());

        Array::save_safetensors(&arrays, None, &path).unwrap();

        let loaded_arrays = Array::load_safetensors(&path).unwrap();

        // compare values
        let mut loaded_keys: Vec<_> = loaded_arrays.keys().cloned().collect();
        let mut original_keys: Vec<_> = arrays.keys().cloned().collect();
        loaded_keys.sort();
        original_keys.sort();
        assert_eq!(loaded_keys, original_keys);

        for key in loaded_keys {
            let loaded_array = loaded_arrays.get(&key).unwrap();
            let original_array = arrays.get(&key).unwrap();
            assert!(
                loaded_array
                    .all_close(original_array, None, None, None)
                    .unwrap()
                    .item::<bool>()
            );
        }
    }

    #[test]
    fn test_save_arrays_with_metadata() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("test.safetensors");

        let mut arrays = std::collections::HashMap::new();
        arrays.insert("foo".to_string(), Array::ones::<i32>(&[1, 2]).unwrap());
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("format".to_string(), "mlx".to_string());
        metadata.insert("step".to_string(), "42".to_string());

        Array::save_safetensors(&arrays, &metadata, &path).unwrap();

        let (loaded_arrays, loaded_metadata) =
            Array::load_safetensors_with_metadata(&path).unwrap();
        assert_eq!(loaded_metadata, metadata);
        assert_eq!(loaded_arrays.len(), 1);
        assert_eq!(loaded_arrays["foo"].as_slice::<i32>(), &[1, 1]);
    }

    #[test]
    fn test_save_arrays_to_missing_directory_is_an_error() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("missing").join("test.safetensors");

        let mut arrays = std::collections::HashMap::new();
        arrays.insert("foo".to_string(), Array::ones::<i32>(&[1, 2]).unwrap());

        assert!(Array::save_safetensors(&arrays, None, &path).is_err());
    }

    #[test]
    fn test_save_array() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("test.npy");

        let a = Array::ones::<i32>(&[2, 4]).unwrap();
        a.save_numpy(&path).unwrap();

        let b = Array::load_numpy(&path).unwrap();
        assert!(a.all_close(&b, None, None, None).unwrap().item::<bool>());
    }
}
