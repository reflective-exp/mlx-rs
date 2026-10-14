//! Reading and writing GGUF files.
//!
//! A [`GgufFile`] holds named tensors and typed metadata. Loading a file with quantized tensors
//! (Q4_0, Q4_1, Q8_0) yields each as an MLX quantized triplet: the packed weights under the
//! tensor's name, plus `<name>.scales` and `<name>.biases`.
//!
//! ```rust,no_run
//! use mlx_rs::{Array, io::GgufFile};
//!
//! let mut file = GgufFile::new().unwrap();
//! file.insert_array("weight", &Array::from_slice(&[1.0f32, 2.0], &[2])).unwrap();
//! file.insert_metadata("general.name", "example").unwrap();
//! file.save("model.gguf").unwrap();
//!
//! let loaded = GgufFile::load("model.gguf").unwrap();
//! assert_eq!(loaded.get_metadata_string("general.name").unwrap().as_deref(), Some("example"));
//! ```

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char};
use std::marker::PhantomData;
use std::path::Path;
use std::rc::Rc;

use mlx_internal_macros::default_device;

use crate::error::{GgufError, ensure_error_handler, last_mlx_error_or};
use crate::utils::SUCCESS;
use crate::{Array, Dtype, Stream};

/// mlx-c status for a key that is not present.
const NOT_FOUND: i32 = 2;
/// mlx-c status for metadata that is present but holds a different kind of value.
const WRONG_METADATA_KIND: i32 = 3;

/// A GGUF metadata value.
#[derive(Debug, Clone)]
pub enum GgufMetadataValue {
    /// A scalar or one-dimensional array.
    Array(Array),
    /// A string.
    String(String),
    /// A list of strings.
    Strings(Vec<String>),
}

/// The kind of value a GGUF metadata entry holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufMetadataKind {
    /// A scalar or one-dimensional array.
    Array,
    /// A string.
    String,
    /// A list of strings.
    Strings,
}

impl GgufMetadataValue {
    /// The kind of this value.
    pub fn kind(&self) -> GgufMetadataKind {
        match self {
            Self::Array(_) => GgufMetadataKind::Array,
            Self::String(_) => GgufMetadataKind::String,
            Self::Strings(_) => GgufMetadataKind::Strings,
        }
    }
}

impl From<Array> for GgufMetadataValue {
    fn from(value: Array) -> Self {
        Self::Array(value)
    }
}

impl From<&Array> for GgufMetadataValue {
    fn from(value: &Array) -> Self {
        Self::Array(value.clone())
    }
}

impl From<String> for GgufMetadataValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for GgufMetadataValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_owned())
    }
}

impl From<Vec<String>> for GgufMetadataValue {
    fn from(value: Vec<String>) -> Self {
        Self::Strings(value)
    }
}

/// The tensors and metadata of a GGUF file.
///
/// Tensor and metadata keys are separate namespaces. Inserting a key already present in its
/// namespace is an error, not an overwrite.
pub struct GgufFile {
    raw: mlx_sys::mlx_io_gguf,
    // The handle is not safe to share or move across threads.
    _not_send: PhantomData<Rc<()>>,
}

impl std::fmt::Debug for GgufFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufFile").finish_non_exhaustive()
    }
}

impl Drop for GgufFile {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_io_gguf_free(self.raw) };
    }
}

impl GgufFile {
    /// An empty file with no tensors or metadata.
    pub fn new() -> Result<Self, GgufError> {
        ensure_error_handler();
        let raw = unsafe { mlx_sys::mlx_io_gguf_new() };
        if raw.ctx.is_null() {
            return Err(status_error(1, "creating a GGUF container"));
        }
        Ok(Self {
            raw,
            _not_send: PhantomData,
        })
    }

    /// Load a `.gguf` file.
    #[default_device(device = "cpu")]
    pub fn load_device(
        path: impl AsRef<Path>,
        stream: impl AsRef<Stream>,
    ) -> Result<Self, GgufError> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(GgufError::NotFile);
        }
        let path = path_c_string(path)?;
        let mut file = Self::new()?;
        let status = unsafe {
            mlx_sys::mlx_load_gguf(&mut file.raw, path.as_ptr(), stream.as_ref().as_ptr())
        };
        status_result(status, "loading a GGUF file")?;
        Ok(file)
    }

    /// Write the tensors and metadata to a `.gguf` file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), GgufError> {
        let path = path_c_string(path.as_ref())?;
        ensure_error_handler();
        let status = unsafe { mlx_sys::mlx_save_gguf(path.as_ptr(), self.raw) };
        status_result(status, "saving a GGUF file")
    }

    /// The tensor keys, sorted.
    pub fn array_keys(&self) -> Result<Vec<String>, GgufError> {
        ensure_error_handler();
        let mut keys = VectorString(unsafe { mlx_sys::mlx_vector_string_new() });
        let status = unsafe { mlx_sys::mlx_io_gguf_get_keys(&mut keys.0, self.raw) };
        status_result(status, "listing GGUF tensor keys")?;
        let mut keys = keys.to_vec()?;
        keys.sort();
        Ok(keys)
    }

    /// The tensor stored under `key`, or `None` if there is none.
    pub fn get_array(&self, key: &str) -> Result<Option<Array>, GgufError> {
        let key = text_c_string(key)?;
        ensure_error_handler();
        let mut res = unsafe { mlx_sys::mlx_array_new() };
        let status = unsafe { mlx_sys::mlx_io_gguf_get_array(&mut res, self.raw, key.as_ptr()) };
        if status != SUCCESS {
            unsafe { mlx_sys::mlx_array_free(res) };
        }
        match status {
            SUCCESS => Ok(Some(unsafe { Array::from_ptr(res) })),
            NOT_FOUND => Ok(None),
            _ => Err(status_error(status, "reading a GGUF tensor")),
        }
    }

    /// Every tensor, by key.
    pub fn arrays(&self) -> Result<HashMap<String, Array>, GgufError> {
        self.array_keys()?
            .into_iter()
            .map(|key| {
                let array = self
                    .get_array(&key)?
                    .ok_or_else(|| status_error(NOT_FOUND, "reading a listed GGUF tensor"))?;
                Ok((key, array))
            })
            .collect()
    }

    /// Add a tensor under a new `key`.
    ///
    /// GGUF stores `float32`, `float16`, `int8`, `int16` and `int32` tensors; other dtypes are
    /// rejected here rather than when saving.
    pub fn insert_array(&mut self, key: impl AsRef<str>, value: &Array) -> Result<(), GgufError> {
        let key_text = key.as_ref();
        if self.get_array(key_text)?.is_some() {
            return Err(GgufError::ArrayKeyAlreadyExists {
                key: key_text.to_owned(),
            });
        }
        validate_tensor_dtype(value.dtype())?;
        let key = text_c_string(key_text)?;
        ensure_error_handler();
        let status =
            unsafe { mlx_sys::mlx_io_gguf_set_array(self.raw, key.as_ptr(), value.as_ptr()) };
        status_result(status, "inserting a GGUF tensor")
    }

    /// The kind of the metadata under `key`, or `None` if there is none.
    pub fn metadata_kind(&self, key: &str) -> Result<Option<GgufMetadataKind>, GgufError> {
        type HasMetadata =
            unsafe extern "C" fn(*mut bool, mlx_sys::mlx_io_gguf, *const c_char) -> i32;
        let probes: [(GgufMetadataKind, HasMetadata); 3] = [
            (
                GgufMetadataKind::Array,
                mlx_sys::mlx_io_gguf_has_metadata_array,
            ),
            (
                GgufMetadataKind::String,
                mlx_sys::mlx_io_gguf_has_metadata_string,
            ),
            (
                GgufMetadataKind::Strings,
                mlx_sys::mlx_io_gguf_has_metadata_vector_string,
            ),
        ];

        let key = text_c_string(key)?;
        ensure_error_handler();
        for (kind, has_metadata) in probes {
            let mut present = false;
            match unsafe { has_metadata(&mut present, self.raw, key.as_ptr()) } {
                SUCCESS if present => return Ok(Some(kind)),
                SUCCESS => {}
                NOT_FOUND => return Ok(None),
                status => return Err(status_error(status, "checking a GGUF metadata kind")),
            }
        }
        Err(status_error(
            WRONG_METADATA_KIND,
            "identifying a GGUF metadata kind",
        ))
    }

    /// The metadata under `key`, or `None` if there is none.
    pub fn get_metadata(&self, key: &str) -> Result<Option<GgufMetadataValue>, GgufError> {
        Ok(match self.metadata_kind(key)? {
            Some(GgufMetadataKind::Array) => self.get_metadata_array(key)?.map(Into::into),
            Some(GgufMetadataKind::String) => self.get_metadata_string(key)?.map(Into::into),
            Some(GgufMetadataKind::Strings) => self.get_metadata_strings(key)?.map(Into::into),
            None => None,
        })
    }

    /// The array metadata under `key`, or `None` if there is none.
    pub fn get_metadata_array(&self, key: &str) -> Result<Option<Array>, GgufError> {
        let c_key = text_c_string(key)?;
        ensure_error_handler();
        let mut res = unsafe { mlx_sys::mlx_array_new() };
        let status =
            unsafe { mlx_sys::mlx_io_gguf_get_metadata_array(&mut res, self.raw, c_key.as_ptr()) };
        if status != SUCCESS {
            unsafe { mlx_sys::mlx_array_free(res) };
        }
        match status {
            SUCCESS => Ok(Some(unsafe { Array::from_ptr(res) })),
            NOT_FOUND => Ok(None),
            WRONG_METADATA_KIND => Err(self.wrong_kind(key, GgufMetadataKind::Array)),
            _ => Err(status_error(status, "reading GGUF array metadata")),
        }
    }

    /// The string metadata under `key`, or `None` if there is none.
    pub fn get_metadata_string(&self, key: &str) -> Result<Option<String>, GgufError> {
        let c_key = text_c_string(key)?;
        ensure_error_handler();
        let mut res = MlxString(unsafe { mlx_sys::mlx_string_new() });
        let status = unsafe {
            mlx_sys::mlx_io_gguf_get_metadata_string(&mut res.0, self.raw, c_key.as_ptr())
        };
        match status {
            SUCCESS => Ok(Some(res.to_string()?)),
            NOT_FOUND => Ok(None),
            WRONG_METADATA_KIND => Err(self.wrong_kind(key, GgufMetadataKind::String)),
            _ => Err(status_error(status, "reading GGUF string metadata")),
        }
    }

    /// The string-list metadata under `key`, or `None` if there is none.
    pub fn get_metadata_strings(&self, key: &str) -> Result<Option<Vec<String>>, GgufError> {
        let c_key = text_c_string(key)?;
        ensure_error_handler();
        let mut res = VectorString(unsafe { mlx_sys::mlx_vector_string_new() });
        let status = unsafe {
            mlx_sys::mlx_io_gguf_get_metadata_vector_string(&mut res.0, self.raw, c_key.as_ptr())
        };
        match status {
            SUCCESS => Ok(Some(res.to_vec()?)),
            NOT_FOUND => Ok(None),
            WRONG_METADATA_KIND => Err(self.wrong_kind(key, GgufMetadataKind::Strings)),
            _ => Err(status_error(status, "reading GGUF string-list metadata")),
        }
    }

    /// Add metadata under a new `key`.
    ///
    /// Array metadata must be a non-empty scalar or one-dimensional array of a boolean, integer
    /// or `float32` dtype.
    pub fn insert_metadata(
        &mut self,
        key: impl AsRef<str>,
        value: impl Into<GgufMetadataValue>,
    ) -> Result<(), GgufError> {
        let key_text = key.as_ref();
        let value = value.into();
        if self.metadata_kind(key_text)?.is_some() {
            return Err(GgufError::MetadataKeyAlreadyExists {
                key: key_text.to_owned(),
            });
        }
        let key = text_c_string(key_text)?;

        let status = match value {
            GgufMetadataValue::Array(array) => {
                validate_metadata_array(&array)?;
                ensure_error_handler();
                unsafe {
                    mlx_sys::mlx_io_gguf_set_metadata_array(self.raw, key.as_ptr(), array.as_ptr())
                }
            }
            GgufMetadataValue::String(value) => {
                let value = text_c_string(&value)?;
                ensure_error_handler();
                unsafe {
                    mlx_sys::mlx_io_gguf_set_metadata_string(self.raw, key.as_ptr(), value.as_ptr())
                }
            }
            GgufMetadataValue::Strings(values) => {
                let values = values
                    .iter()
                    .map(|value| text_c_string(value))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut pointers: Vec<*const c_char> =
                    values.iter().map(|value| value.as_ptr()).collect();
                ensure_error_handler();
                let vector = VectorString(unsafe {
                    mlx_sys::mlx_vector_string_new_data(pointers.as_mut_ptr(), pointers.len())
                });
                unsafe {
                    mlx_sys::mlx_io_gguf_set_metadata_vector_string(
                        self.raw,
                        key.as_ptr(),
                        vector.0,
                    )
                }
            }
        };
        status_result(status, "inserting GGUF metadata")
    }

    fn wrong_kind(&self, key: &str, expected: GgufMetadataKind) -> GgufError {
        match self.metadata_kind(key) {
            Ok(Some(actual)) => GgufError::WrongMetadataKind {
                key: key.to_owned(),
                expected,
                actual,
            },
            Ok(None) => status_error(NOT_FOUND, "resolving a present GGUF metadata key"),
            Err(error) => error,
        }
    }
}

/// An owned `mlx_string`, freed on drop.
struct MlxString(mlx_sys::mlx_string);

impl MlxString {
    fn to_string(&self) -> Result<String, GgufError> {
        let data = unsafe { mlx_sys::mlx_string_data(self.0) };
        if data.is_null() {
            return Err(status_error(1, "reading an MLX string"));
        }
        utf8(data)
    }
}

impl Drop for MlxString {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_string_free(self.0) };
    }
}

/// An owned `mlx_vector_string`, freed on drop.
struct VectorString(mlx_sys::mlx_vector_string);

impl VectorString {
    fn to_vec(&self) -> Result<Vec<String>, GgufError> {
        let len = unsafe { mlx_sys::mlx_vector_string_size(self.0) };
        (0..len)
            .map(|index| {
                let mut data = std::ptr::null_mut();
                let status = unsafe { mlx_sys::mlx_vector_string_get(&mut data, self.0, index) };
                status_result(status, "reading an MLX string vector")?;
                if data.is_null() {
                    return Err(status_error(1, "reading an MLX string vector"));
                }
                utf8(data)
            })
            .collect()
    }
}

impl Drop for VectorString {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_vector_string_free(self.0) };
    }
}

fn utf8(data: *const c_char) -> Result<String, GgufError> {
    unsafe { CStr::from_ptr(data) }
        .to_str()
        .map(str::to_owned)
        .map_err(|_| GgufError::InvalidUtf8)
}

#[track_caller]
fn status_error(status: i32, operation: &str) -> GgufError {
    GgufError::Exception(last_mlx_error_or(&format!(
        "{operation} failed with status {status}"
    )))
}

#[track_caller]
fn status_result(status: i32, operation: &str) -> Result<(), GgufError> {
    match status {
        SUCCESS => Ok(()),
        _ => Err(status_error(status, operation)),
    }
}

fn text_c_string(value: &str) -> Result<CString, GgufError> {
    CString::new(value).map_err(|_| GgufError::InteriorNul)
}

fn path_c_string(path: &Path) -> Result<CString, GgufError> {
    if path.extension().and_then(|extension| extension.to_str()) != Some("gguf") {
        return Err(GgufError::UnsupportedExtension);
    }
    text_c_string(path.to_str().ok_or(GgufError::InvalidPathUtf8)?)
}

fn validate_tensor_dtype(dtype: Dtype) -> Result<(), GgufError> {
    match dtype {
        Dtype::Float32 | Dtype::Float16 | Dtype::Int8 | Dtype::Int16 | Dtype::Int32 => Ok(()),
        dtype => Err(GgufError::UnsupportedTensorDtype { dtype }),
    }
}

fn validate_metadata_array(array: &Array) -> Result<(), GgufError> {
    let rank = array.ndim();
    if rank > 1 {
        return Err(GgufError::InvalidMetadataArrayRank { rank });
    }
    if array.size() == 0 {
        return Err(GgufError::EmptyMetadataArray);
    }
    match array.dtype() {
        Dtype::Bool
        | Dtype::Int8
        | Dtype::Int16
        | Dtype::Int32
        | Dtype::Int64
        | Dtype::Uint8
        | Dtype::Uint16
        | Dtype::Uint32
        | Dtype::Uint64
        | Dtype::Float32 => Ok(()),
        dtype => Err(GgufError::UnsupportedMetadataArrayDtype { dtype }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array;

    #[test]
    fn round_trips_tensors_and_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");

        let weight = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let ids = Array::from_slice(&[7i32, 8, 9], &[3]);
        let mut file = GgufFile::new().unwrap();
        file.insert_array("weight", &weight).unwrap();
        file.insert_array("ids", &ids).unwrap();
        file.insert_metadata("general.name", "tiny").unwrap();
        file.insert_metadata("tokens", vec!["a".to_owned(), "b".to_owned()])
            .unwrap();
        let block_count = Array::from_slice(&[4u32], &[]);
        file.insert_metadata("block_count", block_count).unwrap();
        file.insert_metadata("scales", array!([0.5f32, 0.25]))
            .unwrap();
        file.save(&path).unwrap();

        let loaded = GgufFile::load(&path).unwrap();
        assert_eq!(loaded.array_keys().unwrap(), ["ids", "weight"]);
        let arrays = loaded.arrays().unwrap();
        assert_eq!(arrays["weight"].shape(), &[2, 3]);
        assert_eq!(arrays["weight"].as_slice::<f32>(), weight.as_slice::<f32>());
        assert_eq!(arrays["ids"].as_slice::<i32>(), &[7, 8, 9]);

        assert_eq!(
            loaded
                .get_metadata_string("general.name")
                .unwrap()
                .as_deref(),
            Some("tiny")
        );
        assert_eq!(
            loaded.get_metadata_strings("tokens").unwrap(),
            Some(vec!["a".to_owned(), "b".to_owned()])
        );
        let block_count = loaded.get_metadata_array("block_count").unwrap().unwrap();
        assert_eq!(block_count.item::<u32>(), 4);
        let scales = loaded.get_metadata("scales").unwrap().unwrap();
        assert_eq!(scales.kind(), GgufMetadataKind::Array);
        let GgufMetadataValue::Array(scales) = scales else {
            unreachable!()
        };
        assert_eq!(scales.as_slice::<f32>(), &[0.5, 0.25]);
    }

    #[test]
    fn missing_keys_are_none_and_wrong_kinds_are_errors() {
        let mut file = GgufFile::new().unwrap();
        file.insert_metadata("name", "tiny").unwrap();

        assert!(file.get_array("absent").unwrap().is_none());
        assert!(file.get_metadata("absent").unwrap().is_none());
        assert_eq!(
            file.metadata_kind("name").unwrap(),
            Some(GgufMetadataKind::String)
        );
        assert!(matches!(
            file.get_metadata_array("name"),
            Err(GgufError::WrongMetadataKind {
                expected: GgufMetadataKind::Array,
                actual: GgufMetadataKind::String,
                ..
            })
        ));
    }

    #[test]
    fn duplicate_keys_are_rejected_per_namespace() {
        let mut file = GgufFile::new().unwrap();
        let value = Array::from_int(1);
        file.insert_array("same", &value).unwrap();
        assert!(matches!(
            file.insert_array("same", &value),
            Err(GgufError::ArrayKeyAlreadyExists { key }) if key == "same"
        ));

        // Metadata keys are a separate namespace from tensor keys.
        file.insert_metadata("same", "first").unwrap();
        assert!(matches!(
            file.insert_metadata("same", vec!["second".to_owned()]),
            Err(GgufError::MetadataKeyAlreadyExists { key }) if key == "same"
        ));
    }

    #[test]
    fn unsupported_values_are_rejected_before_ffi() {
        let mut file = GgufFile::new().unwrap();
        for dtype in [Dtype::Bool, Dtype::Uint32, Dtype::Int64, Dtype::Bfloat16] {
            let array = Array::from_int(1).as_dtype(dtype, None).unwrap();
            assert!(matches!(
                file.insert_array("t", &array),
                Err(GgufError::UnsupportedTensorDtype { dtype: actual }) if actual == dtype
            ));
        }

        let matrix = Array::from_slice(&[1i32, 2, 3, 4], &[2, 2]);
        assert!(matches!(
            file.insert_metadata("m", matrix),
            Err(GgufError::InvalidMetadataArrayRank { rank: 2 })
        ));
        let empty = Array::from_slice::<i32>(&[], &[0]);
        assert!(matches!(
            file.insert_metadata("m", empty),
            Err(GgufError::EmptyMetadataArray)
        ));
        let half = Array::from_slice(&[half::f16::from_f32(1.0)], &[1]);
        assert!(matches!(
            file.insert_metadata("m", half),
            Err(GgufError::UnsupportedMetadataArrayDtype { .. })
        ));
        assert!(matches!(
            file.insert_array("bad\0key", &Array::from_int(1)),
            Err(GgufError::InteriorNul)
        ));
    }

    #[test]
    fn paths_are_validated() {
        assert!(matches!(
            GgufFile::load("definitely-absent.gguf"),
            Err(GgufError::NotFile)
        ));
        let file = GgufFile::new().unwrap();
        assert!(matches!(
            file.save("wrong-extension.bin"),
            Err(GgufError::UnsupportedExtension)
        ));
    }
}
