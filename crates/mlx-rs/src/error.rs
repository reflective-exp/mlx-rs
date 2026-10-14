//! Custom error types and handler for the c ffi

use crate::Dtype;
use libc::strdup;
use std::any::Any;
use std::convert::Infallible;
use std::ffi::NulError;
use std::panic::Location;
use std::sync::Once;
use std::{cell::Cell, ffi::c_char};
use thiserror::Error;

/// Type alias for a `Result` with an `Exception` error type.
pub type Result<T> = std::result::Result<T, Exception>;

/// Error reading, building, or writing a [`crate::io::GgufFile`].
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum GgufError {
    /// The path does not point to an existing file.
    #[error("path must point to a local file")]
    NotFile,

    /// The path is not valid UTF-8.
    #[error("path is not valid UTF-8")]
    InvalidPathUtf8,

    /// The path does not end in `.gguf`.
    #[error("path must have a .gguf extension")]
    UnsupportedExtension,

    /// A key or string value contains an interior nul byte.
    #[error("GGUF text contains an interior nul byte")]
    InteriorNul,

    /// A key or string read from the file is not valid UTF-8.
    #[error("GGUF text is not valid UTF-8")]
    InvalidUtf8,

    /// A tensor with this key is already present.
    #[error("array key {key:?} already exists")]
    ArrayKeyAlreadyExists {
        /// The duplicate key.
        key: String,
    },

    /// Metadata with this key is already present.
    #[error("metadata key {key:?} already exists")]
    MetadataKeyAlreadyExists {
        /// The duplicate key.
        key: String,
    },

    /// The metadata holds a different kind of value than requested.
    #[error("metadata {key:?} has kind {actual:?}, expected {expected:?}")]
    WrongMetadataKind {
        /// The metadata key.
        key: String,
        /// The requested kind.
        expected: crate::io::GgufMetadataKind,
        /// The stored kind.
        actual: crate::io::GgufMetadataKind,
    },

    /// GGUF cannot store tensors of this dtype.
    #[error("tensor dtype {dtype:?} cannot be written as GGUF")]
    UnsupportedTensorDtype {
        /// The rejected dtype.
        dtype: Dtype,
    },

    /// GGUF cannot store metadata arrays of this dtype.
    #[error("metadata array dtype {dtype:?} cannot be written as GGUF")]
    UnsupportedMetadataArrayDtype {
        /// The rejected dtype.
        dtype: Dtype,
    },

    /// GGUF metadata arrays must be scalar or one-dimensional.
    #[error("metadata arrays must be scalar or one-dimensional, found rank {rank}")]
    InvalidMetadataArrayRank {
        /// The rejected rank.
        rank: usize,
    },

    /// GGUF metadata arrays cannot be empty.
    #[error("metadata arrays cannot be empty")]
    EmptyMetadataArray,

    /// MLX reported an error.
    #[error(transparent)]
    Exception(#[from] Exception),
}

/// Error with io operations
#[derive(Error, PartialEq, Debug)]
pub enum IoError {
    /// Path must point to a local file
    #[error("Path must point to a local file")]
    NotFile,

    /// Path contains invalid UTF-8
    #[error("Path contains invalid UTF-8")]
    InvalidUtf8,

    /// Path contains null bytes
    #[error("Path contains null bytes")]
    NullBytes,

    /// No file extension found
    #[error("No file extension found")]
    NoExtension,

    /// Unsupported file format
    #[error("Unsupported file format")]
    UnsupportedFormat,

    /// Unable to open file
    #[error("Unable to open file")]
    UnableToOpenFile,

    /// Unable to allocate memory
    #[error("Unable to allocate memory")]
    AllocationError,

    /// Null error
    #[error(transparent)]
    NulError(#[from] NulError),

    /// Error with unfalttening the loaded optimizer state
    #[error(transparent)]
    Unflatten(#[from] UnflattenError),

    /// Exception
    #[error(transparent)]
    Exception(#[from] Exception),
}

impl From<Infallible> for IoError {
    fn from(_: Infallible) -> Self {
        unreachable!()
    }
}

impl From<RawException> for IoError {
    #[track_caller]
    fn from(e: RawException) -> Self {
        let exception = Exception {
            what: e.what,
            location: Location::caller(),
        };
        Self::Exception(exception)
    }
}

/// Error associated with `Array::try_as_slice()`
#[derive(Debug, PartialEq, Error)]
pub enum AsSliceError {
    /// The array is a strided view (e.g. a broadcast or transpose), not row-major, so it cannot
    /// be borrowed as a flat slice.
    #[error(
        "array data is not row-contiguous; call `Array::contiguous()` before borrowing it as a slice"
    )]
    NotContiguous,

    /// The underlying data pointer is null.
    ///
    /// This is likely because the array has not been evaluated yet.
    #[error("The data pointer is null.")]
    Null,

    /// The output dtype does not match the data type of the array.
    #[error("dtype mismatch: expected {expecting:?}, found {found:?}")]
    DtypeMismatch {
        /// The expected data type.
        expecting: Dtype,

        /// The actual data type
        found: Dtype,
    },

    /// Exception
    #[error(transparent)]
    Exception(#[from] Exception),
}

/// Error with unflattening a loaded optimizer state
#[derive(Debug, PartialEq, Error)]
pub enum UnflattenError {
    /// Expecting next (key, value) pair, found none
    #[error("Expecting next (key, value) pair, found none")]
    ExpectingNextPair,

    /// The key is not in a valid format
    #[error("Invalid key")]
    InvalidKey,
}

/// Error with loading an optimizer state
#[derive(Debug, PartialEq, Error)]
pub enum OptimizerStateLoadError {
    /// Error with io operations
    #[error(transparent)]
    Io(#[from] IoError),

    /// Error with unflattening the optimizer state
    #[error(transparent)]
    Unflatten(#[from] UnflattenError),
}

impl From<Infallible> for OptimizerStateLoadError {
    fn from(_: Infallible) -> Self {
        unreachable!()
    }
}

cfg_safetensors! {
    /// Error associated with conversion between `safetensors::tensor::TensorView` and `Array`
    /// when the data type is not supported.
    #[derive(Debug, Error)]
    pub enum ConversionError {
        /// The safetensors data type that is not supported.
        ///
        /// This is the error type for conversions from `safetensors::tensor::TensorView` to `Array`.
        #[error("The safetensors data type {0:?} is not supported.")]
        SafeTensorDtype(safetensors::tensor::Dtype),

        /// The mlx data type that is not supported.
        ///
        /// This is the error type for conversions from `Array` to `safetensors::tensor::TensorView`.
        #[error("The mlx data type {0:?} is not supported.")]
        MlxDtype(crate::Dtype),

        /// The array cannot be borrowed as a contiguous slice.
        #[error(transparent)]
        ArraySlice(#[from] AsSliceError),

        /// Error casting the data buffer to `&[u8]`.
        #[error(transparent)]
        PodCastError(#[from] bytemuck::PodCastError),

        /// Error with creating a `safetensors::tensor::TensorView`.
        #[error(transparent)]
        SafeTensorError(#[from] safetensors::tensor::SafeTensorError),
    }
}

pub(crate) struct RawException {
    pub(crate) what: String,
}

/// Exception. Most will come from the C API.
#[derive(Debug, PartialEq, Error)]
#[error("{what:?} at {location}")]
pub struct Exception {
    pub(crate) what: String,
    pub(crate) location: &'static Location<'static>,
}

impl Exception {
    /// The error message.
    pub fn what(&self) -> &str {
        &self.what
    }

    /// The location of the error.
    ///
    /// The location is obtained from `std::panic::Location::caller()` and points
    /// to the location in the code where the error was created and not where it was
    /// propagated.
    pub fn location(&self) -> &'static Location<'static> {
        self.location
    }

    /// Creates a new exception with the given message.
    #[track_caller]
    pub fn custom(what: impl Into<String>) -> Self {
        Self {
            what: what.into(),
            location: Location::caller(),
        }
    }
}

impl From<RawException> for Exception {
    #[track_caller]
    fn from(e: RawException) -> Self {
        Self {
            what: e.what,
            location: Location::caller(),
        }
    }
}

impl From<&str> for Exception {
    #[track_caller]
    fn from(what: &str) -> Self {
        Self {
            what: what.to_string(),
            location: Location::caller(),
        }
    }
}

impl From<Infallible> for Exception {
    fn from(_: Infallible) -> Self {
        unreachable!()
    }
}

impl From<Exception> for String {
    fn from(e: Exception) -> Self {
        e.what
    }
}

/// The calling thread's most recent MLX error message, `strdup`ed by the handler and freed when
/// drained, overwritten, or left behind at thread exit.
struct LastErrorStash(Cell<*const c_char>);

impl LastErrorStash {
    fn replace(&self, ptr: *const c_char) -> *const c_char {
        self.0.replace(ptr)
    }
}

impl Drop for LastErrorStash {
    fn drop(&mut self) {
        free_message(self.0.replace(std::ptr::null()));
    }
}

fn free_message(ptr: *const c_char) {
    if !ptr.is_null() {
        unsafe { libc::free(ptr as *mut libc::c_void) };
    }
}

/// Why a Rust closure called back from MLX failed. A panic cannot unwind through the FFI frames
/// between the trampoline and Rust, so it is carried across as a value and resumed afterwards.
enum ClosureFailure {
    Error(Exception),
    Panic(Box<dyn Any + Send>),
}

thread_local! {
    static CLOSURE_ERROR: Cell<Option<ClosureFailure>> = const { Cell::new(None) };
    static LAST_MLX_ERROR: LastErrorStash = const { LastErrorStash(Cell::new(std::ptr::null())) };
}

/// mlx-c keeps one error handler for the whole process in unsynchronized globals, so it is
/// registered once rather than per thread.
pub(crate) static INIT_ERR_HANDLER: Once = Once::new();

#[unsafe(no_mangle)]
extern "C" fn default_mlx_error_handler(msg: *const c_char, _data: *mut std::ffi::c_void) {
    // MLX can report several errors before a caller drains one; keep the newest.
    let message = unsafe { strdup(msg) };
    // `try_with` because MLX may report an error while this thread's locals are being destroyed.
    let stashed = LAST_MLX_ERROR.try_with(|last_error| free_message(last_error.replace(message)));
    if stashed.is_err() {
        free_message(message);
    }
}

/// Delivers each error through the calling thread's [`LAST_MLX_ERROR`], so the handler needs no
/// data pointer.
pub(crate) fn setup_mlx_error_handler() {
    unsafe {
        mlx_sys::mlx_set_error_handler(Some(default_mlx_error_handler), std::ptr::null_mut(), None);
    }
}

pub(crate) fn set_closure_error(err: Exception) {
    CLOSURE_ERROR.with(|closure_error| closure_error.set(Some(ClosureFailure::Error(err))));
}

/// Record a panic caught in a closure trampoline, to be resumed once control is back in Rust.
pub(crate) fn set_closure_panic(payload: Box<dyn Any + Send>) {
    CLOSURE_ERROR.with(|closure_error| closure_error.set(Some(ClosureFailure::Panic(payload))));
}

pub(crate) fn get_and_clear_closure_error() -> Option<Exception> {
    CLOSURE_ERROR.with(|closure_error| match closure_error.take() {
        Some(ClosureFailure::Error(err)) => Some(err),
        Some(ClosureFailure::Panic(payload)) => resume_closure_unwind(payload),
        None => None,
    })
}

/// Resume a panic a closure trampoline caught, now that the FFI call has returned to Rust.
///
/// A closure's [`Exception`] is left in place for the transform that reports it.
pub(crate) fn resume_closure_panic() {
    CLOSURE_ERROR.with(|closure_error| match closure_error.take() {
        Some(ClosureFailure::Panic(payload)) => resume_closure_unwind(payload),
        failure => closure_error.set(failure),
    });
}

fn resume_closure_unwind(payload: Box<dyn Any + Send>) -> ! {
    // The trampoline's failure status made MLX record an error of its own; the panic is the real
    // cause, so drop that message rather than let it surface against a later operation.
    let _ = get_and_clear_last_mlx_error();
    std::panic::resume_unwind(payload)
}

/// Install the MLX error handler so a subsequent failing FFI call records its message, and drop
/// any message left over from an earlier call.
///
/// [`crate::utils::guard::Guarded::try_from_op`] does this itself; call this before FFI that
/// cannot go through it (out-params of types `Guarded` does not cover).
///
/// Clearing matters because some mlx-c functions report failure with a status code without calling
/// `mlx_error` at all — `mlx_device_info_get_string` returns `2` for a missing key, for example.
/// Without this, [`last_mlx_error_or`] would report a stale, unrelated message instead of its
/// fallback.
pub(crate) fn ensure_error_handler() {
    INIT_ERR_HANDLER.call_once(setup_mlx_error_handler);
    let _ = get_and_clear_last_mlx_error();
}

/// Take the message recorded by the last failing FFI call, falling back to `fallback` when MLX
/// recorded none. Pair with [`ensure_error_handler`].
#[track_caller]
pub(crate) fn last_mlx_error_or(fallback: &str) -> Exception {
    let what = get_and_clear_last_mlx_error()
        .map(|e| e.what)
        .unwrap_or_else(|| fallback.to_string());
    Exception {
        what,
        location: Location::caller(),
    }
}

#[track_caller]
pub(crate) fn get_and_clear_last_mlx_error() -> Option<RawException> {
    LAST_MLX_ERROR.with(|last_error| {
        let last_err_ptr = last_error.replace(std::ptr::null());
        if last_err_ptr.is_null() {
            return None;
        }

        let last_err = unsafe {
            std::ffi::CStr::from_ptr(last_err_ptr)
                .to_string_lossy()
                .into_owned()
        };
        free_message(last_err_ptr);

        Some(RawException { what: last_err })
    })
}

/// Error with building a cross-entropy loss function
#[derive(Debug, Clone, PartialEq, Error)]
pub enum CrossEntropyBuildError {
    /// Label smoothing factor must be in the range [0, 1)
    #[error("Label smoothing factor must be in the range [0, 1)")]
    InvalidLabelSmoothingFactor,
}

impl From<CrossEntropyBuildError> for Exception {
    fn from(value: CrossEntropyBuildError) -> Self {
        Exception::custom(format!("{value}"))
    }
}

/// Error with building a RmsProp optimizer
#[derive(Debug, Clone, PartialEq, Error)]
pub enum RmsPropBuildError {
    /// Alpha must be non-negative
    #[error("alpha must be non-negative")]
    NegativeAlpha,

    /// Epsilon must be non-negative
    #[error("epsilon must be non-negative")]
    NegativeEpsilon,
}

/// Error with building an AdaDelta optimizer
#[derive(Debug, Clone, PartialEq, Error)]
pub enum AdaDeltaBuildError {
    /// Rho must be non-negative
    #[error("rho must be non-negative")]
    NegativeRho,

    /// Epsilon must be non-negative
    #[error("epsilon must be non-negative")]
    NegativeEps,
}

/// Error with building an Adafactor optimizer.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum AdafactorBuildError {
    /// Either learning rate is provided or relative step is set to true.
    #[error("Either learning rate is provided or relative step is set to true")]
    LrIsNoneAndRelativeStepIsFalse,
}

/// Error with building a dropout layer
#[derive(Debug, Clone, PartialEq, Error)]
pub enum DropoutBuildError {
    /// Dropout probability must be in the range [0, 1)
    #[error("Dropout probability must be in the range [0, 1)")]
    InvalidProbability,
}

/// Error with building a MultiHeadAttention module
#[derive(Debug, PartialEq, Error)]
pub enum MultiHeadAttentionBuildError {
    /// Invalid number of heads
    #[error("Invalid number of heads: {0}")]
    InvalidNumHeads(i32),

    /// Exceptions
    #[error(transparent)]
    Exception(#[from] Exception),
}

/// Error with building a transformer
#[derive(Debug, PartialEq, Error)]
pub enum TransformerBulidError {
    /// Dropout probability must be in the range [0, 1)
    #[error("Dropout probability must be in the range [0, 1)")]
    InvalidProbability,

    /// Invalid number of heads
    #[error("Invalid number of heads: {0}")]
    InvalidNumHeads(i32),

    /// Exceptions
    #[error(transparent)]
    Exception(#[from] Exception),
}

impl From<DropoutBuildError> for TransformerBulidError {
    fn from(e: DropoutBuildError) -> Self {
        match e {
            DropoutBuildError::InvalidProbability => Self::InvalidProbability,
        }
    }
}

impl From<MultiHeadAttentionBuildError> for TransformerBulidError {
    fn from(e: MultiHeadAttentionBuildError) -> Self {
        match e {
            MultiHeadAttentionBuildError::InvalidNumHeads(n) => Self::InvalidNumHeads(n),
            MultiHeadAttentionBuildError::Exception(e) => Self::Exception(e),
        }
    }
}

/// The dtype is not a float-point type
#[derive(Debug, Error)]
#[error("[finfo] dtype {:?} is not inexact", .0)]
pub struct InexactDtypeError(pub Dtype);

impl From<InexactDtypeError> for Exception {
    #[track_caller]
    fn from(value: InexactDtypeError) -> Self {
        Exception::custom(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use super::{default_mlx_error_handler, get_and_clear_last_mlx_error};
    use crate::array;

    fn report(msg: &str) {
        let msg = CString::new(msg).unwrap();
        default_mlx_error_handler(msg.as_ptr(), std::ptr::null_mut());
    }

    #[test]
    fn undrained_error_is_replaced_by_the_newest() {
        let _ = get_and_clear_last_mlx_error();
        report("first");
        report("second");

        assert_eq!(get_and_clear_last_mlx_error().unwrap().what, "second");
        assert!(get_and_clear_last_mlx_error().is_none());
    }

    #[test]
    fn errors_are_reported_on_every_thread() {
        let messages: Vec<String> = (0..4)
            .map(|_| {
                std::thread::spawn(|| {
                    let a = array!([1.0, 2.0, 3.0]);
                    let b = array!([4.0, 5.0]);
                    let error = a.add(&b).expect_err("Expected error");
                    // Exit with an undrained message, which the thread-local stash must free.
                    report("left behind");
                    error.what().to_string()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();

        for message in messages {
            assert!(message.contains("Shapes (3) and (2) cannot be broadcast."));
        }
    }

    #[test]
    fn test_exception() {
        let a = array!([1.0, 2.0, 3.0]);
        let b = array!([4.0, 5.0]);

        let result = a.add(&b);
        let error = result.expect_err("Expected error");

        // The full error message would also contain the full path to the original c++ file,
        // so we just check for a substring
        assert!(
            error
                .what()
                .contains("Shapes (3) and (2) cannot be broadcast.")
        )
    }
}
