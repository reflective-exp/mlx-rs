use std::sync::atomic::{AtomicU64, Ordering};
use std::{cell::RefCell, ffi::CStr};

use crate::device::Device;
use crate::error::Result;
use crate::utils::SUCCESS;
use crate::utils::guard::Guarded;

/// Bumped every time something changes what the default stream resolves to:
/// the default device ([`Device::set_default`]) or the device's default stream
/// ([`Stream::set_default`]). Reading it is a cheap atomic load that gates every
/// thread's cached default stream, so either change invalidates the cache on all
/// threads.
pub(crate) static DEFAULT_STREAM_GENERATION: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static SCOPED_DEFAULT_STREAM: RefCell<Option<Stream>> = const { RefCell::new(None) };

    /// The default stream resolved for this thread, tagged with the generation
    /// it was resolved under. A cache hit hands out a clone instead of
    /// re-resolving the default device and stream through several FFI calls on
    /// every operation.
    static CACHED_DEFAULT_STREAM: RefCell<Option<(u64, Stream)>> = const { RefCell::new(None) };
}

/// The stream set by the innermost enclosing [`with_stream`] or [`with_device`] on this thread,
/// if any.
///
/// Most code should not need this: operations without an explicit stream already use it.
pub fn thread_local_default_stream() -> Option<Stream> {
    SCOPED_DEFAULT_STREAM.with_borrow(|s| s.clone())
}

/// Run `f` with `stream` as the default for operations not given one.
///
/// The scope is per thread and synchronous: it does not reach work `f` hands to other threads or
/// follow an async task across `.await`. Scopes nest, and the previous stream is restored even if
/// `f` panics.
///
/// ```rust
/// use mlx_rs::{Array, Stream, with_stream};
///
/// let a = Array::from_slice(&[1.0f32, 2.0], &[2]);
/// let doubled = with_stream(&Stream::cpu(), || &a * 2.0);
/// assert_eq!(doubled.as_slice::<f32>(), &[2.0, 4.0]);
/// ```
pub fn with_stream<F, T>(stream: &Stream, f: F) -> T
where
    F: FnOnce() -> T,
{
    crate::utils::with_scoped_local(&SCOPED_DEFAULT_STREAM, stream.clone(), f)
}

/// Run `f` with `device`'s default stream as the default for operations not given one. See
/// [`with_stream`] for the scope's extent.
///
/// # Panics
///
/// Panics if MLX has no default stream for `device`, for example a GPU on a machine without one.
pub fn with_device<F, T>(device: &Device, f: F) -> T
where
    F: FnOnce() -> T,
{
    let stream = Stream::try_default_on_device(device)
        .unwrap_or_else(|error| panic!("no default stream for {device:?}: {}", error.what()));
    with_stream(&stream, f)
}

/// Renamed to [`thread_local_default_stream`]: the value is thread-local, not task-local.
#[deprecated(note = "use `thread_local_default_stream`")]
pub fn task_local_default_stream() -> Option<Stream> {
    thread_local_default_stream()
}

/// Use a given default stream for the duration of the closure `f`.
#[deprecated(note = "use `with_stream(&stream, f)`")]
pub fn with_new_default_stream<F, T>(default_stream: Stream, f: F) -> T
where
    F: FnOnce() -> T,
{
    with_stream(&default_stream, f)
}

/// Parameter type for all MLX operations.
///
/// Use this to control where operations are evaluated:
///
/// If omitted it will use the [Default::default()], which will be [Device::gpu()] unless
/// set otherwise.
#[derive(PartialEq)]
pub struct StreamOrDevice {
    pub(crate) stream: Stream,
}

impl StreamOrDevice {
    /// Create a new [`StreamOrDevice`] with a [`Stream`].
    pub fn new(stream: Stream) -> StreamOrDevice {
        StreamOrDevice { stream }
    }

    /// Create a new [`StreamOrDevice`] with a [`Device`].
    pub fn new_with_device(device: &Device) -> StreamOrDevice {
        StreamOrDevice {
            stream: Stream::new_with_device(device),
        }
    }

    /// Current default CPU stream.
    pub fn cpu() -> StreamOrDevice {
        StreamOrDevice {
            stream: Stream::cpu(),
        }
    }

    /// Current default GPU stream.
    pub fn gpu() -> StreamOrDevice {
        StreamOrDevice {
            stream: Stream::gpu(),
        }
    }
}

impl Default for StreamOrDevice {
    /// The stream set by an enclosing [`with_stream`] or [`with_device`] if any, otherwise the
    /// default stream on the default device.
    ///
    /// Without a scope this is [Device::gpu()] unless changed by [Device::set_default()].
    fn default() -> Self {
        Self {
            stream: Stream::thread_local_or_default(),
        }
    }
}

impl AsRef<Stream> for StreamOrDevice {
    fn as_ref(&self) -> &Stream {
        &self.stream
    }
}

impl std::fmt::Debug for StreamOrDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.stream)
    }
}

impl std::fmt::Display for StreamOrDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.stream)
    }
}

/// A stream of evaluation attached to a particular device.
///
/// Typically, this is used via the `stream:` parameter on a method with a [StreamOrDevice]:
pub struct Stream {
    pub(crate) c_stream: mlx_sys::mlx_stream,
}

impl AsRef<Stream> for Stream {
    fn as_ref(&self) -> &Stream {
        self
    }
}

impl Clone for Stream {
    fn clone(&self) -> Self {
        Stream::try_from_op(|res| unsafe { mlx_sys::mlx_stream_set(res, self.c_stream) })
            .expect("Failed to clone stream")
    }
}

impl Stream {
    /// The stream set by an enclosing [`with_stream`] or [`with_device`], or else the default
    /// stream on the default device.
    pub fn thread_local_or_default() -> Self {
        thread_local_default_stream().unwrap_or_default()
    }

    /// The stream set by an enclosing [`with_stream`] or [`with_device`], or else the default
    /// cpu stream.
    pub fn thread_local_or_cpu() -> Self {
        thread_local_default_stream().unwrap_or_else(Stream::cpu)
    }

    /// The stream set by an enclosing [`with_stream`] or [`with_device`], or else the default
    /// gpu stream.
    pub fn thread_local_or_gpu() -> Self {
        thread_local_default_stream().unwrap_or_else(Stream::gpu)
    }

    /// Renamed to [`Stream::thread_local_or_default`].
    #[deprecated(note = "use `Stream::thread_local_or_default`")]
    pub fn task_local_or_default() -> Self {
        Self::thread_local_or_default()
    }

    /// Renamed to [`Stream::thread_local_or_cpu`].
    #[deprecated(note = "use `Stream::thread_local_or_cpu`")]
    pub fn task_local_or_cpu() -> Self {
        Self::thread_local_or_cpu()
    }

    /// Renamed to [`Stream::thread_local_or_gpu`].
    #[deprecated(note = "use `Stream::thread_local_or_gpu`")]
    pub fn task_local_or_gpu() -> Self {
        Self::thread_local_or_gpu()
    }

    /// Create a new stream on the default device. Panics if fails.
    ///
    /// The default stream is cached per thread and only re-resolved when the
    /// default device changes, so repeated calls avoid the FFI round-trip
    /// through [`Self::resolve_default`].
    pub fn new() -> Stream {
        let generation = DEFAULT_STREAM_GENERATION.load(Ordering::Relaxed);
        CACHED_DEFAULT_STREAM.with_borrow_mut(|cache| {
            if let Some((cached_generation, stream)) = cache
                && *cached_generation == generation
            {
                return stream.clone();
            }
            let stream = Self::resolve_default();
            *cache = Some((generation, stream.clone()));
            stream
        })
    }

    /// Resolve the default stream on the default device through FFI, bypassing
    /// the per-thread cache.
    fn resolve_default() -> Stream {
        unsafe {
            let mut dev = mlx_sys::mlx_device_new();
            // SAFETY: mlx_get_default_device internally never throws an error
            mlx_sys::mlx_get_default_device(&mut dev as *mut _);

            let mut c_stream = mlx_sys::mlx_stream_new();
            // SAFETY: mlx_get_default_stream internally never throws if dev is valid
            mlx_sys::mlx_get_default_stream(&mut c_stream as *mut _, dev);

            mlx_sys::mlx_device_free(dev);
            Stream { c_stream }
        }
    }

    /// Try to get the default stream on the given device.
    pub fn try_default_on_device(device: &Device) -> Result<Stream> {
        Stream::try_from_op(|res| unsafe { mlx_sys::mlx_get_default_stream(res, device.c_device) })
    }

    /// Create a new stream on the given device
    pub fn new_with_device(device: &Device) -> Stream {
        unsafe {
            let c_stream = mlx_sys::mlx_stream_new_device(device.c_device);
            Stream { c_stream }
        }
    }

    /// Get the underlying C pointer.
    pub fn as_ptr(&self) -> mlx_sys::mlx_stream {
        self.c_stream
    }

    /// Current default CPU stream.
    pub fn cpu() -> Self {
        unsafe {
            let c_stream = mlx_sys::mlx_default_cpu_stream_new();
            Stream { c_stream }
        }
    }

    /// Current default GPU stream.
    pub fn gpu() -> Self {
        unsafe {
            let c_stream = mlx_sys::mlx_default_gpu_stream_new();
            Stream { c_stream }
        }
    }

    /// Get the index of the stream.
    pub fn get_index(&self) -> Result<i32> {
        i32::try_from_op(|res| unsafe { mlx_sys::mlx_stream_get_index(res, self.c_stream) })
    }

    /// Get the device this stream runs on.
    pub fn device(&self) -> Result<Device> {
        Device::try_from_op(|res| unsafe { mlx_sys::mlx_stream_get_device(res, self.c_stream) })
    }

    /// Make this the default stream for its device.
    ///
    /// The device itself is unchanged; use [`Device::set_default`] to change which device the
    /// default stream is resolved on.
    pub fn set_default(stream: &Stream) -> Result<()> {
        <()>::try_from_op(|_res| unsafe { mlx_sys::mlx_set_default_stream(stream.c_stream) })?;
        // The default stream for the current default device may have changed, so every
        // thread's cached default stream is now stale.
        DEFAULT_STREAM_GENERATION.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn describe(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        unsafe {
            let mut mlx_str = mlx_sys::mlx_string_new();
            let result = match mlx_sys::mlx_stream_tostring(&mut mlx_str as *mut _, self.c_stream) {
                SUCCESS => {
                    let ptr = mlx_sys::mlx_string_data(mlx_str);
                    let c_str = CStr::from_ptr(ptr);
                    write!(f, "{}", c_str.to_string_lossy())
                }
                _ => Err(std::fmt::Error),
            };
            mlx_sys::mlx_string_free(mlx_str);
            result
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_stream_free(self.c_stream) };
    }
}

impl Default for Stream {
    fn default() -> Self {
        Stream::new()
    }
}

impl std::fmt::Debug for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.describe(f)
    }
}

impl std::fmt::Display for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.describe(f)
    }
}

impl PartialEq for Stream {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlx_sys::mlx_stream_equal(self.c_stream, other.c_stream) }
    }
}

/// Block until all work previously submitted to `stream` has finished.
///
/// MLX evaluates lazily and asynchronously, so this is only needed when the completion of the
/// work matters on its own — timing a benchmark, or observing a side effect outside MLX.
/// Reading array data (for example with `as_slice`) already synchronizes.
pub fn synchronize(stream: impl AsRef<Stream>) -> Result<()> {
    <()>::try_from_op(|_res| unsafe { mlx_sys::mlx_synchronize(stream.as_ref().c_stream) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scoped_default_stream() {
        // First set default stream to CPU
        let cpu_device = Device::cpu();
        Device::set_default(&cpu_device);
        let cpu_stream = Stream::default();

        let scoped = Stream::gpu();
        with_stream(&scoped, || {
            let first = Stream::thread_local_or_default();
            let second = Stream::thread_local_or_default();
            assert_eq!(first, second);
            assert_ne!(first, cpu_stream);
        });
    }

    #[test]
    fn scoped_default_streams_nest_and_restore_after_panic() {
        let outer = Stream::cpu();
        with_stream(&outer, || {
            let panic = std::panic::catch_unwind(|| {
                with_stream(&Stream::gpu(), || {
                    assert_eq!(thread_local_default_stream(), Some(Stream::gpu()));
                    panic!("stream scope panic");
                })
            });

            assert!(panic.is_err());
            assert_eq!(thread_local_default_stream(), Some(outer.clone()));
        });

        assert!(thread_local_default_stream().is_none());
    }

    #[test]
    fn with_device_scopes_that_devices_default_stream() {
        with_device(&Device::cpu(), || {
            assert_eq!(thread_local_default_stream(), Some(Stream::cpu()));
        });
        assert!(thread_local_default_stream().is_none());
    }

    #[test]
    fn scoped_stream_does_not_reach_other_threads() {
        with_stream(&Stream::cpu(), || {
            let other = std::thread::spawn(|| thread_local_default_stream().is_none());
            assert!(other.join().unwrap());
        });
    }

    #[test]
    #[allow(deprecated)]
    fn deprecated_names_still_scope_the_stream() {
        with_new_default_stream(Stream::cpu(), || {
            assert_eq!(task_local_default_stream(), Some(Stream::cpu()));
            assert_eq!(Stream::task_local_or_gpu(), Stream::cpu());
        });
    }

    #[test]
    fn test_stream_or_device_default_uses_scoped_stream() {
        Device::set_default(&Device::cpu());
        let cpu_default = StreamOrDevice::default();

        let scoped = Stream::gpu();
        with_stream(&scoped, || {
            assert_eq!(StreamOrDevice::default().as_ref(), &scoped);
            assert_ne!(StreamOrDevice::default(), cpu_default);
        });
    }

    #[test]
    fn test_new_reflects_default_device_change() {
        // Repeated calls under the same default device return the cached stream.
        Device::set_default(&Device::cpu());
        let cpu_stream = Stream::new();
        assert_eq!(cpu_stream, Stream::new());

        // Changing the default device invalidates the cache, so the next call
        // resolves the new device's stream rather than the stale cached one.
        Device::set_default(&Device::gpu());
        let gpu_stream = Stream::new();
        assert_ne!(cpu_stream, gpu_stream);
        assert_eq!(gpu_stream, Stream::new());
    }

    #[test]
    fn test_stream_get_device() {
        assert_eq!(Stream::cpu().device().unwrap(), Device::cpu());
        assert_eq!(Stream::gpu().device().unwrap(), Device::gpu());
    }

    #[test]
    fn test_synchronize() {
        let a = crate::Array::from_slice(&[1.0f32, 2.0], &[2]);
        let b = a.add(&a).unwrap();
        synchronize(Stream::cpu()).unwrap();
        assert_eq!(b.as_slice::<f32>(), &[2.0, 4.0]);
    }

    #[test]
    fn test_set_default_stream() {
        Device::set_default(&Device::cpu());
        let previous = Stream::default();

        // A freshly created stream on the same device has its own index, so making
        // it the default is observable through `Stream::default()`.
        let fresh = Stream::new_with_device(&Device::cpu());
        Stream::set_default(&fresh).unwrap();
        assert_eq!(Stream::default(), fresh);

        Stream::set_default(&previous).unwrap();
        assert_eq!(Stream::default(), previous);
    }

    #[test]
    fn test_stream_clone() {
        let stream = Stream::new();
        let cloned_stream = stream.clone();
        assert_eq!(stream, cloned_stream);
    }

    #[test]
    fn test_cpu_gpu_stream_not_equal() {
        let cpu_device = Device::cpu();
        let gpu_device = Device::gpu();

        // First set default stream to CPU
        Device::set_default(&cpu_device);
        let cpu_stream = Stream::default();

        // Then set default stream to GPU
        Device::set_default(&gpu_device);
        let gpu_stream = Stream::default();

        // Assert that CPU and GPU streams are not equal
        assert_ne!(cpu_stream, gpu_stream);
    }
}
