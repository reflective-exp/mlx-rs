use std::ffi::{CStr, CString};

use crate::error::{Result, ensure_error_handler, last_mlx_error_or};
use crate::utils::SUCCESS;
use crate::utils::guard::Guarded;

///Type of device.
#[derive(num_enum::IntoPrimitive, Debug, Clone, Copy)]
#[repr(u32)]
pub enum DeviceType {
    /// CPU device
    Cpu = mlx_sys::mlx_device_type__MLX_CPU,

    /// GPU device
    Gpu = mlx_sys::mlx_device_type__MLX_GPU,
}

/// Representation of a Device in MLX.
pub struct Device {
    pub(crate) c_device: mlx_sys::mlx_device,
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlx_sys::mlx_device_equal(self.c_device, other.c_device) }
    }
}

impl Device {
    /// Create a new [`Device`]
    pub fn new(device_type: DeviceType, index: i32) -> Device {
        let c_device = unsafe { mlx_sys::mlx_device_new_type(device_type.into(), index) };
        Device { c_device }
    }

    /// Try to get the default device.
    pub fn try_default() -> Result<Self> {
        Device::try_from_op(|res| unsafe { mlx_sys::mlx_get_default_device(res) })
    }

    /// Create a default CPU device.
    pub fn cpu() -> Device {
        Device::new(DeviceType::Cpu, 0)
    }

    /// Create a default GPU device.
    pub fn gpu() -> Device {
        Device::new(DeviceType::Gpu, 0)
    }

    /// Get the device index
    pub fn get_index(&self) -> Result<i32> {
        i32::try_from_op(|res| unsafe { mlx_sys::mlx_device_get_index(res, self.c_device) })
    }

    /// Get the device type
    pub fn get_type(&self) -> Result<DeviceType> {
        DeviceType::try_from_op(|res| unsafe { mlx_sys::mlx_device_get_type(res, self.c_device) })
    }

    /// Whether this device is available to run operations on.
    pub fn is_available(&self) -> Result<bool> {
        bool::try_from_op(|res| unsafe { mlx_sys::mlx_device_is_available(res, self.c_device) })
    }

    /// The number of devices of the given type.
    pub fn count(device_type: DeviceType) -> Result<i32> {
        i32::try_from_op(|res| unsafe { mlx_sys::mlx_device_count(res, device_type.into()) })
    }

    /// Query the backend-reported properties of this device.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Device;
    ///
    /// let info = Device::gpu().info().unwrap();
    /// assert!(!info.get_string("device_name").unwrap().is_empty());
    /// ```
    pub fn info(&self) -> Result<DeviceInfo> {
        ensure_error_handler();
        let mut c_info = unsafe { mlx_sys::mlx_device_info_new() };
        let status = unsafe { mlx_sys::mlx_device_info_get(&mut c_info as *mut _, self.c_device) };
        if status != SUCCESS {
            unsafe { mlx_sys::mlx_device_info_free(c_info) };
            return Err(last_mlx_error_or("Failed to get device info"));
        }
        Ok(DeviceInfo { c_info })
    }

    /// Set the default device.
    ///
    /// # Example:
    ///
    /// ```rust
    /// use mlx_rs::{Device, DeviceType};
    /// Device::set_default(&Device::new(DeviceType::Cpu, 1));
    /// ```
    ///
    /// By default, this is `gpu()`.
    pub fn set_default(device: &Device) {
        unsafe { mlx_sys::mlx_set_default_device(device.c_device) };
        // Invalidate every thread's cached default stream, since the default
        // stream now resolves to this device.
        crate::stream::DEFAULT_STREAM_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn describe(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        unsafe {
            let mut mlx_str = mlx_sys::mlx_string_new();
            let result = match mlx_sys::mlx_device_tostring(&mut mlx_str as *mut _, self.c_device) {
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

impl Drop for Device {
    fn drop(&mut self) {
        let status = unsafe { mlx_sys::mlx_device_free(self.c_device) };
        debug_assert_eq!(status, SUCCESS);
    }
}

/// Backend-reported properties of a [`Device`], obtained from [`Device::info`].
///
/// The available keys vary by backend. The Metal backend reports `architecture` and `device_name`
/// as strings, and `max_buffer_length`, `max_recommended_working_set_size`, `memory_size` and
/// `resource_limit` as sizes.
pub struct DeviceInfo {
    c_info: mlx_sys::mlx_device_info,
}

/// Borrow `key` as a C string, rejecting the interior nul that MLX cannot represent.
fn c_key(key: &str) -> Result<CString> {
    CString::new(key).map_err(|_| crate::error::Exception::from("Invalid key"))
}

impl DeviceInfo {
    /// Every key reported for the device.
    pub fn keys(&self) -> Result<Vec<String>> {
        ensure_error_handler();
        unsafe {
            let mut c_keys = mlx_sys::mlx_vector_string_new();
            let status = mlx_sys::mlx_device_info_get_keys(&mut c_keys as *mut _, self.c_info);
            if status != SUCCESS {
                mlx_sys::mlx_vector_string_free(c_keys);
                return Err(last_mlx_error_or("Failed to get device info keys"));
            }

            let len = mlx_sys::mlx_vector_string_size(c_keys);
            let mut keys = Vec::with_capacity(len);
            for i in 0..len {
                let mut ptr: *mut std::os::raw::c_char = std::ptr::null_mut();
                let status = mlx_sys::mlx_vector_string_get(&mut ptr as *mut _, c_keys, i);
                if status != SUCCESS {
                    mlx_sys::mlx_vector_string_free(c_keys);
                    return Err(last_mlx_error_or("Failed to read device info key"));
                }
                keys.push(CStr::from_ptr(ptr).to_string_lossy().into_owned());
            }

            mlx_sys::mlx_vector_string_free(c_keys);
            Ok(keys)
        }
    }

    /// Whether the device reports the given key.
    pub fn has_key(&self, key: &str) -> Result<bool> {
        let key = c_key(key)?;
        bool::try_from_op(|res| unsafe {
            mlx_sys::mlx_device_info_has_key(res, self.c_info, key.as_ptr())
        })
    }

    /// Whether the value for `key` is a string rather than a size.
    ///
    /// Returns an error if the key is not present.
    pub fn is_string(&self, key: &str) -> Result<bool> {
        ensure_error_handler();
        let key = c_key(key)?;
        unsafe {
            let mut is_string = false;
            let status = mlx_sys::mlx_device_info_is_string(
                &mut is_string as *mut _,
                self.c_info,
                key.as_ptr(),
            );
            if status != SUCCESS {
                return Err(last_mlx_error_or("Device info key is missing"));
            }
            Ok(is_string)
        }
    }

    /// The string value for `key`.
    ///
    /// Returns an error if the key is missing or holds a size.
    pub fn get_string(&self, key: &str) -> Result<String> {
        ensure_error_handler();
        let key = c_key(key)?;
        unsafe {
            let mut ptr: *const std::os::raw::c_char = std::ptr::null();
            let status =
                mlx_sys::mlx_device_info_get_string(&mut ptr as *mut _, self.c_info, key.as_ptr());
            if status != SUCCESS {
                return Err(last_mlx_error_or(
                    "Device info key is missing or is not a string",
                ));
            }
            Ok(CStr::from_ptr(ptr).to_string_lossy().into_owned())
        }
    }

    /// The size value for `key`.
    ///
    /// Returns an error if the key is missing or holds a string.
    pub fn get_size(&self, key: &str) -> Result<usize> {
        ensure_error_handler();
        let key = c_key(key)?;
        unsafe {
            let mut value = 0usize;
            let status =
                mlx_sys::mlx_device_info_get_size(&mut value as *mut _, self.c_info, key.as_ptr());
            if status != SUCCESS {
                return Err(last_mlx_error_or(
                    "Device info key is missing or is not a size",
                ));
            }
            Ok(value)
        }
    }
}

impl Drop for DeviceInfo {
    fn drop(&mut self) {
        let status = unsafe { mlx_sys::mlx_device_info_free(self.c_info) };
        debug_assert_eq!(status, SUCCESS);
    }
}

impl std::fmt::Debug for DeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut map = f.debug_map();
        if let Ok(keys) = self.keys() {
            for key in keys {
                match self.is_string(&key) {
                    Ok(true) => map.entry(&key, &self.get_string(&key).ok()),
                    Ok(false) => map.entry(&key, &self.get_size(&key).ok()),
                    Err(_) => map.entry(&key, &"<error>"),
                };
            }
        }
        map.finish()
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::try_default().unwrap()
    }
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.describe(f)
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.describe(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fmt() {
        let device = Device::default();
        let description = format!("{device}");
        assert_eq!(description, "Device(gpu, 0)");
    }

    #[test]
    fn test_is_available() {
        assert!(Device::cpu().is_available().unwrap());
    }

    #[test]
    fn test_count() {
        assert!(Device::count(DeviceType::Cpu).unwrap() >= 1);
    }

    #[test]
    fn test_info_keys() {
        let info = Device::gpu().info().unwrap();
        let keys = info.keys().unwrap();

        assert!(keys.contains(&"architecture".to_string()));
        assert!(keys.contains(&"device_name".to_string()));
        assert!(keys.contains(&"memory_size".to_string()));
    }

    #[test]
    fn test_info_string_value() {
        let info = Device::gpu().info().unwrap();

        assert!(info.has_key("device_name").unwrap());
        assert!(info.is_string("device_name").unwrap());
        assert!(!info.get_string("device_name").unwrap().is_empty());
    }

    #[test]
    fn test_info_size_value() {
        let info = Device::gpu().info().unwrap();

        assert!(info.has_key("memory_size").unwrap());
        assert!(!info.is_string("memory_size").unwrap());
        assert!(info.get_size("memory_size").unwrap() > 0);
    }

    #[test]
    fn test_info_missing_key() {
        let info = Device::gpu().info().unwrap();

        assert!(!info.has_key("not_a_real_key").unwrap());
        assert!(info.get_string("not_a_real_key").is_err());
        assert!(info.get_size("not_a_real_key").is_err());

        // Reading a size as a string (and vice versa) is an error, not a coercion
        assert!(info.get_size("device_name").is_err());
        assert!(info.get_string("memory_size").is_err());
    }
}
