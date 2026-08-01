//! Rendering the lazily-built computation graph behind an array, for debugging.
//!
//! MLX only evaluates an array when its values are needed, so until then an array carries the
//! graph of primitives that will produce it. [`print_graph`] renders that graph as text and
//! [`export_to_dot`] renders it as Graphviz DOT. Both only show unevaluated work — the graph of an
//! array that has already been evaluated is just the array itself.

use std::ffi::{CStr, CString};

use crate::array::Array;
use crate::error::{Exception, Result, ensure_error_handler, last_mlx_error_or};
use crate::utils::SUCCESS;
use crate::utils::VectorArray;
use crate::utils::guard::Guarded;

/// Assigns display names to the arrays in a graph.
///
/// Nodes that have not been named are given sequential names on first use — `A`, `B`, ..., `Z`,
/// `AA`, `AB`, and so on. Use [`NodeNamer::set_name`] to label the nodes you care about before
/// passing the namer to [`export_to_dot`] or [`print_graph`].
#[derive(Debug)]
pub struct NodeNamer {
    c_namer: mlx_sys::mlx_node_namer,
}

impl NodeNamer {
    /// Create a namer with no names assigned.
    pub fn new() -> Self {
        Self {
            c_namer: unsafe { mlx_sys::mlx_node_namer_new() },
        }
    }

    /// Name `array` in the rendered graph.
    pub fn set_name(&mut self, array: impl AsRef<Array>, name: &str) -> Result<()> {
        let name = CString::new(name).map_err(|_| Exception::from("Invalid node name"))?;
        <()>::try_from_op(|_res| unsafe {
            mlx_sys::mlx_node_namer_set_name(self.c_namer, array.as_ref().as_ptr(), name.as_ptr())
        })
    }

    /// The name of `array`, assigning the next sequential name if it has none.
    ///
    /// This takes `&mut self` because reading an unassigned name records it.
    pub fn get_name(&mut self, array: impl AsRef<Array>) -> Result<String> {
        ensure_error_handler();
        unsafe {
            let mut ptr: *const std::os::raw::c_char = std::ptr::null();
            let status = mlx_sys::mlx_node_namer_get_name(
                &mut ptr as *mut _,
                self.c_namer,
                array.as_ref().as_ptr(),
            );
            if status != SUCCESS {
                return Err(last_mlx_error_or("Failed to get node name"));
            }
            Ok(CStr::from_ptr(ptr).to_string_lossy().into_owned())
        }
    }
}

impl Default for NodeNamer {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for NodeNamer {
    fn drop(&mut self) {
        let status = unsafe { mlx_sys::mlx_node_namer_free(self.c_namer) };
        debug_assert_eq!(status, SUCCESS);
    }
}

/// Run `write`, which renders into the `FILE*` it is given, and return what it wrote.
///
/// The C API only writes to a `FILE*`, so this hands it a temporary file and reads the result
/// back rather than exposing the stream to callers.
fn capture_output<F>(write: F) -> Result<String>
where
    F: FnOnce(*mut mlx_sys::FILE) -> i32,
{
    ensure_error_handler();
    unsafe {
        let file = libc::tmpfile();
        if file.is_null() {
            return Err(Exception::from("Failed to open a temporary file"));
        }

        let status = write(file as *mut mlx_sys::FILE);
        if status != SUCCESS {
            libc::fclose(file);
            return Err(last_mlx_error_or("Failed to render the graph"));
        }

        if libc::fflush(file) != 0 {
            libc::fclose(file);
            return Err(Exception::from("Failed to flush the rendered graph"));
        }

        let len = libc::ftell(file);
        if len < 0 {
            libc::fclose(file);
            return Err(Exception::from("Failed to size the rendered graph"));
        }

        libc::rewind(file);
        let mut buf = vec![0u8; len as usize];
        let read = libc::fread(buf.as_mut_ptr() as *mut libc::c_void, 1, buf.len(), file);
        libc::fclose(file);

        buf.truncate(read);
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }
}

/// Run `render` with `namer`, or with a fresh namer that names every node when it is `None`.
fn with_namer<R>(namer: Option<&NodeNamer>, render: impl FnOnce(&NodeNamer) -> R) -> R {
    match namer {
        Some(namer) => render(namer),
        None => render(&NodeNamer::new()),
    }
}

/// Render the graph producing `outputs` as Graphviz DOT.
///
/// # Params
///
/// - `outputs`: the arrays whose graphs to render
/// - `namer`: names for the nodes; `None` names every node sequentially
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, graph_utils::export_to_dot};
///
/// let a = Array::from_slice(&[1.0f32, 2.0], &[2]);
/// let b = a.add(&a).unwrap();
///
/// let dot = export_to_dot([&b], None).unwrap();
/// assert!(dot.starts_with("digraph {"));
/// ```
pub fn export_to_dot<'a>(
    outputs: impl IntoIterator<Item = &'a Array>,
    namer: Option<&NodeNamer>,
) -> Result<String> {
    let outputs = VectorArray::try_from_iter(outputs.into_iter())?;
    with_namer(namer, |namer| {
        capture_output(|file| unsafe {
            mlx_sys::mlx_export_to_dot(file, namer.c_namer, outputs.as_ptr())
        })
    })
}

/// Render the graph producing `outputs` as text: its inputs, its outputs, and one line per
/// primitive in evaluation order.
///
/// # Params
///
/// - `outputs`: the arrays whose graphs to render
/// - `namer`: names for the nodes; `None` names every node sequentially
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, graph_utils::print_graph};
///
/// let a = Array::from_slice(&[1.0f32, 2.0], &[2]);
/// let b = a.add(&a).unwrap();
///
/// let graph = print_graph([&b], None).unwrap();
/// assert!(graph.contains("Add"));
/// ```
pub fn print_graph<'a>(
    outputs: impl IntoIterator<Item = &'a Array>,
    namer: Option<&NodeNamer>,
) -> Result<String> {
    let outputs = VectorArray::try_from_iter(outputs.into_iter())?;
    with_namer(namer, |namer| {
        capture_output(|file| unsafe {
            mlx_sys::mlx_print_graph(file, namer.c_namer, outputs.as_ptr())
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Array;

    fn unevaluated_graph() -> Array {
        let a = Array::from_slice(&[1.0f32, 2.0], &[2]);
        let b = Array::from_slice(&[3.0f32, 4.0], &[2]);
        a.add(&b).unwrap()
    }

    #[test]
    fn test_export_to_dot() {
        let c = unevaluated_graph();
        let dot = export_to_dot([&c], None).unwrap();

        assert!(dot.starts_with("digraph {"));
        assert!(dot.contains("Add"));
        assert!(dot.trim_end().ends_with('}'));
    }

    #[test]
    fn test_print_graph() {
        let c = unevaluated_graph();
        let graph = print_graph([&c], None).unwrap();

        assert!(graph.starts_with("Inputs: "));
        assert!(graph.contains("Outputs: "));
        assert!(graph.contains("Add"));
    }

    #[test]
    fn test_node_namer() {
        let a = Array::from_slice(&[1.0f32, 2.0], &[2]);
        let b = Array::from_slice(&[3.0f32, 4.0], &[2]);
        let c = a.add(&b).unwrap();

        let mut namer = NodeNamer::new();
        namer.set_name(&a, "lhs").unwrap();
        namer.set_name(&b, "rhs").unwrap();

        assert_eq!(namer.get_name(&a).unwrap(), "lhs");
        assert_eq!(namer.get_name(&b).unwrap(), "rhs");

        let graph = print_graph([&c], Some(&namer)).unwrap();
        assert!(graph.contains("lhs"));
        assert!(graph.contains("rhs"));
    }

    #[test]
    fn test_node_namer_assigns_default_names() {
        let a = Array::from_slice(&[1.0f32, 2.0], &[2]);

        // Unnamed nodes get sequential names starting at "A"
        let mut namer = NodeNamer::new();
        assert_eq!(namer.get_name(&a).unwrap(), "A");
    }
}
