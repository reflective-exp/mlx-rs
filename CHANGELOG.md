# Changelog

## Unreleased

Safe wrappers for further mlx-c entry points that had no mlx-rs binding:

- `cuda::is_available`, and `metal::{is_available, metallib_path, set_metallib_path,
  start_capture, stop_capture}`.
- `Device::{count, info, is_available}` and the `DeviceInfo` accessors, which expose
  the backend-reported device properties.
- `Stream::{device, set_default}` and `stream::synchronize`.
- `graph_utils::{export_to_dot, print_graph}` and `NodeNamer`, which render the
  unevaluated computation graph as text or Graphviz DOT.
- `fft::{fftfreq, rfftfreq}`, `linalg::{det, slogdet}`, and `ops::{copy, depends}`.
- `random::{bits, laplace, normal_broadcast, permutation, permutation_arange}`.
- `ops::indexing::{scatter, scatter_add, scatter_add_axis, scatter_max, scatter_min,
  scatter_prod}`, the multi-index counterparts of the existing `*_single` forms.
- `ops::indexing::{slice_dynamic, slice_update, slice_update_add, slice_update_dynamic,
  slice_update_max, slice_update_min, slice_update_prod}`.

`slice_dynamic` and `slice_update_dynamic` clamp their starting indices to the bounds of
the input. MLX computes the offset for a dynamic slice without any bounds check, so an
out-of-range start otherwise reads or writes past the array and segfaults; MLX leaves that
case undefined, so clamping does not change any behavior it defines.

## 0156f661d615b6a3f05d9e01f1aab06bfad8e05a

- `rms_norm`/`rms_norm_device` weight parameter is now optional; when omitted,
  the underlying kernel skips the per-element multiply instead of requiring a
  unit-weight array.
- `StreamOrDevice::default()` now resolves through `Stream::task_local_or_default()`,
  so code inside `with_new_default_stream` picks up the task-local stream
  instead of always falling back to the per-thread device default.
