# Changelog

## Unreleased

The mlx-c submodule now tracks upstream `ml-explore/mlx-c` at `c74db530` (MLX 0.32.2) instead
of a fork. `as_type`/`as_dtype`/`astype` keep their `copy` flag; upstream `mlx_astype` dropped
it, so a forced same-dtype copy uses `mlx_copy`.

`try_as_slice` returns `AsSliceError::NotContiguous` for strided views (broadcasts, transposes,
multi-axis FFT output) instead of reading the underlying buffer in the wrong order or past its
end; `as_slice` panics. Call `Array::contiguous()` first. Converting such an array to a
safetensors `TensorView` returns `ConversionError::ArraySlice`, and `deep_clone` now evaluates
the array and copies its logical elements.

Each compiled function gets its own cache id instead of one derived from the closure type, so
same-type closures with different captures no longer share a traced graph. `compile` and
`compile_with_state` build the compiled function once instead of per call, so the trace is reused
rather than erased and retraced, and no longer require `F: Copy`. `Compile::compile` and
`CompileWithState::compile` return a value usable for any argument lifetime.

Adafactor's `step` is now part of its updatable state, so `compile_with_state` carries it
between steps instead of freezing it and leaving compiled training on the wrong schedule.

The MLX error handler no longer leaks error messages that are overwritten before being read or
left unread at thread exit. It is registered once per process instead of per thread, since
mlx-c stores it in unsynchronized globals.

Loading safetensors no longer leaks map iterators: the metadata iterator was never freed, and the
array iterator leaked on error. An iterator failure with no MLX message now returns an error
instead of panicking.

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
