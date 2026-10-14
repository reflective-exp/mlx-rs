# Changelog

## Unreleased

Breaking: one function per op, streams chosen by scope. Every `foo_device(.., stream)` twin is
gone, as are the generated op macros (`exp!`, `sum!`, `normal!`, ...) and the internal
`#[default_device]`/`#[generate_macro]` attributes. Call `foo(..)` and wrap it in
`with_stream(&stream, || ..)` or `with_device(&device, || ..)` to pick where it runs; ops that
only run on the CPU (`multivariate_normal`, file loading) still default to it. The `_device`
indexing trait methods (`try_index_device`, `index_mut_device`, ...), `MetalKernel::apply_device`
and `GgufFile::load_device` went the same way. `nn::NonLinearity` takes just the array; its
stream argument was always `Stream::default()`.

Breaking renames, with no aliases. Each changes the signature, so old calls fail to compile
instead of changing meaning:

- `concatenate_axis(arrays, axis)` is `concatenate(arrays, axis)`; the old flattening
  `concatenate(arrays)` is `concatenate_flat`.
- `stack_axis(arrays, axis)` is `stack(arrays, axis)`; the old axis-0 `stack(arrays)` is gone.
- `split(a, n, axis)` is `split_equal`; `split_sections` / `Array::split_axis` is
  `split_at_indices`.
- `r#where` and `which` are `select`.

Breaking: ops with several independent settings take an options struct, and `Axes`
(`All`/`Axis`/`Axes`, from `i32`, slices or vecs) selects axes for the new ones:

- `trace(a, TraceOptions)`, `logcumsumexp(a, LogCumsumExpOptions)`,
  `linspace(start, stop, LinspaceOptions)` (now honours `endpoint: false`).
- `count_nonzero(a, CountNonzeroOptions)` replaces `count_nonzero`/`count_nonzero_axis`. The
  whole-array form now calls MLX directly, so with `keep_dims` it keeps the input's rank.
- `contiguous(a, ContiguousOptions)`; `Array::contiguous()` plus `contiguous_with_options`.
- `linalg::{norm, norm_matrix, norm_l2}` take `NormOptions { axes, keep_dims }`.
- `fft::{fftn, ifftn, rfftn, irfftn, fft2, ifft2, rfft2, irfft2}` take
  `FftnOptions { lengths, axes, norm }`; for the 2-D ones `Axes::All` means the last two axes.
  Lengths must match the selected axes, and with `Axes::All` in an n-D transform there must be
  one per dimension; mismatches, out-of-range axes and empty real
  transforms return errors instead of reaching MLX or panicking.

Added:

- `io::GgufFile`: load and save GGUF files with tensors and typed metadata (arrays, strings,
  string lists). Q4_0, Q4_1 and Q8_0 tensors load as quantized triplets. `mlx-sys` now links
  MLX's `gguflib`.
- `ops::search_sorted` with `SearchSide`.
- `ops::indexing::TryIndexUpdateOp` and `UpdateMode::{Replace, Add, Min, Max, Product}`:
  functional updates through the same indices as `index_mut`, returning a new array.
- `with_stream(&stream, f)` and `with_device(&device, f)`: run a block of operations on a stream
  or a device's default stream. `with_new_default_stream` is deprecated in favour of
  `with_stream`.

Compiled functions (from `compile`, `compile_with_state`, or `Compile::compile`) are now
`!Send` and `!Sync`. MLX keeps one compile cache per thread, so one used or dropped on another
thread traced into, or erased from, the wrong cache.

Renamed: the scoped default stream is per thread, not per async task, so `task_local_*` is now
`thread_local_*` (`thread_local_default_stream`, `Stream::thread_local_or_{default,cpu,gpu}`).
The old names remain as deprecated aliases.

Breaking: keyed state. `Updatable` is now one method, `state_projection`, which declares every
mutable array under a key (`utils::StateProjection`, with required and optional slots) instead of
three positional `updatable_states*` methods that had to agree on order. Optimizers implement it
through `OptimizerState::state_projection`, which also replaces `OptimizerState::flatten`/
`flatten_mut` as the thing to implement; those two, and `save_safetensors`, now take `&mut self`
and return a `Result`. Saving, loading and `compile_with_state` all derive from the one
declaration.

`compile_with_state` writes state back by key and checks the output count exactly, instead of
zipping outputs onto state by position. A call whose state layout differs from the cached
graph's (an optimizer's first step, an added parameter) retraces instead of misreading inputs,
and a failed call restores the state. The blind retry after a failed call is gone; its one case,
an optimizer's first step creating several zero slots that MLX's compiler cannot return from a
trace, now retraces deliberately.

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

A panic inside a closure passed to a transform (`grad`, `value_and_grad`, `jvp`, `vjp`,
`compile`, `compile_with_state`) propagates to the caller with its original payload instead of
aborting the process at the FFI boundary.

`with_new_default_stream` and `with_random_state` restore the previous stream or random state
when their closure panics, instead of leaving the scoped value in place for the rest of the
thread.

An MLX call that fails without recording a message returns an error naming its status instead
of panicking, and failures no longer report a stale message left by an earlier call.
`save_safetensors` no longer leaks its array and metadata maps on error.

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
