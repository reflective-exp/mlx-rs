# Changelog

## Unreleased

MLX is now 0.32.2, built from upstream `ml-explore/mlx-c` (`c74db530`) instead of a fork.
- `with_stream(&stream, f)` and `with_device(&device, f)` run a block of ops on a stream or
  device.
- `io::GgufFile` loads and saves GGUF files, including typed metadata. Q4_0, Q4_1 and Q8_0
  tensors load as quantized triplets.
- `ops::indexing::TryIndexUpdateOp` with `UpdateMode::{Replace, Add, Min, Max, Product}`:
  functional updates through `index_mut`-style indices, returning a new array.
- Ops: `fast::cross_entropy`, `gather_qqmm`, `search_sorted`, `zeros_like_dtype`,
  `ones_like_dtype`, `copy`, `depends`, `fft::{fftfreq, rfftfreq}`, `linalg::{det, slogdet}`,
  `random::{bits, laplace, normal_broadcast, permutation, permutation_arange}`.
- Indexing: multi-index `scatter*` (`scatter`, `scatter_add`, `scatter_add_axis`,
  `scatter_max`, `scatter_min`, `scatter_prod`), `slice_dynamic` and `slice_update*`
  (`slice_update`, `_add`, `_dynamic`, `_max`, `_min`, `_prod`). Dynamic slices clamp their
  start to the array bounds instead of accessing out of bounds.
- Devices and streams: `Device::{count, info, is_available}`, `DeviceInfo`,
  `Stream::{device, set_default}`, `stream::synchronize`, `cuda::is_available`,
  `metal::{is_available, metallib_path, set_metallib_path, start_capture, stop_capture}`.
- `graph_utils::{export_to_dot, print_graph}` and `NodeNamer` print an unevaluated graph as text
  or Graphviz DOT.
- `qqmm` now builds on macOS.

### Breaking changes

- **One function per op.** The `*_device(.., stream)` variants and generated op macros
  (`exp!`, `sum!`, ...) are gone, including the `_device` indexing methods,
  `MetalKernel::apply_device` and `GgufFile::load_device`. Call `foo(..)` inside
  `with_stream`/`with_device` instead. `nn::NonLinearity` takes only the array.
- **Renames** (old calls fail to compile rather than change meaning):
  - `concatenate_axis` → `concatenate`; the old flattening `concatenate` → `concatenate_flat`.
  - `stack_axis` → `stack`; the old axis-0 `stack` is removed.
  - `split` → `split_equal`; `split_sections`/`Array::split_axis` → `split_at_indices`.
  - `r#where` and `which` → `select`.
- **Options structs**, with axes as `Axes::{All, Axis, Axes}`:
  `trace`, `logcumsumexp`, `linspace` (now honours `endpoint: false`), `count_nonzero`
  (replaces `count_nonzero_axis`), `contiguous`, `qqmm`, `linalg::{norm, norm_matrix, norm_l2}`
  and the n-D/2-D `fft` functions. Bad FFT lengths or axes return errors instead of panicking.
- **Keyed state.** `Updatable` is a single method, `state_projection`, declaring each mutable
  array under a key, replacing the three order-dependent `updatable_states*` methods.
  Optimizers implement `OptimizerState::state_projection`; `flatten`, `flatten_mut` and
  `save_safetensors` now take `&mut self` and return a `Result`.
- Compiled functions are `!Send` and `!Sync`, since MLX's compile cache is per thread. They no
  longer require `F: Copy`.
- `try_as_slice` returns `AsSliceError::NotContiguous` for strided views, and `as_slice` panics;
  call `Array::contiguous()` first. Previously these misordered or overran the buffer.

### Deprecated

- `with_new_default_stream`: use `with_stream`.
- `task_local_*` stream functions: renamed to `thread_local_*`, since the scope is per thread.

### Fixed

- `compile_with_state` matches state by key, retraces when the state layout changes (such as an
  optimizer's first step) and restores state on failure.
- Same-type closures with different captures no longer share a compiled graph, and a compiled
  function reuses its trace instead of retracing on every call.
- Adafactor's `step` is carried through `compile_with_state` instead of frozen.
- Panics inside `grad`, `value_and_grad`, `jvp`, `vjp` and `compile` closures reach the caller
  instead of aborting. `with_new_default_stream` and `with_random_state` restore their previous
  value on panic.
- MLX failures without a message return an error instead of panicking, and no longer report a
  stale message from an earlier call.
- Memory leaks in the error handler and safetensors load/save.
- `deep_clone` copies logical elements of strided arrays.

## 0156f661d615b6a3f05d9e01f1aab06bfad8e05a

- `rms_norm`/`rms_norm_device` weight parameter is now optional; when omitted,
  the underlying kernel skips the per-element multiply instead of requiring a
  unit-weight array.
- `StreamOrDevice::default()` now resolves through `Stream::task_local_or_default()`,
  so code inside `with_new_default_stream` picks up the task-local stream
  instead of always falling back to the per-thread device default.
