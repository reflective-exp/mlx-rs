# Changelog

## 0156f661d615b6a3f05d9e01f1aab06bfad8e05a

- `rms_norm`/`rms_norm_device` weight parameter is now optional; when omitted,
  the underlying kernel skips the per-element multiply instead of requiring a
  unit-weight array.
- `StreamOrDevice::default()` now resolves through `Stream::task_local_or_default()`,
  so code inside `with_new_default_stream` picks up the task-local stream
  instead of always falling back to the per-thread device default.
