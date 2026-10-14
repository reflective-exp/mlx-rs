//! Thread-safety claims: which types may cross threads, and that the ones that may still work
//! when they do.

use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::{Arc, Barrier};
use std::thread;

use mlx_rs::transforms::compile::{CallMut, Compile, compile, compile_with_state};
use mlx_rs::{Array, Device, array, with_device};

/// Reports whether a value's type implements `Send` / `Sync`, resolved at compile time but
/// returned as a `bool` so a test can assert either way. This is autoref specialization: the
/// `Yes` traits are implemented for `Probe<T>` only when `T` has the trait, and method lookup
/// tries `Probe<T>` before falling back to the `No` traits on `&Probe<T>`. It works for closure
/// and other unnameable types because the checked type is always concrete at the call site.
struct Probe<T>(PhantomData<T>);

fn probe<T>(_: &T) -> Probe<T> {
    Probe(PhantomData)
}

trait SendYes {
    fn is_send(&self) -> bool {
        true
    }
}
impl<T: Send> SendYes for Probe<T> {}

trait SendNo {
    fn is_send(&self) -> bool {
        false
    }
}
impl<T> SendNo for &Probe<T> {}

trait SyncYes {
    fn is_sync(&self) -> bool {
        true
    }
}
impl<T: Sync> SyncYes for Probe<T> {}

trait SyncNo {
    fn is_sync(&self) -> bool {
        false
    }
}
impl<T> SyncNo for &Probe<T> {}

macro_rules! is_send {
    ($value:expr) => {
        (&probe(&$value)).is_send()
    };
}

macro_rules! is_sync {
    ($value:expr) => {
        (&probe(&$value)).is_sync()
    };
}

fn double(x: &Array) -> Array {
    x * 2.0
}

#[test]
fn probe_reports_known_types_correctly() {
    assert!(is_send!(vec![1.0f32]));
    assert!(is_sync!(vec![1.0f32]));
    assert!(!is_send!(Rc::new(1)));
    assert!(!is_sync!(Rc::new(1)));
    assert!(is_send!(std::cell::Cell::new(1)));
    assert!(!is_sync!(std::cell::Cell::new(1)));
}

#[test]
fn compiled_functions_are_thread_bound() {
    let compiled = compile(double, None);
    assert!(!is_send!(compiled));
    assert!(!is_sync!(compiled));

    let compiled = Compile::compile(double, false);
    assert!(!is_send!(compiled));
    assert!(!is_sync!(compiled));

    let step = |state: &mut Vec<Array>, x: &Array| -> Array { &state[0] + x };
    let compiled = compile_with_state(step, None);
    assert!(!is_send!(compiled));
    assert!(!is_sync!(compiled));
}

#[test]
fn each_thread_compiles_and_runs_its_own_function() {
    let threads = 4;
    let barrier = Arc::new(Barrier::new(threads));
    let handles: Vec<_> = (0..threads)
        .map(|i| {
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                let k = i as f32 + 1.0;
                let mut compiled = Compile::compile(move |x: &Array| x * k, false);
                let x = array!([1.0f32, 2.0]);
                // Compile and run on every thread at once.
                barrier.wait();
                for _ in 0..3 {
                    let y: Array = compiled.call_mut(&x).unwrap();
                    assert_eq!(y.as_slice::<f32>(), &[k, 2.0 * k]);
                }
                // Dropped on the thread that compiled it, so it erases its own cache entry.
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
}

// `Array` is `Send` by an `unsafe impl`. An evaluated array can be read from any thread. An
// unevaluated one cannot be evaluated off the thread that built it: MLX streams belong to the
// thread that created them, so evaluation elsewhere fails. These check that it returns an error
// rather than crashing or reading garbage.

fn assert_no_stream_error(result: Result<Vec<f32>, mlx_rs::error::AsSliceError>) {
    let error = result.expect_err("evaluated a graph on a thread that does not own its stream");
    assert!(
        error.to_string().contains("in current thread"),
        "unexpected error: {error}"
    );
}

#[test]
fn array_is_send_but_not_sync() {
    let a = array!(1.0f32);
    assert!(is_send!(a));
    assert!(!is_sync!(a));
}

#[test]
fn evaluated_array_can_be_read_on_another_thread() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
    a.eval().unwrap();
    let read = thread::spawn(move || a.as_slice::<f32>().to_vec());
    assert_eq!(read.join().unwrap(), [1.0, 2.0, 3.0]);
}

#[test]
fn unevaluated_graph_errors_on_another_thread() {
    for device in [Device::cpu(), Device::gpu()] {
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let lazy = with_device(&device, || &a * 2.0 + 1.0);
        let lazy_on_other_thread = lazy.clone();
        let read = thread::spawn(move || {
            lazy_on_other_thread
                .try_as_slice::<f32>()
                .map(<[f32]>::to_vec)
        });
        assert_no_stream_error(read.join().unwrap());

        // The failed attempt leaves the graph intact for the thread that owns its stream.
        assert_eq!(lazy.as_slice::<f32>(), &[3.0, 5.0, 7.0]);
    }
}

#[test]
fn evaluated_array_can_be_read_from_several_threads_at_once() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
    let doubled = &a * 2.0;
    doubled.eval().unwrap();

    let barrier = Arc::new(Barrier::new(4));
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let doubled = doubled.clone();
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                // New work on this thread, built from an array evaluated elsewhere.
                let tripled = &doubled * 1.5;
                (
                    doubled.as_slice::<f32>().to_vec(),
                    tripled.as_slice::<f32>().to_vec(),
                )
            })
        })
        .collect();
    for handle in handles {
        let (doubled, tripled) = handle.join().unwrap();
        assert_eq!(doubled, [2.0, 4.0, 6.0]);
        assert_eq!(tripled, [3.0, 6.0, 9.0]);
    }
}
