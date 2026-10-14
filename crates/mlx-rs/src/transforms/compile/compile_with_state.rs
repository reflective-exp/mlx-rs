//! Compilation of functions with state.
//!
//! The state is threaded through the compiled graph as extra inputs and outputs. Which arrays
//! those are, and in what order, comes from the state's [`Updatable::state_projection`]: inputs
//! are its present arrays in key order, and outputs are written back by key.

// TODO: there's plenty boilerplate code here but it's not clear how to reduce it

use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::Array;
use crate::error::Exception;
use crate::transforms::compile::CompiledState;
use crate::utils::{StateLayoutEntry, StateSnapshot, Updatable};

use super::{Closure, Compiled, Guarded, VectorArray, erase_compiled};

/// Similar to [`crate::transforms::compile`] but allows for functions that take
/// a mutable reference to a state `U`.
pub fn compile_with_state<F, U, A, O, E>(
    f: F,
    shapeless: impl Into<Option<bool>>,
) -> impl for<'a> FnMut(&mut U, F::Args<'a>) -> Result<O, Exception>
where
    F: CompileWithState<U, A, O, E> + 'static,
    U: Updatable,
{
    let shapeless = shapeless.into().unwrap_or(false);
    // Built once so every call reuses one cache entry, which is erased when the closure drops.
    let mut compiled = f.compile(shapeless);
    move |state, args| compiled.call_mut(state, args)
}

/// A trait for functions that can be compiled with state.
///
/// This trait is used to compile a function that takes a mutable reference to a state
/// and some arguments and returns a result.
///
/// # Generic parameters
///
/// - `U`: The type of the state.
/// - `A`: The type of the arguments.
/// - `O`: The type of the output.
/// - `E`: The type of the exception.
pub trait CompileWithState<U, A, O, E> {
    /// The type of the arguments that the returned closure takes.
    ///
    /// This is needed to relax the lifetime requirements of the returned
    /// closure. Otherwise, the arguments to the returned closure would have to
    /// live longer than the closure itself.
    type Args<'a>;

    /// Compile the function.
    fn compile(
        self,
        shapeless: bool,
    ) -> impl for<'args> CallMutWithState<U, Self::Args<'args>, O, E>;
}

impl<F, U> CompileWithState<U, &[Array], Vec<Array>, ()> for F
where
    F: FnMut(&mut U, &[Array]) -> Vec<Array> + 'static,
    U: Updatable,
{
    type Args<'a> = &'a [Array];

    fn compile(
        self,
        shapeless: bool,
    ) -> impl for<'args> CallMutWithState<U, Self::Args<'args>, Vec<Array>, ()> {
        let state = CompiledState::new(self, shapeless);
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F, U> CompileWithState<U, &Array, Array, ()> for F
where
    F: FnMut(&mut U, &Array) -> Array + 'static,
    U: Updatable,
{
    type Args<'a> = &'a Array;

    fn compile(
        mut self,
        shapeless: bool,
    ) -> impl for<'args> CallMutWithState<U, Self::Args<'args>, Array, ()> {
        let f = move |state: &mut U, args: &[Array]| -> Vec<Array> {
            let result = (self)(state, &args[0]);
            vec![result]
        };
        let state = CompiledState::new(f, shapeless);
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F, U> CompileWithState<U, (&Array, &Array), Array, ()> for F
where
    F: FnMut(&mut U, (&Array, &Array)) -> Array + 'static,
    U: Updatable,
{
    type Args<'a> = (&'a Array, &'a Array);

    fn compile(
        mut self,
        shapeless: bool,
    ) -> impl for<'args> CallMutWithState<U, Self::Args<'args>, Array, ()> {
        let f = move |state: &mut U, args: &[Array]| -> Vec<Array> {
            let result = (self)(state, (&args[0], &args[1]));
            vec![result]
        };
        let state = CompiledState::new(f, shapeless);
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F, U> CompileWithState<U, (&Array, &Array, &Array), Array, ()> for F
where
    F: FnMut(&mut U, (&Array, &Array, &Array)) -> Array + 'static,
    U: Updatable,
{
    type Args<'a> = (&'a Array, &'a Array, &'a Array);

    fn compile(
        mut self,
        shapeless: bool,
    ) -> impl for<'args> CallMutWithState<U, Self::Args<'args>, Array, ()> {
        let f = move |state: &mut U, args: &[Array]| -> Vec<Array> {
            let result = (self)(state, (&args[0], &args[1], &args[2]));
            vec![result]
        };
        let state = CompiledState::new(f, shapeless);
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F, U> CompileWithState<U, &[Array], Vec<Array>, Exception> for F
where
    F: FnMut(&mut U, &[Array]) -> Result<Vec<Array>, Exception> + 'static,
    U: Updatable,
{
    type Args<'a> = &'a [Array];

    fn compile(
        self,
        shapeless: bool,
    ) -> impl for<'args> CallMutWithState<U, Self::Args<'args>, Vec<Array>, Exception> {
        let state = CompiledState::new(self, shapeless);
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F, U> CompileWithState<U, &Array, Array, Exception> for F
where
    F: FnMut(&mut U, &Array) -> Result<Array, Exception> + 'static,
    U: Updatable,
{
    type Args<'a> = &'a Array;

    fn compile(
        mut self,
        shapeless: bool,
    ) -> impl for<'args> CallMutWithState<U, Self::Args<'args>, Array, Exception> {
        let f = move |state: &mut U, args: &[Array]| -> Result<Vec<Array>, Exception> {
            let result = (self)(state, &args[0])?;
            Ok(vec![result])
        };
        let state = CompiledState::new(f, shapeless);
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F, U> CompileWithState<U, (&Array, &Array), Array, Exception> for F
where
    F: FnMut(&mut U, (&Array, &Array)) -> Result<Array, Exception> + 'static,
    U: Updatable,
{
    type Args<'a> = (&'a Array, &'a Array);

    fn compile(
        mut self,
        shapeless: bool,
    ) -> impl for<'args> CallMutWithState<U, Self::Args<'args>, Array, Exception> {
        let f = move |state: &mut U, args: &[Array]| -> Result<Vec<Array>, Exception> {
            let result = (self)(state, (&args[0], &args[1]))?;
            Ok(vec![result])
        };
        let state = CompiledState::new(f, shapeless);
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F, U> CompileWithState<U, (&Array, &Array, &Array), Array, Exception> for F
where
    F: FnMut(&mut U, (&Array, &Array, &Array)) -> Result<Array, Exception> + 'static,
    U: Updatable,
{
    type Args<'a> = (&'a Array, &'a Array, &'a Array);

    fn compile(
        mut self,
        shapeless: bool,
    ) -> impl for<'args> CallMutWithState<U, Self::Args<'args>, Array, Exception> {
        let f = move |state: &mut U, args: &[Array]| -> Result<Vec<Array>, Exception> {
            let result = (self)(state, (&args[0], &args[1], &args[2]))?;
            Ok(vec![result])
        };
        let state = CompiledState::new(f, shapeless);
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

/// A trait for functions that can be called with state.
pub trait CallMutWithState<U, A, O, E> {
    /// Call the function with the given state and arguments.
    fn call_mut(&mut self, state: &mut U, args: A) -> Result<O, Exception>;
}

impl<U, F, G> CallMutWithState<U, &[Array], Vec<Array>, ()> for Compiled<F, G>
where
    F: FnMut(&mut U, &[Array]) -> Vec<Array>,
    G: FnMut(&mut U, &[Array]) -> Vec<Array>,
    U: Updatable,
{
    fn call_mut(&mut self, state: &mut U, args: &[Array]) -> Result<Vec<Array>, Exception> {
        self.state.call_mut_with_state(state, args)
    }
}

impl<U, F, G> CallMutWithState<U, &Array, Array, ()> for Compiled<F, G>
where
    F: FnMut(&mut U, &Array) -> Array,
    G: FnMut(&mut U, &[Array]) -> Vec<Array>,
    U: Updatable,
{
    fn call_mut(&mut self, state: &mut U, args: &Array) -> Result<Array, Exception> {
        let args = std::slice::from_ref(args);
        let result = self.state.call_mut_with_state(state, args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<U, F, G> CallMutWithState<U, (&Array, &Array), Array, ()> for Compiled<F, G>
where
    F: FnMut(&mut U, (&Array, &Array)) -> Array,
    G: FnMut(&mut U, &[Array]) -> Vec<Array>,
    U: Updatable,
{
    fn call_mut(&mut self, state: &mut U, args: (&Array, &Array)) -> Result<Array, Exception> {
        let args = &[args.0, args.1];
        let result = self.state.call_mut_with_state(state, args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<U, F, G> CallMutWithState<U, (&Array, &Array, &Array), Array, ()> for Compiled<F, G>
where
    F: FnMut(&mut U, (&Array, &Array, &Array)) -> Array,
    G: FnMut(&mut U, &[Array]) -> Vec<Array>,
    U: Updatable,
{
    fn call_mut(
        &mut self,
        state: &mut U,
        args: (&Array, &Array, &Array),
    ) -> Result<Array, Exception> {
        let args = &[args.0, args.1, args.2];
        let result = self.state.call_mut_with_state(state, args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<U, F, G> CallMutWithState<U, &[Array], Vec<Array>, Exception> for Compiled<F, G>
where
    F: FnMut(&mut U, &[Array]) -> Result<Vec<Array>, Exception>,
    G: FnMut(&mut U, &[Array]) -> Result<Vec<Array>, Exception>,
    U: Updatable,
{
    fn call_mut(&mut self, state: &mut U, args: &[Array]) -> Result<Vec<Array>, Exception> {
        self.state.fallible_call_mut_with_state(state, args)
    }
}

impl<U, F, G> CallMutWithState<U, &Array, Array, Exception> for Compiled<F, G>
where
    F: FnMut(&mut U, &Array) -> Result<Array, Exception>,
    G: FnMut(&mut U, &[Array]) -> Result<Vec<Array>, Exception>,
    U: Updatable,
{
    fn call_mut(&mut self, state: &mut U, args: &Array) -> Result<Array, Exception> {
        let args = std::slice::from_ref(args);
        let result = self.state.fallible_call_mut_with_state(state, args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<U, F, G> CallMutWithState<U, (&Array, &Array), Array, Exception> for Compiled<F, G>
where
    F: FnMut(&mut U, (&Array, &Array)) -> Result<Array, Exception>,
    G: FnMut(&mut U, &[Array]) -> Result<Vec<Array>, Exception>,
    U: Updatable,
{
    fn call_mut(&mut self, state: &mut U, args: (&Array, &Array)) -> Result<Array, Exception> {
        let args = &[args.0, args.1];
        let result = self.state.fallible_call_mut_with_state(state, args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<U, F, G> CallMutWithState<U, (&Array, &Array, &Array), Array, Exception> for Compiled<F, G>
where
    F: FnMut(&mut U, (&Array, &Array, &Array)) -> Result<Array, Exception>,
    G: FnMut(&mut U, &[Array]) -> Result<Vec<Array>, Exception>,
    U: Updatable,
{
    fn call_mut(
        &mut self,
        state: &mut U,
        args: (&Array, &Array, &Array),
    ) -> Result<Array, Exception> {
        let args = &[args.0, args.1, args.2];
        let result = self.state.fallible_call_mut_with_state(state, args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

/// What tracing a `compile_with_state` function recorded, needed to replay the cached graph.
#[derive(Debug, Clone)]
pub(super) struct StateTrace {
    /// The state layout the graph takes as its trailing inputs.
    input_layout: Vec<StateLayoutEntry>,
    /// The state layout the function left behind: the graph's trailing outputs, in this order.
    output_layout: Vec<StateLayoutEntry>,
    /// How many leading outputs are the function's own results rather than state.
    num_function_outputs: usize,
}

type Step<'f, U> = dyn FnMut(&mut U, &[Array]) -> Result<Vec<Array>, Exception> + 'f;

impl<F> CompiledState<F> {
    fn call_mut_with_state<U>(
        &mut self,
        state: &mut U,
        args: &[impl AsRef<Array>],
    ) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&mut U, &[Array]) -> Vec<Array>,
        U: Updatable,
    {
        let CompiledState {
            f,
            shapeless,
            id,
            state_trace,
            ..
        } = self;
        let mut step = |state: &mut U, args: &[Array]| Ok(f(state, args));
        call_with_state(*id, *shapeless, state_trace, &mut step, state, args)
    }

    fn fallible_call_mut_with_state<U>(
        &mut self,
        state: &mut U,
        args: &[impl AsRef<Array>],
    ) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&mut U, &[Array]) -> Result<Vec<Array>, Exception>,
        U: Updatable,
    {
        let CompiledState {
            f,
            shapeless,
            id,
            state_trace,
            ..
        } = self;
        call_with_state(*id, *shapeless, state_trace, f, state, args)
    }
}

/// Run the compiled function `id` on `args` and `state`, writing the new state back by key.
///
/// If the call fails, `state` is restored to its value before the call.
fn call_with_state<U: Updatable>(
    id: usize,
    shapeless: bool,
    trace: &mut Option<StateTrace>,
    step: &mut Step<'_, U>,
    state: &mut U,
    args: &[impl AsRef<Array>],
) -> Result<Vec<Array>, Exception> {
    let before = state.state_projection()?.snapshot();

    // The cached graph takes a fixed state layout as input. If the layout has changed since it was
    // traced -- an optimizer creates its state on the first step, or a parameter was added -- the
    // graph would read the wrong inputs, so drop it and trace again.
    if trace
        .as_ref()
        .is_some_and(|trace| trace.input_layout != before.layout())
    {
        erase_compiled(id);
        *trace = None;
    }

    let fresh_trace = trace.is_none();
    let mut result = apply_with_state(id, shapeless, trace, step, state, args, &before);

    // MLX's compiler fails when a traced function returns more than one array it created from
    // scratch rather than computed from its inputs, and state an optimizer creates on its first
    // step (zeros) is exactly that. The failed trace left the new slots in place, zeroed, so trace
    // once more with them as inputs: they are then computed from inputs like any other state.
    if result.is_err() && fresh_trace {
        let grown = state.state_projection()?.snapshot();
        if grown.len() > before.len() {
            erase_compiled(id);
            *trace = None;
            result = apply_with_state(id, shapeless, trace, step, state, args, &grown);
        }
    }

    if let Err(error) = &result {
        // Tracing may have stopped partway, so trace afresh next time.
        erase_compiled(id);
        *trace = None;
        // Slots created by a failed trace did not exist before the call; they are kept, zeroed.
        if let Err(restore_error) = state.state_projection()?.restore(before, true) {
            return Err(Exception::custom(format!(
                "{}; restoring the state afterwards also failed: {restore_error}",
                error.what()
            )));
        }
    }
    result
}

fn apply_with_state<U: Updatable>(
    id: usize,
    shapeless: bool,
    trace: &mut Option<StateTrace>,
    step: &mut Step<'_, U>,
    state: &mut U,
    args: &[impl AsRef<Array>],
    before: &StateSnapshot,
) -> Result<Vec<Array>, Exception> {
    let args_len = args.len();
    let state = Rc::new(RefCell::new(state));
    let recorded: Rc<RefCell<Option<StateTrace>>> = Rc::default();

    // MLX calls this only when it traces: the state slots are swapped for the tracers, the
    // function runs, and the real values are put back.
    let inner = {
        let state = Rc::clone(&state);
        let recorded = Rc::clone(&recorded);
        move |tracers: &[Array]| -> Result<Vec<Array>, Exception> {
            let mut state = state.borrow_mut();
            let inputs = state.state_projection()?.snapshot();
            let input_layout = inputs.layout();
            let (tracer_args, tracer_state) =
                tracers.split_at_checked(args_len).ok_or_else(|| {
                    Exception::custom("compile_with_state: fewer traced inputs than arguments")
                })?;
            let tracer_state = StateSnapshot::from_layout_and_values(&input_layout, tracer_state)?;
            state.state_projection()?.restore(tracer_state, false)?;

            let traced = step(&mut **state, tracer_args);
            let outputs = state.state_projection()?.snapshot();
            // Put the real values back even if the function failed, so no tracer leaks into the
            // caller's state. Slots the function created are left in place but zeroed;
            // the graph's outputs fill them in.
            state.state_projection()?.restore(inputs, true)?;

            let mut results = traced?;
            *recorded.borrow_mut() = Some(StateTrace {
                input_layout,
                output_layout: outputs.layout(),
                num_function_outputs: results.len(),
            });
            results.extend(outputs.present_values().cloned());
            Ok(results)
        }
    };
    let inner = Closure::new_fallible(inner);

    // Reuses the cached graph for `id` if the inputs match; traces otherwise.
    let compiled = Closure::try_from_op(|res| unsafe {
        let constants = &[];
        mlx_sys::mlx_detail_compile(res, inner.as_ptr(), id, shapeless, constants.as_ptr(), 0)
    })?;
    let inputs = VectorArray::try_from_iter(
        args.iter()
            .map(AsRef::as_ref)
            .chain(before.present_values()),
    )?;
    let outputs: Vec<Array> = VectorArray::try_from_op(|res| unsafe {
        mlx_sys::mlx_closure_apply(res, compiled.as_ptr(), inputs.as_ptr())
    })?
    .try_into_values()?;

    if let Some(recorded) = recorded.take() {
        *trace = Some(recorded);
    }
    let trace = trace.as_ref().ok_or_else(|| {
        Exception::custom("compile_with_state: the compiled function ran without a trace")
    })?;

    let num_state_outputs = trace
        .output_layout
        .iter()
        .filter(|entry| entry.is_present())
        .count();
    if outputs.len() != trace.num_function_outputs + num_state_outputs {
        return Err(Exception::custom(format!(
            "compile_with_state: expected {} function outputs and {num_state_outputs} state \
             outputs, got {} outputs",
            trace.num_function_outputs,
            outputs.len()
        )));
    }
    let (results, state_outputs) = outputs.split_at(trace.num_function_outputs);
    let new_state = StateSnapshot::from_layout_and_values(&trace.output_layout, state_outputs)?;
    state
        .borrow_mut()
        .state_projection()?
        .restore(new_state, false)?;

    Ok(results.to_vec())
}

#[cfg(test)]
mod tests {
    use crate::{
        Array, array,
        error::Exception,
        random::{RandomState, uniform},
    };

    use super::compile_with_state;

    /// Returned closures share one type and differ only in captures.
    fn offset_by(k: f32) -> impl Fn(&mut RandomState, &Array) -> Array + Copy + 'static {
        move |_state: &mut RandomState, x: &Array| x + k
    }

    #[test]
    fn same_type_closures_with_different_captures_compile_separately() {
        let x = array!([1.0f32, 2.0]);
        let mut state = RandomState::with_seed(0).unwrap();

        let mut plus_one = compile_with_state(offset_by(1.0), None);
        let mut plus_two = compile_with_state(offset_by(2.0), None);
        assert_eq!(
            plus_one(&mut state, &x).unwrap().as_slice::<f32>(),
            &[2.0, 3.0]
        );
        assert_eq!(
            plus_two(&mut state, &x).unwrap().as_slice::<f32>(),
            &[3.0, 4.0]
        );
        assert_eq!(
            plus_one(&mut state, &x).unwrap().as_slice::<f32>(),
            &[2.0, 3.0]
        );
    }

    #[test]
    fn state_advances_across_calls_to_one_compiled_function() {
        let draw = |state: &mut RandomState, _x: &Array| -> Array {
            let key = state.next_key().unwrap();
            uniform::<_, f32>(0.0, 1.0, &[4][..], &key).unwrap()
        };
        let x = array!(0.0f32);

        let mut eager_state = RandomState::with_seed(7).unwrap();
        let eager_first = draw(&mut eager_state, &x);
        let eager_second = draw(&mut eager_state, &x);

        let mut state = RandomState::with_seed(7).unwrap();
        let mut compiled = compile_with_state(draw, None);
        let first = compiled(&mut state, &x).unwrap();
        let second = compiled(&mut state, &x).unwrap();

        // The compiled graph can differ from eager evaluation in the last ulp.
        let close = |a: &Array, b: &Array| a.all_close(b, None, None, None).unwrap().item::<bool>();
        assert!(close(&first, &eager_first));
        assert!(close(&second, &eager_second));
        assert!(!close(&first, &second));
    }

    #[test]
    fn state_that_changes_layout_between_calls_is_retraced() {
        // Each call scales every state array by 2 and returns their sum.
        let step = |state: &mut Vec<Array>, _x: &[Array]| -> Vec<Array> {
            for value in state.iter_mut() {
                *value = &*value * 2.0;
            }
            let total = state.iter().fold(array!(0.0f32), |sum, value| sum + value);
            vec![total]
        };
        let mut compiled = compile_with_state(step, None);
        let mut state = vec![array!(1.0f32)];

        assert_eq!(compiled(&mut state, &[]).unwrap()[0].item::<f32>(), 2.0);
        assert_eq!(compiled(&mut state, &[]).unwrap()[0].item::<f32>(), 4.0);

        // A new slot changes the layout the cached graph was traced with.
        state.push(array!(10.0f32));
        assert_eq!(compiled(&mut state, &[]).unwrap()[0].item::<f32>(), 28.0);
        assert_eq!(state[0].item::<f32>(), 8.0);
        assert_eq!(state[1].item::<f32>(), 20.0);
    }

    #[test]
    fn state_grown_by_the_first_call_is_written_back_and_then_traced() {
        // Like an optimizer, the step creates its state on first use.
        let step = |state: &mut Vec<Array>, x: &[Array]| -> Vec<Array> {
            if state.is_empty() {
                state.push(array!(0.0f32));
            }
            state[0] = &state[0] + &x[0];
            vec![state[0].clone()]
        };
        let mut compiled = compile_with_state(step, None);
        let mut state = Vec::new();
        let x = [array!(3.0f32)];

        for expected in [3.0, 6.0, 9.0] {
            let result = compiled(&mut state, &x).unwrap();
            assert_eq!(result[0].item::<f32>(), expected);
            assert_eq!(state[0].item::<f32>(), expected);
        }
    }

    #[test]
    fn several_slots_created_by_the_first_call_are_traced() {
        // MLX cannot return two arrays the function created from scratch, which is what two fresh
        // zero slots are on the first trace.
        let step = |state: &mut Vec<Array>, x: &[Array]| -> Vec<Array> {
            if state.is_empty() {
                state.push(array!(0.0f32));
                state.push(array!(0.0f32));
            }
            state[0] = &state[0] + &x[0];
            vec![&state[0] + &state[1]]
        };
        let mut compiled = compile_with_state(step, None);
        let mut state = Vec::new();
        let x = [array!(2.0f32)];

        for expected in [2.0, 4.0, 6.0] {
            assert_eq!(compiled(&mut state, &x).unwrap()[0].item::<f32>(), expected);
        }
        assert_eq!(state[0].item::<f32>(), 6.0);
        assert_eq!(state[1].item::<f32>(), 0.0);
    }

    #[test]
    fn failed_call_leaves_state_unchanged() {
        let step = |state: &mut Vec<Array>, x: &[Array]| -> Result<Vec<Array>, Exception> {
            state[0] = &state[0] + 1.0;
            let sum = state[0].add(&x[0])?;
            Ok(vec![sum])
        };
        let mut compiled = compile_with_state(step, None);
        let mut state = vec![array!([1.0f32, 2.0])];

        // A shape that does not broadcast fails while tracing.
        let bad = [array!([1.0f32, 2.0, 3.0])];
        assert!(compiled(&mut state, &bad).is_err());
        assert_eq!(state[0].as_slice::<f32>(), &[1.0, 2.0]);

        let good = [array!([0.5f32, 0.5])];
        let result = compiled(&mut state, &good).unwrap();
        assert_eq!(result[0].as_slice::<f32>(), &[2.5, 3.5]);
        assert_eq!(state[0].as_slice::<f32>(), &[2.0, 3.0]);
    }
}
