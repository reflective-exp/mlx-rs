//! `compile_with_state` with a module, and with a module and optimizer, as the state.
//!
//! Each training test runs the same steps eagerly and compiled from identical starting points.
//! The compiled function is built once, so steps after the first replay a cached trace; any
//! parameter or optimizer state not threaded through it makes the compiled run fall behind.

use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParameters, Param},
    nn,
    ops::ones,
    optimizers::{
        AdaDelta, AdaGrad, Adafactor, Adam, AdamW, Adamax, Lion, Optimizer, OptimizerState,
        RmsProp, Sgd, SgdBuilder,
    },
    random::{key, uniform},
    transforms::{compile::compile_with_state, eval_params},
};

const STEPS: usize = 5;

#[derive(Debug, ModuleParameters)]
struct LinearFunctionModel {
    #[param]
    m: Param<Array>,

    #[param]
    b: Param<Array>,
}

impl Module<&Array> for LinearFunctionModel {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        self.m.multiply(x)?.add(&self.b)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

impl LinearFunctionModel {
    /// Models built from the same seed start with identical parameters.
    fn from_seed(seed: u64, shape: &[i32]) -> Self {
        let k = key(seed).unwrap();
        let m = uniform::<_, f32>(-5.0, 5.0, shape, &k).unwrap();
        let b = uniform::<_, f32>(-5.0, 5.0, shape, &k).unwrap();
        Self {
            m: Param::new(m),
            b: Param::new(b),
        }
    }
}

fn loss(model: &mut LinearFunctionModel, x: &Array) -> Result<Array, Exception> {
    model.forward(x)?.square()?.sum(None)
}

fn assert_close(actual: &Array, expected: &Array, what: &str) {
    let close = actual
        .all_close(expected, 1e-4, 1e-5, None)
        .unwrap()
        .item::<bool>();
    assert!(close, "{what}: compiled {actual:?} != eager {expected:?}");
}

/// Train eagerly and compiled for [`STEPS`] steps and check that every loss and the final
/// parameters agree, and that training actually moved the parameters.
fn assert_compiled_training_matches_eager<O>(make_optimizer: impl Fn() -> O)
where
    O: Optimizer + 'static,
{
    let step = |(model, optimizer): &mut (LinearFunctionModel, O), x: &Array| -> Array {
        let mut loss_and_grad = nn::value_and_grad(loss);
        let (loss, grad) = loss_and_grad(model, x).unwrap();
        optimizer.update(model, grad).unwrap();
        loss
    };
    let x = Array::from_slice(&[0.5f32, -1.0, 2.0, 1.5], &[4]);
    let initial = LinearFunctionModel::from_seed(0, &[4]);

    let mut eager = (LinearFunctionModel::from_seed(0, &[4]), make_optimizer());
    let mut compiled_state = (LinearFunctionModel::from_seed(0, &[4]), make_optimizer());
    let mut compiled = compile_with_state(step, None);

    for i in 0..STEPS {
        let eager_loss = step(&mut eager, &x);
        eval_params(eager.0.parameters()).unwrap();

        let compiled_loss = compiled(&mut compiled_state, &x).unwrap();
        eval_params(compiled_state.0.parameters()).unwrap();

        assert_close(&compiled_loss, &eager_loss, &format!("loss at step {i}"));
    }

    assert_close(&compiled_state.0.m, &eager.0.m, "final m");
    assert_close(&compiled_state.0.b, &eager.0.b, "final b");
    assert!(
        !eager
            .0
            .m
            .all_close(&initial.m, None, None, None)
            .unwrap()
            .item::<bool>(),
        "training did not change the parameters, so the comparison proves nothing"
    );
}

#[test]
fn compiled_training_matches_eager_sgd() {
    assert_compiled_training_matches_eager(|| Sgd::new(0.01_f32));
}

#[test]
fn compiled_training_matches_eager_sgd_with_momentum() {
    assert_compiled_training_matches_eager(|| {
        SgdBuilder::new(0.01_f32).momentum(0.9_f32).build().unwrap()
    });
}

#[test]
fn compiled_training_matches_eager_adam() {
    assert_compiled_training_matches_eager(|| Adam::new(0.1_f32));
}

#[test]
fn compiled_training_matches_eager_adamw() {
    assert_compiled_training_matches_eager(|| AdamW::new(0.1_f32));
}

#[test]
fn compiled_training_matches_eager_adamax() {
    assert_compiled_training_matches_eager(|| Adamax::new(0.1_f32));
}

#[test]
fn compiled_training_matches_eager_adagrad() {
    assert_compiled_training_matches_eager(|| AdaGrad::new(0.1_f32));
}

#[test]
fn compiled_training_matches_eager_adadelta() {
    assert_compiled_training_matches_eager(|| AdaDelta::new(1.0_f32).unwrap());
}

#[test]
fn compiled_training_matches_eager_rmsprop() {
    assert_compiled_training_matches_eager(|| RmsProp::new(0.01_f32).unwrap());
}

#[test]
fn compiled_training_matches_eager_lion() {
    assert_compiled_training_matches_eager(|| Lion::new(0.01_f32));
}

#[test]
fn compiled_training_matches_eager_adafactor() {
    assert_compiled_training_matches_eager(|| Adafactor::new().unwrap());
}

#[test]
fn compiled_module_matches_eager_across_calls() {
    let step = |model: &mut LinearFunctionModel, x: &[Array]| -> Vec<Array> {
        let mut loss_and_grad = nn::value_and_grad(loss);
        let (loss, _grad) = loss_and_grad(model, &x[0]).unwrap();
        vec![loss]
    };
    let mut model = LinearFunctionModel::from_seed(1, &[10]);
    let x = vec![ones::<f32>(&[10, 1]).unwrap()];

    let expected = step(&mut model, &x);
    let mut compiled = compile_with_state(step, None);
    for _ in 0..3 {
        let result = compiled(&mut model, &x).unwrap();
        assert_close(&result[0], &expected[0], "loss");
    }
}

#[test]
fn compiled_module_reports_errors() {
    let step = |model: &mut LinearFunctionModel, x: &[Array]| -> Result<Vec<Array>, Exception> {
        let mut loss_and_grad = nn::value_and_grad(loss);
        let (loss, _grad) = loss_and_grad(model, &x[0])?;
        Ok(vec![loss])
    };
    let mut model = LinearFunctionModel::from_seed(2, &[10]);
    let x_ok = vec![ones::<f32>(&[10, 1]).unwrap()];
    let x_err = vec![ones::<f32>(&[1, 2, 3]).unwrap()];

    let expected = step(&mut model, &x_ok).unwrap();
    let mut compiled = compile_with_state(step, None);
    assert_close(
        &compiled(&mut model, &x_ok).unwrap()[0],
        &expected[0],
        "loss",
    );
    assert_close(
        &compiled(&mut model, &x_ok).unwrap()[0],
        &expected[0],
        "loss",
    );

    assert!(step(&mut model, &x_err).is_err());
    assert!(compiled(&mut model, &x_err).is_err());
}

#[test]
fn compiled_module_and_optimizer_report_errors() {
    let step = |(model, optimizer): &mut (LinearFunctionModel, Sgd),
                x: &[Array]|
     -> Result<Vec<Array>, Exception> {
        let mut loss_and_grad = nn::value_and_grad(loss);
        let (loss, grad) = loss_and_grad(model, &x[0])?;
        optimizer.update(model, grad)?;
        Ok(vec![loss])
    };
    let mut state = (LinearFunctionModel::from_seed(3, &[10]), Sgd::new(0.0_f32));
    let x_ok = vec![ones::<f32>(&[10, 1]).unwrap()];
    let x_err = vec![ones::<f32>(&[1, 2, 3]).unwrap()];

    // A zero learning rate keeps the parameters fixed, so every call returns the same loss.
    let expected = step(&mut state, &x_ok).unwrap();
    let mut compiled = compile_with_state(step, None);
    for _ in 0..2 {
        let result = compiled(&mut state, &x_ok).unwrap();
        eval_params(state.0.parameters()).unwrap();
        assert_close(&result[0], &expected[0], "loss");
    }

    assert!(step(&mut state, &x_err).is_err());
    assert!(compiled(&mut state, &x_err).is_err());
}

/// Train two steps, save the optimizer state, load it into a fresh optimizer, and check every key
/// and value survives. Saving goes through the same keyed projection as compiled state.
fn assert_optimizer_state_round_trips<O: Optimizer>(mut optimizer: O, mut fresh: O) {
    let mut model = LinearFunctionModel::from_seed(4, &[4]);
    let x = Array::from_slice(&[0.5f32, -1.0, 2.0, 1.5], &[4]);
    for _ in 0..2 {
        let mut loss_and_grad = nn::value_and_grad(loss);
        let (_, grad) = loss_and_grad(&mut model, &x).unwrap();
        optimizer.update(&mut model, grad).unwrap();
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state.safetensors");
    optimizer.state_mut().save_safetensors(&path).unwrap();
    fresh.state_mut().load_safetensors(&path).unwrap();

    let owned = |state: &mut O::State| -> Vec<(String, Array)> {
        let entries = state.flatten().unwrap();
        entries
            .into_iter()
            .map(|(key, value)| (key.to_string(), value.clone()))
            .collect()
    };
    let saved = owned(optimizer.state_mut());
    let loaded = owned(fresh.state_mut());
    assert!(!saved.is_empty());
    let keys = |entries: &[(String, Array)]| -> Vec<String> {
        entries.iter().map(|(key, _)| key.clone()).collect()
    };
    assert_eq!(keys(&loaded), keys(&saved));
    for ((key, loaded), (_, saved)) in loaded.iter().zip(&saved) {
        assert_close(loaded, saved, key);
    }
}

#[test]
fn optimizer_state_round_trips_adam() {
    assert_optimizer_state_round_trips(Adam::new(0.1_f32), Adam::new(0.1_f32));
}

#[test]
fn optimizer_state_round_trips_sgd_with_momentum() {
    let sgd = || SgdBuilder::new(0.01_f32).momentum(0.9_f32).build().unwrap();
    assert_optimizer_state_round_trips(sgd(), sgd());
}

#[test]
fn optimizer_state_round_trips_adafactor() {
    // Adafactor's state has optional slots, which must keep their keys through a round trip.
    assert_optimizer_state_round_trips(Adafactor::new().unwrap(), Adafactor::new().unwrap());
}
