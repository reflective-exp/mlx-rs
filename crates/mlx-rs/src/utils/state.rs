//! Keyed declarations of mutable array state.
//!
//! A type declares its mutable arrays once, by key, through [`Updatable::state_projection`].
//! Saving, loading and `compile_with_state` all derive from that one declaration, so they cannot
//! disagree about which arrays exist or how they are ordered.

use std::collections::HashMap;
use std::rc::Rc;

use crate::Array;
use crate::error::StateProjectionError;
use crate::module::ModuleParameters;

#[derive(Debug)]
enum Slot<'a> {
    Required(&'a mut Array),
    Optional(&'a mut Option<Array>),
}

impl<'a> Slot<'a> {
    fn value(&self) -> Option<&Array> {
        match self {
            Self::Required(value) => Some(value),
            Self::Optional(value) => value.as_ref(),
        }
    }

    fn value_mut(&mut self) -> Option<&mut Array> {
        match self {
            Self::Required(value) => Some(value),
            Self::Optional(value) => value.as_mut(),
        }
    }

    fn into_value(self) -> Option<&'a Array> {
        match self {
            Self::Required(value) => Some(value),
            Self::Optional(value) => value.as_ref(),
        }
    }

    fn into_value_mut(self) -> Option<&'a mut Array> {
        match self {
            Self::Required(value) => Some(value),
            Self::Optional(value) => value.as_mut(),
        }
    }

    fn restore(&mut self, key: &str, value: Option<Array>) -> Result<(), StateProjectionError> {
        match (self, value) {
            (Self::Required(target), Some(value)) => **target = value,
            (Self::Required(_), None) => {
                return Err(StateProjectionError::RequiredSlotAbsent(key.to_owned()));
            }
            (Self::Optional(target), value) => **target = value,
        }
        Ok(())
    }

    /// Replace the value with zeros of the same shape and dtype, keeping an optional slot's
    /// presence. The zeros are a fresh array rather than `zeros_like`, so a value that is a trace
    /// placeholder leaves no dependency on it behind.
    fn reset(&mut self) -> Result<(), StateProjectionError> {
        if let Some(value) = self.value_mut() {
            *value = crate::ops::zeros_dtype(value.shape(), value.dtype())?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct Entry<'a> {
    key: Rc<str>,
    slot: Slot<'a>,
}

/// A keyed declaration of a value's mutable arrays.
///
/// Each slot is declared once under a unique key, as either required or optional. Every view of
/// the projection lists slots in sorted key order, and an absent optional slot keeps its key.
#[derive(Debug, Default)]
pub struct StateProjection<'a> {
    entries: Vec<Entry<'a>>,
}

impl<'a> StateProjection<'a> {
    /// An empty projection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare a slot that always holds an array.
    pub fn required(
        &mut self,
        key: impl Into<Rc<str>>,
        value: &'a mut Array,
    ) -> Result<(), StateProjectionError> {
        self.insert(key.into(), Slot::Required(value))
    }

    /// Declare a slot that may be absent. Its key is kept either way.
    pub fn optional(
        &mut self,
        key: impl Into<Rc<str>>,
        value: &'a mut Option<Array>,
    ) -> Result<(), StateProjectionError> {
        self.insert(key.into(), Slot::Optional(value))
    }

    /// Add every slot of `projection`, with `prefix` prepended to its key.
    pub fn extend_prefixed(
        &mut self,
        prefix: &str,
        projection: StateProjection<'a>,
    ) -> Result<(), StateProjectionError> {
        for entry in projection.entries {
            self.insert(Rc::from(format!("{prefix}{}", entry.key)), entry.slot)?;
        }
        Ok(())
    }

    fn insert(&mut self, key: Rc<str>, slot: Slot<'a>) -> Result<(), StateProjectionError> {
        match self.entries.binary_search_by(|entry| entry.key.cmp(&key)) {
            Ok(_) => Err(StateProjectionError::DuplicateKey(key.to_string())),
            Err(index) => {
                self.entries.insert(index, Entry { key, slot });
                Ok(())
            }
        }
    }

    /// The number of declared slots, including absent optional ones.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no slots are declared.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The number of slots that currently hold an array.
    pub fn present_len(&self) -> usize {
        self.values().count()
    }

    /// The present arrays, in key order.
    pub fn values(&self) -> impl Iterator<Item = &Array> {
        self.entries.iter().filter_map(|entry| entry.slot.value())
    }

    /// The present arrays, mutably, in key order.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut Array> + use<'_, 'a> {
        self.entries
            .iter_mut()
            .filter_map(|entry| entry.slot.value_mut())
    }

    /// Every key with its array, or `None` for an absent optional slot, in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, Option<&Array>)> {
        self.entries
            .iter()
            .map(|entry| (entry.key.as_ref(), entry.slot.value()))
    }

    /// The present slots as `(key, array)` pairs, in key order.
    pub fn into_entries(self) -> impl Iterator<Item = (Rc<str>, &'a Array)> {
        self.entries
            .into_iter()
            .filter_map(|entry| entry.slot.into_value().map(|value| (entry.key, value)))
    }

    /// The present slots as `(key, mutable array)` pairs, in key order.
    pub fn into_entries_mut(self) -> impl Iterator<Item = (Rc<str>, &'a mut Array)> {
        self.entries
            .into_iter()
            .filter_map(|entry| entry.slot.into_value_mut().map(|value| (entry.key, value)))
    }

    /// Copy every key and its current value. Arrays are shared handles, so this is cheap.
    pub fn snapshot(&self) -> StateSnapshot {
        StateSnapshot {
            entries: self
                .entries
                .iter()
                .map(|entry| (entry.key.clone(), entry.slot.value().cloned()))
                .collect(),
        }
    }

    /// The key, presence, dtype and shape of every slot.
    pub fn layout(&self) -> Vec<StateLayoutEntry> {
        self.entries
            .iter()
            .map(|entry| StateLayoutEntry::new(entry.key.clone(), entry.slot.value()))
            .collect()
    }

    /// Write `snapshot` back into the slots by key.
    ///
    /// Every key in the snapshot must be declared. A declared key missing from the snapshot is an
    /// error, unless `reset_new` is set: then the slot is treated as created after the snapshot was
    /// taken, and its value (if any) is replaced with zeros of the same shape and dtype.
    pub fn restore(
        &mut self,
        snapshot: StateSnapshot,
        reset_new: bool,
    ) -> Result<(), StateProjectionError> {
        let mut saved: HashMap<_, _> = snapshot.entries.into_iter().collect();
        for entry in &mut self.entries {
            match saved.remove(&entry.key) {
                Some(value) => entry.slot.restore(&entry.key, value)?,
                None if reset_new => entry.slot.reset()?,
                None => return Err(StateProjectionError::MissingKey(entry.key.to_string())),
            }
        }
        if !saved.is_empty() {
            let mut keys: Vec<_> = saved.into_keys().map(|key| key.to_string()).collect();
            keys.sort();
            return Err(StateProjectionError::UnknownKeys(keys));
        }
        Ok(())
    }
}

/// The keys and values of a [`StateProjection`] at one point in time, including which optional
/// slots were absent.
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    entries: Vec<(Rc<str>, Option<Array>)>,
}

impl StateSnapshot {
    /// The number of slots, including absent optional ones.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the snapshot has no slots.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Every key with its array, or `None` for an absent optional slot, in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, Option<&Array>)> {
        self.entries
            .iter()
            .map(|(key, value)| (key.as_ref(), value.as_ref()))
    }

    pub(crate) fn present_values(&self) -> impl Iterator<Item = &Array> {
        self.entries.iter().filter_map(|(_, value)| value.as_ref())
    }

    pub(crate) fn layout(&self) -> Vec<StateLayoutEntry> {
        self.entries
            .iter()
            .map(|(key, value)| StateLayoutEntry::new(key.clone(), value.as_ref()))
            .collect()
    }

    /// Pair `values` with the present slots of `layout`, in order. The number of values must match
    /// the number of present slots exactly.
    pub(crate) fn from_layout_and_values(
        layout: &[StateLayoutEntry],
        values: &[Array],
    ) -> Result<Self, StateProjectionError> {
        let expected = layout.iter().filter(|entry| entry.is_present()).count();
        if values.len() != expected {
            return Err(StateProjectionError::Cardinality {
                expected,
                actual: values.len(),
            });
        }
        let mut values = values.iter();
        let entries = layout
            .iter()
            .map(|entry| {
                let value = entry.is_present().then(|| values.next().unwrap().clone());
                (entry.key.clone(), value)
            })
            .collect();
        Ok(Self { entries })
    }
}

/// The key, presence, dtype and shape of one slot in a [`StateProjection`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateLayoutEntry {
    key: Rc<str>,
    dtype: Option<crate::Dtype>,
    shape: Option<Vec<i32>>,
}

impl StateLayoutEntry {
    fn new(key: Rc<str>, value: Option<&Array>) -> Self {
        Self {
            key,
            dtype: value.map(Array::dtype),
            shape: value.map(|array| array.shape().to_vec()),
        }
    }

    /// The slot's key.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Whether the slot holds an array.
    pub fn is_present(&self) -> bool {
        self.dtype.is_some()
    }

    /// The array's dtype, if present.
    pub fn dtype(&self) -> Option<crate::Dtype> {
        self.dtype
    }

    /// The array's shape, if present.
    pub fn shape(&self) -> Option<&[i32]> {
        self.shape.as_deref()
    }
}

/// A value whose mutable arrays can be threaded through
/// [`compile_with_state`](crate::transforms::compile::compile_with_state).
///
/// Implemented for every [`ModuleParameters`] type, for optimizers, for `Vec<Array>`, and for
/// pairs of `Updatable` values.
pub trait Updatable {
    /// Declare every mutable array, by key.
    fn state_projection(&mut self) -> Result<StateProjection<'_>, StateProjectionError>;
}

impl<T> Updatable for T
where
    T: ModuleParameters,
{
    fn state_projection(&mut self) -> Result<StateProjection<'_>, StateProjectionError> {
        let mut projection = StateProjection::new();
        for (key, value) in self.parameters_mut().flatten() {
            projection.required(key, value)?;
        }
        Ok(projection)
    }
}

impl<T1, T2> Updatable for (T1, T2)
where
    T1: Updatable,
    T2: Updatable,
{
    fn state_projection(&mut self) -> Result<StateProjection<'_>, StateProjectionError> {
        let (first, second) = self;
        let mut projection = StateProjection::new();
        projection.extend_prefixed("0.", first.state_projection()?)?;
        projection.extend_prefixed("1.", second.state_projection()?)?;
        Ok(projection)
    }
}

impl Updatable for Vec<Array> {
    fn state_projection(&mut self) -> Result<StateProjection<'_>, StateProjectionError> {
        let mut projection = StateProjection::new();
        for (index, value) in self.iter_mut().enumerate() {
            projection.required(index.to_string(), value)?;
        }
        Ok(projection)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array;

    #[test]
    fn slots_are_sorted_and_keys_unique() {
        let (mut b, mut a, mut c) = (array!(2.0f32), array!(1.0f32), None);
        let mut projection = StateProjection::new();
        projection.required("b", &mut b).unwrap();
        projection.required("a", &mut a).unwrap();
        projection.optional("c", &mut c).unwrap();

        let keys: Vec<_> = projection.iter().map(|(key, _)| key.to_owned()).collect();
        assert_eq!(keys, ["a", "b", "c"]);
        assert_eq!(projection.len(), 3);
        assert_eq!(projection.present_len(), 2);

        let mut duplicate = array!(0.0f32);
        assert_eq!(
            projection.required("a", &mut duplicate),
            Err(StateProjectionError::DuplicateKey("a".to_owned()))
        );
    }

    #[test]
    fn restore_is_keyed_and_checks_keys() {
        let mut state = vec![array!(1.0f32), array!(2.0f32)];
        let snapshot = state.state_projection().unwrap().snapshot();

        state[0] = array!(10.0f32);
        state[1] = array!(20.0f32);
        state
            .state_projection()
            .unwrap()
            .restore(snapshot, false)
            .unwrap();
        assert_eq!(state[0].item::<f32>(), 1.0);
        assert_eq!(state[1].item::<f32>(), 2.0);

        let mut shorter = vec![array!(0.0f32)];
        let snapshot = shorter.state_projection().unwrap().snapshot();
        assert_eq!(
            state
                .state_projection()
                .unwrap()
                .restore(snapshot.clone(), false),
            Err(StateProjectionError::MissingKey("1".to_owned()))
        );

        // With `reset_new`, a slot the snapshot predates is zeroed instead.
        state
            .state_projection()
            .unwrap()
            .restore(snapshot, true)
            .unwrap();
        assert_eq!(state[1].item::<f32>(), 0.0);

        let mut longer = vec![array!(0.0f32); 3];
        let snapshot = longer.state_projection().unwrap().snapshot();
        assert_eq!(
            state.state_projection().unwrap().restore(snapshot, false),
            Err(StateProjectionError::UnknownKeys(vec!["2".to_owned()]))
        );
    }

    #[test]
    fn values_pair_with_present_slots_only() {
        let (mut a, mut b) = (array!(1.0f32), None::<Array>);
        let mut projection = StateProjection::new();
        projection.required("a", &mut a).unwrap();
        projection.optional("b", &mut b).unwrap();
        let layout = projection.layout();

        assert_eq!(
            StateSnapshot::from_layout_and_values(&layout, &[]).unwrap_err(),
            StateProjectionError::Cardinality {
                expected: 1,
                actual: 0
            }
        );
        let snapshot = StateSnapshot::from_layout_and_values(&layout, &[array!(5.0f32)]).unwrap();
        projection.restore(snapshot, false).unwrap();
        assert_eq!(a.item::<f32>(), 5.0);
        assert!(b.is_none());
    }

    #[test]
    fn pairs_prefix_their_halves() {
        let mut pair = (vec![array!(1.0f32)], vec![array!(2.0f32)]);
        let projection = pair.state_projection().unwrap();
        let keys: Vec<_> = projection.iter().map(|(key, _)| key.to_owned()).collect();
        assert_eq!(keys, ["0.0", "1.0"]);
    }
}
