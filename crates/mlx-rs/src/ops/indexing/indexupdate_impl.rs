use crate::error::Result;
use crate::{Array, Stream};

use super::indexmut_impl::index_update;
use super::{ArrayIndex, ArrayIndexOp, TryIndexUpdateOp, UpdateMode};

impl<'a, Val> TryIndexUpdateOp<&'a [ArrayIndexOp<'a>], Val> for Array
where
    Val: AsRef<Array>,
{
    fn try_index_update_device(
        &self,
        i: &'a [ArrayIndexOp<'a>],
        val: Val,
        mode: UpdateMode,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        index_update(self, i, val.as_ref(), mode, stream)
    }
}

impl<'a, A, Val> TryIndexUpdateOp<A, Val> for Array
where
    A: ArrayIndex<'a>,
    Val: AsRef<Array>,
{
    fn try_index_update_device(
        &self,
        i: A,
        val: Val,
        mode: UpdateMode,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        index_update(self, &[i.index_op()], val.as_ref(), mode, stream)
    }
}

macro_rules! impl_try_index_update_for_tuple {
    ($(($lifetime:lifetime, $index:ident, $field:tt)),+ $(,)?) => {
        impl<$($lifetime,)+ $($index,)+ Val> TryIndexUpdateOp<($($index,)+), Val> for Array
        where
            $($index: ArrayIndex<$lifetime>,)+
            Val: AsRef<Array>,
        {
            fn try_index_update_device(
                &self,
                i: ($($index,)+),
                val: Val,
                mode: UpdateMode,
                stream: impl AsRef<Stream>,
            ) -> Result<Array> {
                let operations = [$(i.$field.index_op(),)+];
                index_update(self, &operations, val.as_ref(), mode, stream)
            }
        }
    };
}

impl_try_index_update_for_tuple!(('a, A, 0));
impl_try_index_update_for_tuple!(('a, A, 0), ('b, B, 1));
impl_try_index_update_for_tuple!(('a, A, 0), ('b, B, 1), ('c, C, 2));
impl_try_index_update_for_tuple!(('a, A, 0), ('b, B, 1), ('c, C, 2), ('d, D, 3));
impl_try_index_update_for_tuple!(('a, A, 0), ('b, B, 1), ('c, C, 2), ('d, D, 3), ('e, E, 4));
impl_try_index_update_for_tuple!(
    ('a, A, 0),
    ('b, B, 1),
    ('c, C, 2),
    ('d, D, 3),
    ('e, E, 4),
    ('f, F, 5)
);
impl_try_index_update_for_tuple!(
    ('a, A, 0),
    ('b, B, 1),
    ('c, C, 2),
    ('d, D, 3),
    ('e, E, 4),
    ('f, F, 5),
    ('g, G, 6)
);
impl_try_index_update_for_tuple!(
    ('a, A, 0),
    ('b, B, 1),
    ('c, C, 2),
    ('d, D, 3),
    ('e, E, 4),
    ('f, F, 5),
    ('g, G, 6),
    ('h, H, 7)
);

#[cfg(test)]
mod tests {
    use crate::ops::indexing::{NewAxis, TryIndexUpdateOp, UpdateMode};
    use crate::{Array, array};

    fn grid() -> Array {
        Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
    }

    #[test]
    fn slice_updates_follow_the_mode() {
        let a = grid();
        let update = array!([10.0f32, 0.5]);

        let cases = [
            (UpdateMode::Replace, [1.0, 10.0, 3.0, 4.0, 0.5, 6.0]),
            (UpdateMode::Add, [1.0, 12.0, 3.0, 4.0, 5.5, 6.0]),
            (UpdateMode::Min, [1.0, 2.0, 3.0, 4.0, 0.5, 6.0]),
            (UpdateMode::Max, [1.0, 10.0, 3.0, 4.0, 5.0, 6.0]),
            (UpdateMode::Product, [1.0, 20.0, 3.0, 4.0, 2.5, 6.0]),
        ];
        for (mode, expected) in cases {
            // Column 1 of both rows: a strided slice update.
            let b = a.try_index_update((.., 1), &update, mode).unwrap();
            assert_eq!(b.as_slice::<f32>(), &expected, "{mode:?}");
        }
        assert_eq!(
            a.as_slice::<f32>(),
            grid().as_slice::<f32>(),
            "source must be unchanged"
        );
    }

    #[test]
    fn array_index_updates_accumulate_repeated_indices() {
        let a = array!([1.0f32, 1.0, 1.0, 1.0]);
        let indices = array!([0u32, 2, 2]);

        let added = a
            .try_index_update(&indices, array!(1.0f32), UpdateMode::Add)
            .unwrap();
        assert_eq!(added.as_slice::<f32>(), &[2.0, 1.0, 3.0, 1.0]);

        let multiplied = a
            .try_index_update(&indices, array!([2.0f32, 3.0, 4.0]), UpdateMode::Product)
            .unwrap();
        assert_eq!(multiplied.as_slice::<f32>(), &[2.0, 1.0, 12.0, 1.0]);
    }

    #[test]
    fn whole_array_index_combines_with_the_source() {
        let a = grid();

        let added = a
            .try_index_update(.., array!(1.0f32), UpdateMode::Add)
            .unwrap();
        assert_eq!(added.as_slice::<f32>(), &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        let maxed = a
            .try_index_update(NewAxis, array!(3.5f32), UpdateMode::Max)
            .unwrap();
        assert_eq!(maxed.as_slice::<f32>(), &[3.5, 3.5, 3.5, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn replace_matches_index_mut() {
        let mut expected = grid();
        crate::ops::indexing::IndexMutOp::index_mut(&mut expected, (1, 0..2), array!(9.0f32));

        let b = grid()
            .try_index_update((1, 0..2), array!(9.0f32), UpdateMode::Replace)
            .unwrap();
        assert_eq!(b.as_slice::<f32>(), expected.as_slice::<f32>());
    }
}
