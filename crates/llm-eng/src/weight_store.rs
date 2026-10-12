use std::{collections::HashMap, sync::Arc};

use crate::EngError;

/// A weight in the format it is stored in. Matrices are row-major
/// `[nrows, ncols]`. Data is shared through `Arc` so mat-vec jobs on worker
/// threads can hold it without copying.
pub enum TensorBlob {
    F32(Arc<Vec<f32>>),
    /// ggml Q8_0 blocks: per 32 values, an f16 scale then 32 int8 values.
    Q8 {
        data: Arc<Vec<u8>>,
        nrows: usize,
        ncols: usize,
    },
    /// bfloat16 bit patterns.
    Bf16 {
        data: Arc<Vec<u16>>,
        nrows: usize,
        ncols: usize,
    },
}

#[derive(Default)]
pub struct WeightStore {
    weights: HashMap<String, TensorBlob>,
}

impl WeightStore {
    pub fn insert(&mut self, name: impl Into<String>, tensor: TensorBlob) -> Option<TensorBlob> {
        self.weights.insert(name.into(), tensor)
    }

    pub fn get(&self, name: &str) -> Result<&TensorBlob, EngError> {
        self.weights
            .get(name)
            .ok_or_else(|| EngError::MissingWeight(name.to_owned()))
    }
}
