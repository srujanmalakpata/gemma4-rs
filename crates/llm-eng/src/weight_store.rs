use std::collections::HashMap;

use crate::EngError;

/// A weight in the format it is stored in. Matrices are row-major
/// `[nrows, ncols]`.
pub enum TensorBlob {
    F32(Vec<f32>),
    /// ggml Q8_0 blocks: per 32 values, an f16 scale then 32 int8 values.
    Q8 {
        data: Vec<u8>,
        nrows: usize,
        ncols: usize,
    },
    /// bfloat16 bit patterns.
    Bf16 {
        data: Vec<u16>,
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
