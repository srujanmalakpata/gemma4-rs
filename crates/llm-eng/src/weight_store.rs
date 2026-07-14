use std::collections::HashMap;

use crate::EngError;

pub enum TensorBlob {
    F32(Vec<f32>),
    Q8 {
        data: Vec<u8>,
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
