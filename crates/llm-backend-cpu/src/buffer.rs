#[derive(Clone, Debug, PartialEq)]
pub struct F32Buf(Vec<f32>);

impl F32Buf {
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<Vec<f32>> for F32Buf {
    fn from(values: Vec<f32>) -> Self {
        Self(values)
    }
}
