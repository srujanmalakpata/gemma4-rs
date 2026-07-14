use crate::EngError;

pub trait ComputePort {
    type Buffer;

    fn matmul(
        &self,
        a: &Self::Buffer,
        b: &Self::Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Self::Buffer, EngError>;
    fn rms_norm(
        &self,
        x: &Self::Buffer,
        weight: &Self::Buffer,
        eps: f32,
    ) -> Result<Self::Buffer, EngError>;
    fn softmax(&self, x: &Self::Buffer) -> Result<Self::Buffer, EngError>;
    fn rope(
        &self,
        x: &mut Self::Buffer,
        pos: usize,
        theta: f32,
        dims: usize,
    ) -> Result<(), EngError>;
    fn add(&self, a: &Self::Buffer, b: &Self::Buffer) -> Result<Self::Buffer, EngError>;
}
