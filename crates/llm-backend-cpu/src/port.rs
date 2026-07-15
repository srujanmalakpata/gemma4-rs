use llm_eng::{compute::ComputePort, EngError};

use crate::{
    buffer::F32Buf,
    ops::{add_f32, matmul_f32, rms_norm_f32, rope_f32, softmax_f32},
};

#[derive(Clone, Copy, Debug, Default)]
pub struct CpuPort;

impl ComputePort for CpuPort {
    type Buffer = F32Buf;

    fn matmul(
        &self,
        a: &Self::Buffer,
        b: &Self::Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Self::Buffer, EngError> {
        matmul_f32(a, b, m, n, k)
    }

    fn rms_norm(
        &self,
        x: &Self::Buffer,
        weight: &Self::Buffer,
        eps: f32,
    ) -> Result<Self::Buffer, EngError> {
        rms_norm_f32(x, weight, eps)
    }

    fn softmax(&self, x: &Self::Buffer) -> Result<Self::Buffer, EngError> {
        softmax_f32(x)
    }

    fn rope(
        &self,
        x: &mut Self::Buffer,
        pos: usize,
        theta: f32,
        dims: usize,
    ) -> Result<(), EngError> {
        rope_f32(x, pos, theta, dims)
    }

    fn add(&self, a: &Self::Buffer, b: &Self::Buffer) -> Result<Self::Buffer, EngError> {
        add_f32(a, b)
    }
}
