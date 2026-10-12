use llm_eng::EngError;

use crate::buffer::F32Buf;

fn compute_error(message: impl Into<String>) -> EngError {
    EngError::Compute(message.into())
}

pub fn matmul_f32(
    a: &F32Buf,
    b: &F32Buf,
    m: usize,
    n: usize,
    k: usize,
) -> Result<F32Buf, EngError> {
    let a_len = m
        .checked_mul(k)
        .ok_or_else(|| compute_error("matmul dimensions overflow"))?;
    let b_len = k
        .checked_mul(n)
        .ok_or_else(|| compute_error("matmul dimensions overflow"))?;
    if a.len() != a_len || b.len() != b_len {
        return Err(compute_error(
            "matmul buffer shape does not match dimensions",
        ));
    }

    let mut result = vec![0.0; m * n];
    // ponytail: O(n³) f32 matmul — replace with blocked/SIMD or Metal port later
    for row in 0..m {
        for col in 0..n {
            for inner in 0..k {
                result[row * n + col] +=
                    a.as_slice()[row * k + inner] * b.as_slice()[inner * n + col];
            }
        }
    }
    Ok(result.into())
}

pub fn rms_norm_f32(x: &F32Buf, weight: &F32Buf, eps: f32) -> Result<F32Buf, EngError> {
    if x.len() != weight.len() || x.is_empty() || eps < 0.0 {
        return Err(compute_error("invalid RMS norm inputs"));
    }

    let mean_square = x.as_slice().iter().map(|value| value * value).sum::<f32>() / x.len() as f32;
    let scale = (mean_square + eps).sqrt().recip();
    Ok(x.as_slice()
        .iter()
        .zip(weight.as_slice())
        .map(|(value, weight)| value * scale * weight)
        .collect::<Vec<_>>()
        .into())
}

/// Gemma stores RMSNorm weights as a delta, so the effective scale is `1 + weight`.
pub fn rms_norm_gemma_f32(x: &F32Buf, weight: &F32Buf, eps: f32) -> Result<F32Buf, EngError> {
    if x.len() != weight.len() || x.is_empty() || eps < 0.0 {
        return Err(compute_error("invalid RMS norm inputs"));
    }

    let mean_square = x.as_slice().iter().map(|value| value * value).sum::<f32>() / x.len() as f32;
    let scale = (mean_square + eps).sqrt().recip();
    Ok(x.as_slice()
        .iter()
        .zip(weight.as_slice())
        .map(|(value, weight)| value * scale * (1.0 + weight))
        .collect::<Vec<_>>()
        .into())
}

pub fn softmax_f32(x: &F32Buf) -> Result<F32Buf, EngError> {
    let Some(&max) = x
        .as_slice()
        .iter()
        .max_by(|left, right| left.total_cmp(right))
    else {
        return Err(compute_error("softmax requires a non-empty buffer"));
    };
    let values = x
        .as_slice()
        .iter()
        .map(|value| (value - max).exp())
        .collect::<Vec<_>>();
    let sum = values.iter().sum::<f32>();
    if !sum.is_finite() || sum == 0.0 {
        return Err(compute_error("softmax input is not finite"));
    }
    Ok(values
        .into_iter()
        .map(|value| value / sum)
        .collect::<Vec<_>>()
        .into())
}

pub fn rope_f32(x: &mut F32Buf, pos: usize, theta: f32, dims: usize) -> Result<(), EngError> {
    if dims == 0 || !dims.is_multiple_of(2) || dims > x.len() || theta <= 0.0 {
        return Err(compute_error("invalid RoPE inputs"));
    }

    for pair in 0..dims / 2 {
        let angle = pos as f32 / theta.powf((2 * pair) as f32 / dims as f32);
        let (sin, cos) = angle.sin_cos();
        let left = 2 * pair;
        let right = left + 1;
        let (a, b) = (x.as_slice()[left], x.as_slice()[right]);
        x.as_mut_slice()[left] = a * cos - b * sin;
        x.as_mut_slice()[right] = a * sin + b * cos;
    }
    Ok(())
}

pub fn add_f32(a: &F32Buf, b: &F32Buf) -> Result<F32Buf, EngError> {
    if a.len() != b.len() {
        return Err(compute_error("add buffer lengths differ"));
    }
    Ok(a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(left, right)| left + right)
        .collect::<Vec<_>>()
        .into())
}
