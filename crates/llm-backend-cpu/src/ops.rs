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

/// RMSNorm as ggml computes it: the mean of squares is accumulated in f64,
/// then `x / sqrt(mean + eps)`, times `weight` when there is one. Gemma 4
/// GGUFs store norm weights as plain scales (the converter applies no
/// `1 + w` shift), and Gemma 4 normalizes V with no weight at all.
pub fn rms_norm(x: &[f32], weight: Option<&[f32]>, eps: f32) -> Result<Vec<f32>, EngError> {
    if x.is_empty() || eps < 0.0 || weight.is_some_and(|weight| weight.len() != x.len()) {
        return Err(compute_error("invalid RMS norm inputs"));
    }
    let sum: f64 = x.iter().map(|value| f64::from(value * value)).sum();
    let mean = (sum / x.len() as f64) as f32;
    let scale = 1.0 / (mean + eps).sqrt();
    Ok(match weight {
        Some(weight) => x.iter().zip(weight).map(|(value, weight)| value * scale * weight).collect(),
        None => x.iter().map(|value| value * scale).collect(),
    })
}

pub fn rms_norm_f32(x: &F32Buf, weight: &F32Buf, eps: f32) -> Result<F32Buf, EngError> {
    rms_norm(x.as_slice(), Some(weight.as_slice()), eps).map(F32Buf::from)
}

/// GELU with the tanh approximation (`gelu_pytorch_tanh`, Gemma's FFN
/// activation), with ggml's constants.
pub fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEF_A: f32 = 0.044_715;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + COEF_A * x * x)).tanh())
}

/// GELU as ggml's CPU kernels evaluate it: through a table indexed by the
/// fp16 rounding of `x`, whose entries are themselves rounded to fp16, with
/// `x <= -10` mapped to 0 and `x >= 10` to `x`. Only used to reproduce
/// llama.cpp's arithmetic in comparisons; [`gelu_tanh`] is more precise.
pub fn gelu_tanh_fp16_table(x: f32) -> f32 {
    if x <= -10.0 {
        0.0
    } else if x >= 10.0 {
        x
    } else {
        half::f16::from_f32(gelu_tanh(half::f16::from_f32(x).to_f32())).to_f32()
    }
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

/// Rotary position embedding in the NEOX layout used by Gemma: the first
/// `n_rot` values form `n_rot / 2` pairs `(x[i], x[i + n_rot / 2])`, pair
/// `i` turning by `position * base^(-2i / n_rot) / freq_factors[i]`.
/// Values past `n_rot` are left alone. Follows ggml's CPU kernel, including
/// building the angles by repeated multiplication.
pub fn rope_neox(
    x: &mut [f32],
    position: usize,
    base: f32,
    n_rot: usize,
    freq_factors: Option<&[f32]>,
) -> Result<(), EngError> {
    let half = n_rot / 2;
    if n_rot == 0
        || !n_rot.is_multiple_of(2)
        || n_rot > x.len()
        || base <= 0.0
        || freq_factors.is_some_and(|factors| factors.len() < half)
    {
        return Err(compute_error("invalid RoPE inputs"));
    }
    let theta_scale = base.powf(-2.0 / n_rot as f32);
    let mut theta = position as f32;
    for i in 0..half {
        let factor = freq_factors.map_or(1.0, |factors| factors[i]);
        let angle = theta / factor;
        let (sin, cos) = (angle.sin(), angle.cos());
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * cos - b * sin;
        x[i + half] = a * sin + b * cos;
        theta *= theta_scale;
    }
    Ok(())
}

pub fn rope_f32(x: &mut F32Buf, pos: usize, theta: f32, dims: usize) -> Result<(), EngError> {
    rope_neox(x.as_mut_slice(), pos, theta, dims, None)
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
