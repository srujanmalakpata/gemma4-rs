//! Matrix-vector products over weights kept in their GGUF storage formats.
//!
//! Like ggml's CPU backend, the activation vector is first converted to the
//! weight's format (Q8_0 blocks for Q8_0 weights, bf16 for BF16 weights), so
//! the inner loops are integer or bf16 dot products. Weights are row-major
//! `[rows, cols]` and the result is `W @ x`.

use half::f16;
use llm_eng::EngError;

pub const Q8_0_BLOCK_VALUES: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// An activation vector quantized to Q8_0: one scale per 32 values.
#[derive(Clone, Debug, PartialEq)]
pub struct Q8Activation {
    scales: Vec<f32>,
    quants: Vec<i8>,
}

impl Q8Activation {
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    pub fn quants(&self) -> &[i8] {
        &self.quants
    }
}

/// Quantizes `x` to Q8_0 the way ggml's `quantize_row_q8_0` does: scale =
/// max|x| / 127 stored as f16, values rounded to nearest (ties to even).
pub fn quantize_q8_0(x: &[f32]) -> Result<Q8Activation, EngError> {
    if !x.len().is_multiple_of(Q8_0_BLOCK_VALUES) {
        return Err(EngError::Compute(format!(
            "Q8_0 activation length {} is not a multiple of {Q8_0_BLOCK_VALUES}",
            x.len()
        )));
    }
    let mut scales = Vec::with_capacity(x.len() / Q8_0_BLOCK_VALUES);
    let mut quants = Vec::with_capacity(x.len());
    for block in x.chunks_exact(Q8_0_BLOCK_VALUES) {
        let max = block.iter().fold(0.0_f32, |max, value| max.max(value.abs()));
        let scale = max / 127.0;
        let inverse = if scale == 0.0 { 0.0 } else { scale.recip() };
        scales.push(f16::from_f32(scale).to_f32());
        quants.extend(block.iter().map(|value| (value * inverse).round_ties_even() as i8));
    }
    Ok(Q8Activation { scales, quants })
}

/// `W @ x` for Q8_0 weights: each row is `cols / 32` blocks of an f16 scale
/// followed by 32 int8 values.
pub fn matvec_q8_0(weights: &[u8], rows: usize, x: &Q8Activation) -> Result<Vec<f32>, EngError> {
    let cols = x.quants.len();
    let row_bytes = cols / Q8_0_BLOCK_VALUES * Q8_0_BLOCK_BYTES;
    check_shape(weights.len(), rows, row_bytes, "Q8_0")?;
    Ok(parallel_rows(rows, cols, |row| {
        let row = &weights[row * row_bytes..(row + 1) * row_bytes];
        row.chunks_exact(Q8_0_BLOCK_BYTES)
            .zip(x.quants.chunks_exact(Q8_0_BLOCK_VALUES).zip(&x.scales))
            .map(|(block, (quants, &x_scale))| {
                let w_scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
                let dot: i32 = block[2..]
                    .iter()
                    .zip(quants)
                    .map(|(&w, &q)| i32::from(w as i8) * i32::from(q))
                    .sum();
                dot as f32 * (w_scale * x_scale)
            })
            .sum()
    }))
}

/// `W @ x` for Q8_0 weights with an unquantized f32 activation: each weight
/// block is dequantized on the fly. Slower than [`matvec_q8_0`], but the
/// only rounding is in the stored weights.
pub fn matvec_q8_0_f32(weights: &[u8], rows: usize, x: &[f32]) -> Result<Vec<f32>, EngError> {
    let cols = x.len();
    if !cols.is_multiple_of(Q8_0_BLOCK_VALUES) {
        return Err(EngError::Compute(format!("Q8_0 row length {cols} is not a multiple of {Q8_0_BLOCK_VALUES}")));
    }
    let row_bytes = cols / Q8_0_BLOCK_VALUES * Q8_0_BLOCK_BYTES;
    check_shape(weights.len(), rows, row_bytes, "Q8_0")?;
    Ok(parallel_rows(rows, cols, |row| {
        let row = &weights[row * row_bytes..(row + 1) * row_bytes];
        row.chunks_exact(Q8_0_BLOCK_BYTES)
            .zip(x.chunks_exact(Q8_0_BLOCK_VALUES))
            .map(|(block, x)| {
                let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
                let dot: f32 = block[2..].iter().zip(x).map(|(&w, &x)| f32::from(w as i8) * x).sum();
                dot * scale
            })
            .sum()
    }))
}

/// `W @ x` for BF16 weights; `x` is rounded to bf16 first, as ggml does.
pub fn matvec_bf16(weights: &[u16], rows: usize, x: &[f32]) -> Result<Vec<f32>, EngError> {
    let cols = x.len();
    check_shape(weights.len(), rows, cols, "BF16")?;
    let x: Vec<f32> = x.iter().map(|&value| bf16_to_f32(f32_to_bf16(value))).collect();
    Ok(parallel_rows(rows, cols, |row| {
        weights[row * cols..(row + 1) * cols]
            .iter()
            .zip(&x)
            .map(|(&w, &x)| bf16_to_f32(w) * x)
            .sum()
    }))
}

/// `W @ x` for f32 weights.
pub fn matvec_f32(weights: &[f32], rows: usize, x: &[f32]) -> Result<Vec<f32>, EngError> {
    let cols = x.len();
    check_shape(weights.len(), rows, cols, "f32")?;
    Ok(parallel_rows(rows, cols, |row| {
        weights[row * cols..(row + 1) * cols].iter().zip(x).map(|(w, x)| w * x).sum()
    }))
}

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Rounds to the nearest bf16, ties to even (ggml's `GGML_FP32_TO_BF16`).
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) | 64) as u16;
    }
    ((bits + (0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}

fn check_shape(len: usize, rows: usize, row_len: usize, kind: &str) -> Result<(), EngError> {
    if rows.checked_mul(row_len) == Some(len) {
        Ok(())
    } else {
        Err(EngError::Compute(format!(
            "{kind} weight has {len} elements, expected {rows} rows of {row_len}"
        )))
    }
}

/// Below this many multiply-adds a mat-vec runs on one thread, since
/// starting threads would cost more than it saves.
const PARALLEL_MIN_WORK: usize = 1 << 20;

/// Evaluates `row_value` for every row, split across threads for large
/// matrices.
fn parallel_rows(rows: usize, cols: usize, row_value: impl Fn(usize) -> f32 + Sync) -> Vec<f32> {
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    if threads == 1 || rows * cols < PARALLEL_MIN_WORK {
        return (0..rows).map(row_value).collect();
    }
    let mut out = vec![0.0; rows];
    let chunk = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        for (index, part) in out.chunks_mut(chunk).enumerate() {
            let row_value = &row_value;
            scope.spawn(move || {
                for (offset, value) in part.iter_mut().enumerate() {
                    *value = row_value(index * chunk + offset);
                }
            });
        }
    });
    out
}
