//! Matrix-vector products over weights kept in their GGUF storage formats.
//!
//! Like ggml's CPU backend, the activation vector is first converted to the
//! weight's format (Q8_0 blocks for Q8_0 weights, bf16 for BF16 weights), so
//! the inner loops are integer or bf16 dot products. Weights are row-major
//! `[rows, cols]` and the result is `W @ x`.

use std::{borrow::Borrow, sync::Arc};

use half::f16;
use llm_eng::EngError;

use crate::pool::map_indices;

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
pub fn matvec_q8_0(weights: &Arc<Vec<u8>>, rows: usize, x: Q8Activation) -> Result<Vec<f32>, EngError> {
    let cols = x.quants.len();
    let row_len = cols / Q8_0_BLOCK_VALUES * Q8_0_BLOCK_BYTES;
    check_shape(weights.len(), rows, row_len, "Q8_0")?;
    Ok(map_rows(weights, row_len, rows, cols, x, dot_q8_0))
}

/// [`matvec_q8_0`] followed by `post(value, arg)` on every output, done on
/// the worker threads (used for the logit soft-cap over Gemma's 262,144
/// vocabulary entries, which was slow on one thread).
pub fn matvec_q8_0_then(
    weights: &Arc<Vec<u8>>,
    rows: usize,
    x: Q8Activation,
    post: fn(f32, f32) -> f32,
    arg: f32,
) -> Result<Vec<f32>, EngError> {
    let cols = x.quants.len();
    let row_len = cols / Q8_0_BLOCK_VALUES * Q8_0_BLOCK_BYTES;
    check_shape(weights.len(), rows, row_len, "Q8_0")?;
    let job = Arc::new(MappedRows { weights: Arc::clone(weights), row_len, x, post, arg });
    Ok(map_indices(job, rows, cols, PARALLEL_MIN_WORK, |job, row| {
        (job.post)(dot_q8_0(&job.weights[row * job.row_len..(row + 1) * job.row_len], &job.x), job.arg)
    }))
}

/// Gemma's gated FFN input in one pass over two Q8_0 matrices of the same
/// shape: `gelu(gate_row . x) * (up_row . x)` for every row. Fusing them
/// saves a task per layer and runs GELU on the worker threads.
pub fn geglu_q8_0(
    gate: &Arc<Vec<u8>>,
    up: &Arc<Vec<u8>>,
    rows: usize,
    x: Q8Activation,
    gelu: fn(f32) -> f32,
) -> Result<Vec<f32>, EngError> {
    let cols = x.quants.len();
    let row_len = cols / Q8_0_BLOCK_VALUES * Q8_0_BLOCK_BYTES;
    check_shape(gate.len(), rows, row_len, "Q8_0")?;
    check_shape(up.len(), rows, row_len, "Q8_0")?;
    let job = Arc::new(GegluRows { gate: Arc::clone(gate), up: Arc::clone(up), row_len, x, gelu });
    Ok(map_indices(job, rows, 2 * cols, PARALLEL_MIN_WORK, |job, row| {
        let range = row * job.row_len..(row + 1) * job.row_len;
        (job.gelu)(dot_q8_0(&job.gate[range.clone()], &job.x)) * dot_q8_0(&job.up[range], &job.x)
    }))
}

/// `W @ x` for Q8_0 weights with an unquantized f32 activation: each weight
/// block is dequantized on the fly. Slower than [`matvec_q8_0`], but the
/// only rounding is in the stored weights.
pub fn matvec_q8_0_f32(weights: &Arc<Vec<u8>>, rows: usize, x: &[f32]) -> Result<Vec<f32>, EngError> {
    let row_len = q8_0_row_len(x.len())?;
    check_shape(weights.len(), rows, row_len, "Q8_0")?;
    Ok(map_rows(weights, row_len, rows, x.len(), x.to_vec(), dot_q8_0_f32))
}

/// Like [`matvec_q8_0_f32`], but every sum is taken in f64. Slower, with
/// almost no rounding in the sums; comparing the two shows how much float
/// summation order alone moves a model's output.
pub fn matvec_q8_0_f64_sums(weights: &Arc<Vec<u8>>, rows: usize, x: &[f32]) -> Result<Vec<f32>, EngError> {
    let row_len = q8_0_row_len(x.len())?;
    check_shape(weights.len(), rows, row_len, "Q8_0")?;
    Ok(map_rows(weights, row_len, rows, x.len(), x.to_vec(), dot_q8_0_f64_sums))
}

/// `W @ x` for BF16 weights; `x` is rounded to bf16 first, as ggml does.
pub fn matvec_bf16(weights: &Arc<Vec<u16>>, rows: usize, x: &[f32]) -> Result<Vec<f32>, EngError> {
    let cols = x.len();
    check_shape(weights.len(), rows, cols, "BF16")?;
    let x: Vec<f32> = x.iter().map(|&value| bf16_to_f32(f32_to_bf16(value))).collect();
    Ok(map_rows(weights, cols, rows, cols, x, dot_bf16))
}

/// `W @ x` for f32 weights.
pub fn matvec_f32(weights: &Arc<Vec<f32>>, rows: usize, x: &[f32]) -> Result<Vec<f32>, EngError> {
    let cols = x.len();
    check_shape(weights.len(), rows, cols, "f32")?;
    Ok(map_rows(weights, cols, rows, cols, x.to_vec(), dot_f32))
}

fn dot_q8_0(row: &[u8], x: &Q8Activation) -> f32 {
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
}

fn dot_q8_0_f32(row: &[u8], x: &[f32]) -> f32 {
    row.chunks_exact(Q8_0_BLOCK_BYTES)
        .zip(x.chunks_exact(Q8_0_BLOCK_VALUES))
        .map(|(block, x)| {
            let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dot: f32 = block[2..].iter().zip(x).map(|(&w, &x)| f32::from(w as i8) * x).sum();
            dot * scale
        })
        .sum()
}

fn dot_q8_0_f64_sums(row: &[u8], x: &[f32]) -> f32 {
    let sum: f64 = row
        .chunks_exact(Q8_0_BLOCK_BYTES)
        .zip(x.chunks_exact(Q8_0_BLOCK_VALUES))
        .map(|(block, x)| {
            let scale = f64::from(f16::from_le_bytes([block[0], block[1]]).to_f32());
            let dot: f64 = block[2..].iter().zip(x).map(|(&w, &x)| f64::from(w as i8) * f64::from(x)).sum();
            dot * scale
        })
        .sum();
    sum as f32
}

fn dot_bf16(row: &[u16], x: &[f32]) -> f32 {
    row.iter().zip(x).map(|(&w, &x)| bf16_to_f32(w) * x).sum()
}

fn dot_f32(row: &[f32], x: &[f32]) -> f32 {
    row.iter().zip(x).map(|(w, x)| w * x).sum()
}

fn q8_0_row_len(cols: usize) -> Result<usize, EngError> {
    if cols.is_multiple_of(Q8_0_BLOCK_VALUES) {
        Ok(cols / Q8_0_BLOCK_VALUES * Q8_0_BLOCK_BYTES)
    } else {
        Err(EngError::Compute(format!("Q8_0 row length {cols} is not a multiple of {Q8_0_BLOCK_VALUES}")))
    }
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

/// Below this many multiply-adds a mat-vec runs on the calling thread,
/// since handing it to the workers would cost more than it saves.
const PARALLEL_MIN_WORK: usize = 1 << 20;

/// One mat-vec as a job the worker threads can share. The job owns its
/// activation `x` and hands `dot` a borrowed view of it (`&[f32]` for a
/// `Vec<f32>`).
struct Rows<W, X, V: ?Sized> {
    weights: Arc<Vec<W>>,
    row_len: usize,
    x: X,
    dot: fn(&[W], &V) -> f32,
}

struct MappedRows {
    weights: Arc<Vec<u8>>,
    row_len: usize,
    x: Q8Activation,
    post: fn(f32, f32) -> f32,
    arg: f32,
}

struct GegluRows {
    gate: Arc<Vec<u8>>,
    up: Arc<Vec<u8>>,
    row_len: usize,
    x: Q8Activation,
    gelu: fn(f32) -> f32,
}

/// Computes `dot(row, x)` for every row of `weights`, split across the
/// worker threads for large matrices.
fn map_rows<W, X, V>(weights: &Arc<Vec<W>>, row_len: usize, rows: usize, cols: usize, x: X, dot: fn(&[W], &V) -> f32) -> Vec<f32>
where
    W: Send + Sync + 'static,
    X: Borrow<V> + Send + Sync + 'static,
    V: ?Sized + 'static,
{
    let job = Arc::new(Rows { weights: Arc::clone(weights), row_len, x, dot });
    map_indices(job, rows, cols, PARALLEL_MIN_WORK, |job, row| {
        (job.dot)(&job.weights[row * job.row_len..(row + 1) * job.row_len], job.x.borrow())
    })
}
