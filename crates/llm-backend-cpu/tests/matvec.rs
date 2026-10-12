use std::sync::Arc;

use half::f16;
use llm_backend_cpu::dequant::dequant_q8_0;
use llm_backend_cpu::matvec::{
    bf16_to_f32, f32_to_bf16, geglu_q8_0, matvec_bf16, matvec_f32, matvec_q8_0, matvec_q8_0_f32, matvec_q8_0_f64_sums, matvec_q8_0_then, quantize_q8_0,
};

/// Deterministic values in [-1, 1).
fn values(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (state >> 40) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

/// Quantizes a row-major matrix to Q8_0 bytes, as llama.cpp's quantizer does.
fn q8_0_bytes(weights: &[f32]) -> Vec<u8> {
    weights
        .chunks_exact(32)
        .flat_map(|block| {
            let scale = block.iter().fold(0.0_f32, |max, v| max.max(v.abs())) / 127.0;
            let mut bytes = f16::from_f32(scale).to_le_bytes().to_vec();
            bytes.extend(block.iter().map(|v| (v / scale).round() as i8 as u8));
            bytes
        })
        .collect()
}

fn naive(weights: &[f32], x: &[f32]) -> Vec<f32> {
    weights
        .chunks_exact(x.len())
        .map(|row| row.iter().zip(x).map(|(w, x)| w * x).sum())
        .collect()
}

#[test]
fn quantize_q8_0_uses_max_over_127_and_rounds_half_to_even() {
    let mut block = [0.0_f32; 32];
    block[0] = 127.0;
    block[1] = 2.5;
    block[2] = -3.5;
    let quantized = quantize_q8_0(&block).unwrap();
    assert_eq!(quantized.scales(), [1.0]);
    assert_eq!(quantized.quants()[..4], [127, 2, -4, 0]);
}

#[test]
fn q8_0_matvec_equals_the_dot_product_of_the_dequantized_values() {
    let (rows, cols) = (64, 256);
    let bytes = Arc::new(q8_0_bytes(&values(1, rows * cols)));
    let x = quantize_q8_0(&values(2, cols)).unwrap();
    let x_dequantized: Vec<f32> = x
        .quants()
        .chunks_exact(32)
        .zip(x.scales())
        .flat_map(|(quants, scale)| quants.iter().map(move |&q| f32::from(q) * scale))
        .collect();
    let expected = naive(&dequant_q8_0(&bytes).unwrap(), &x_dequantized);
    let actual = matvec_q8_0(&bytes, rows, x).unwrap();
    for (e, a) in expected.iter().zip(&actual) {
        assert!((e - a).abs() < 1e-4, "{e} vs {a}");
    }
}

#[test]
fn q8_0_matvec_with_f32_activations_equals_the_dequantized_product() {
    let (rows, cols) = (64, 256);
    let bytes = Arc::new(q8_0_bytes(&values(1, rows * cols)));
    let x = values(2, cols);
    let expected = naive(&dequant_q8_0(&bytes).unwrap(), &x);
    let actual = matvec_q8_0_f32(&bytes, rows, &x).unwrap();
    let f64_sums = matvec_q8_0_f64_sums(&bytes, rows, &x).unwrap();
    for ((e, a), b) in expected.iter().zip(&actual).zip(&f64_sums) {
        assert!((e - a).abs() < 1e-4, "{e} vs {a}");
        assert!((e - b).abs() < 1e-4, "{e} vs {b}");
    }
}

#[test]
fn fused_kernels_give_the_same_bits_as_separate_steps() {
    let (rows, cols) = (300, 256);
    let gate = Arc::new(q8_0_bytes(&values(5, rows * cols)));
    let up = Arc::new(q8_0_bytes(&values(6, rows * cols)));
    let x = quantize_q8_0(&values(7, cols)).unwrap();
    let gelu = |v: f32| 0.5 * v * (1.0 + v.tanh());
    let g = matvec_q8_0(&gate, rows, x.clone()).unwrap();
    let u = matvec_q8_0(&up, rows, x.clone()).unwrap();
    let separate: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| gelu(g) * u).collect();
    assert_eq!(geglu_q8_0(&gate, &up, rows, x.clone(), gelu).unwrap(), separate);

    let capped: Vec<f32> = g.iter().map(|&v| (v * 0.5).tanh() * 2.0).collect();
    let post = |v: f32, cap: f32| (v * cap.recip()).tanh() * cap;
    assert_eq!(matvec_q8_0_then(&gate, rows, x, post, 2.0).unwrap(), capped);
}

#[test]
fn q8_0_matvec_is_close_to_the_f32_product() {
    let (rows, cols) = (64, 256);
    let weights = values(1, rows * cols);
    let x = values(2, cols);
    let expected = naive(&weights, &x);
    let actual = matvec_q8_0(&Arc::new(q8_0_bytes(&weights)), rows, quantize_q8_0(&x).unwrap()).unwrap();
    // Two 8-bit roundings per term over 256 terms: a few hundredths.
    for (e, a) in expected.iter().zip(&actual) {
        assert!((e - a).abs() < 0.25, "{e} vs {a}");
    }
}

#[test]
fn large_matvecs_split_across_threads_give_the_same_result() {
    // 2^20 multiply-adds or more go to the worker threads, in chunks.
    let (rows, cols) = (1031, 1024);
    let weights = values(3, rows * cols);
    let x = values(4, cols);
    let expected = naive(&weights, &x);
    let actual = matvec_f32(&Arc::new(weights), rows, &x).unwrap();
    assert_eq!(actual.len(), rows);
    for (e, a) in expected.iter().zip(&actual) {
        assert!((e - a).abs() < 1e-3, "{e} vs {a}");
    }
}

#[test]
fn bf16_rounds_to_nearest_even_and_matvec_uses_rounded_inputs() {
    assert_eq!(f32_to_bf16(1.0), 0x3f80);
    // 1 + 2^-8 is exactly halfway between bf16 1.0 and 1.0078125: ties to even.
    assert_eq!(f32_to_bf16(1.0 + 1.0 / 256.0), 0x3f80);
    assert_eq!(f32_to_bf16(1.0 + 3.0 / 256.0), 0x3f82);
    assert_eq!(bf16_to_f32(0x3f82), 1.015625);

    let weights = Arc::new(vec![f32_to_bf16(2.0), f32_to_bf16(-0.5)]);
    let out = matvec_bf16(&weights, 1, &[1.0 + 1.0 / 256.0, 4.0]).unwrap();
    assert_eq!(out, [2.0 * 1.0 - 0.5 * 4.0]);
}

#[test]
fn shape_mismatches_are_errors() {
    assert!(matvec_f32(&Arc::new(vec![1.0; 6]), 4, &[1.0, 1.0]).is_err());
    assert!(quantize_q8_0(&[1.0; 31]).is_err());
}
