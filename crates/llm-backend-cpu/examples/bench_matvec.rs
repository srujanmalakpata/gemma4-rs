use std::{sync::Arc, time::Instant};
use llm_backend_cpu::matvec::{matvec_q8_0, quantize_q8_0};
fn main() {
    let (rows, cols) = (262_144usize, 1536usize);
    let blocks = rows * cols / 32;
    let mut bytes = Vec::with_capacity(blocks * 34);
    let mut s: u64 = 1;
    for _ in 0..blocks {
        bytes.extend_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
        for _ in 0..32 { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); bytes.push((s >> 56) as u8); }
    }
    let w = Arc::new(bytes);
    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.37).sin()).collect();
    for _ in 0..2 { let _ = matvec_q8_0(&w, rows, quantize_q8_0(&x).unwrap()).unwrap(); }
    let n = 10;
    let t = Instant::now();
    for _ in 0..n { let _ = matvec_q8_0(&w, rows, quantize_q8_0(&x).unwrap()).unwrap(); }
    let per = t.elapsed().as_secs_f64() / n as f64;
    println!("threads={} {:.2} ms per matvec, {:.1} GB/s, {:.1} GMAC/s", llm_backend_cpu::pool::threads(), per * 1e3, w.len() as f64 / per / 1e9, (rows * cols) as f64 / per / 1e9);
}
