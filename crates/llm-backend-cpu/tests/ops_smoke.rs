use half::f16;
use llm_backend_cpu::buffer::F32Buf;
use llm_backend_cpu::dequant::dequant_q8_0;
use llm_backend_cpu::ops::{
    add_f32, gelu_tanh, gelu_tanh_fp16_table, matmul_f32, rms_norm, rms_norm_f32, rope_f32, rope_neox, softmax_f32,
};
use llm_backend_cpu::port::CpuPort;
use llm_eng::compute::ComputePort;

fn valid_q8_block_scale1_sequential() -> Vec<u8> {
    let mut bytes = f16::from_f32(1.0).to_le_bytes().to_vec();
    bytes.extend((0_i8..32).map(|value| value as u8));
    bytes
}

#[test]
fn matmul_2x2() {
    let a = F32Buf::from(vec![1.0, 2.0, 3.0, 4.0]);
    let b = F32Buf::from(vec![5.0, 6.0, 7.0, 8.0]);

    let c = matmul_f32(&a, &b, 2, 2, 2).unwrap();

    assert_eq!(c.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn q8_0_block_dequant_roundtrip_scale() {
    let buf = dequant_q8_0(&valid_q8_block_scale1_sequential()).unwrap();

    assert_eq!(buf.len(), 32);
    assert!((buf[0] - 0.0).abs() < 1e-3);
    assert!((buf[31] - 31.0).abs() < 1e-2);
}

#[test]
fn rms_norm_scales_unit_vector() {
    let x = F32Buf::from(vec![3.0, 4.0]);
    let weight = F32Buf::from(vec![1.0, 1.0]);

    let normalized = rms_norm_f32(&x, &weight, 0.0).unwrap();

    assert!((normalized.as_slice()[0] - 0.848_528_15).abs() < 1e-6);
    assert!((normalized.as_slice()[1] - 1.131_370_9).abs() < 1e-6);
}

#[test]
fn rms_norm_uses_weights_as_plain_scales_or_none() {
    let weighted = rms_norm(&[3.0, 4.0], Some(&[2.0, 0.5]), 0.0).unwrap();
    assert!((weighted[0] - 1.697_056_3).abs() < 1e-6);
    assert!((weighted[1] - 0.565_685_4).abs() < 1e-6);

    let unweighted = rms_norm(&[3.0, 4.0], None, 0.0).unwrap();
    assert!((unweighted[0] - 0.848_528_15).abs() < 1e-6);
    assert!(rms_norm(&[3.0, 4.0], Some(&[1.0]), 0.0).is_err());
}

#[test]
fn gelu_tanh_matches_pytorch_tanh_approximation() {
    assert_eq!(gelu_tanh(0.0), 0.0);
    // torch.nn.functional.gelu(x, approximate="tanh") for x = 1, -1, 3.
    assert!((gelu_tanh(1.0) - 0.841_192).abs() < 1e-6);
    assert!((gelu_tanh(-1.0) + 0.158_808).abs() < 1e-6);
    assert!((gelu_tanh(3.0) - 2.996_363_7).abs() < 1e-6);
}

#[test]
fn fp16_table_gelu_rounds_input_and_output_to_fp16() {
    // 1.0001 rounds to fp16 1.0; gelu(1.0) = 0.841192 rounds to fp16 0.84130859.
    assert_eq!(gelu_tanh_fp16_table(1.0001), 0.841_308_6);
    assert_eq!(gelu_tanh_fp16_table(-10.0), 0.0);
    assert_eq!(gelu_tanh_fp16_table(12.5), 12.5);
}

#[test]
fn elementwise_ops_and_rope_rotate_pairs() {
    let a = F32Buf::from(vec![1.0, 2.0]);
    let b = F32Buf::from(vec![3.0, 4.0]);
    assert_eq!(add_f32(&a, &b).unwrap().as_slice(), &[4.0, 6.0]);

    let softmax = softmax_f32(&a).unwrap();
    assert!((softmax.as_slice()[0] - 0.268_941_43).abs() < 1e-6);
    assert!((softmax.as_slice()[1] - 0.731_058_6).abs() < 1e-6);

    let mut values = F32Buf::from(vec![1.0, 0.0]);
    rope_f32(&mut values, 1, 1.0, 2).unwrap();
    assert!((values.as_slice()[0] - 1.0_f32.cos()).abs() < 1e-6);
    assert!((values.as_slice()[1] - 1.0_f32.sin()).abs() < 1e-6);
}

#[test]
fn rope_neox_pairs_halves_and_scales_frequencies() {
    // n_rot = 4: pairs (x0, x2) at frequency 1 and (x1, x3) at base^-0.5.
    let mut x = [1.0, 1.0, 0.0, 0.0, 7.0];
    rope_neox(&mut x, 2, 100.0, 4, None).unwrap();
    let (slow_sin, slow_cos) = (2.0_f32 / 10.0).sin_cos();
    assert!((x[0] - 2.0_f32.cos()).abs() < 1e-6);
    assert!((x[2] - 2.0_f32.sin()).abs() < 1e-6);
    assert!((x[1] - slow_cos).abs() < 1e-6);
    assert!((x[3] - slow_sin).abs() < 1e-6);
    assert_eq!(x[4], 7.0, "values past n_rot are not rotated");
}

#[test]
fn rope_neox_huge_freq_factors_leave_pairs_unrotated() {
    let mut x = [1.0, 2.0, 3.0, 4.0];
    rope_neox(&mut x, 5, 100.0, 4, Some(&[1.0, 1e30])).unwrap();
    assert!((x[0] - (5.0_f32.cos() - 3.0 * 5.0_f32.sin())).abs() < 1e-5);
    assert!((x[1] - 2.0).abs() < 1e-6 && (x[3] - 4.0).abs() < 1e-6);
}

#[test]
fn cpu_port_dispatches_compute_operations() {
    let port = CpuPort;
    let a = F32Buf::from(vec![1.0, 2.0, 3.0, 4.0]);
    let b = F32Buf::from(vec![5.0, 6.0, 7.0, 8.0]);

    assert_eq!(
        port.matmul(&a, &b, 2, 2, 2).unwrap().as_slice(),
        &[19.0, 22.0, 43.0, 50.0]
    );
}
