use half::f16;
use llm_backend_cpu::buffer::F32Buf;
use llm_backend_cpu::dequant::dequant_q8_0;
use llm_backend_cpu::ops::{add_f32, matmul_f32, rms_norm_f32, rope_f32, softmax_f32};
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
fn cpu_port_dispatches_compute_operations() {
    let port = CpuPort;
    let a = F32Buf::from(vec![1.0, 2.0, 3.0, 4.0]);
    let b = F32Buf::from(vec![5.0, 6.0, 7.0, 8.0]);

    assert_eq!(
        port.matmul(&a, &b, 2, 2, 2).unwrap().as_slice(),
        &[19.0, 22.0, 43.0, 50.0]
    );
}
