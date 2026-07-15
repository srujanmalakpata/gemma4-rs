use half::f16;
use thiserror::Error;

const Q8_0_BLOCK_SIZE: usize = 34;
const Q8_0_VALUES_PER_BLOCK: usize = 32;

#[derive(Debug, Error)]
pub enum DequantError {
    #[error("Q8_0 data length {0} is not a multiple of {Q8_0_BLOCK_SIZE}")]
    InvalidLength(usize),
}

pub fn dequant_q8_0(bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    if bytes.len() % Q8_0_BLOCK_SIZE != 0 {
        return Err(DequantError::InvalidLength(bytes.len()));
    }

    let mut values = Vec::with_capacity(bytes.len() / Q8_0_BLOCK_SIZE * Q8_0_VALUES_PER_BLOCK);
    for block in bytes.chunks_exact(Q8_0_BLOCK_SIZE) {
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        values.extend(block[2..].iter().map(|&value| (value as i8) as f32 * scale));
    }
    Ok(values)
}
