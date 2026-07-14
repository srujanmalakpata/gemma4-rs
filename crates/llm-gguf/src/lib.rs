pub mod header;
pub mod reader;
pub mod tensor;

// Integration tests compile this crate without `cfg(test)`, so this needs to remain public.
#[doc(hidden)]
pub mod testutil {
    const GGUF_VERSION: u32 = 3;
    const GGML_TYPE_Q8_0: u32 = 8;

    pub fn write_minimal_q8_tensor(name: &str, values: &[i8; 32]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, GGUF_VERSION);
        write_u64(&mut bytes, 1);
        write_u64(&mut bytes, 0);

        write_string(&mut bytes, name);
        write_u32(&mut bytes, 1);
        write_u64(&mut bytes, 32);
        write_u32(&mut bytes, GGML_TYPE_Q8_0);
        write_u64(&mut bytes, 0);

        bytes.resize(bytes.len().next_multiple_of(32), 0);
        bytes.extend_from_slice(&0x3c00u16.to_le_bytes());
        bytes.extend(values.iter().map(|&value| value as u8));
        bytes
    }

    fn write_string(bytes: &mut Vec<u8>, value: &str) {
        write_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }

    fn write_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}
