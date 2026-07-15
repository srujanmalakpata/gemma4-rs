pub const GGUF_MAGIC: u32 = u32::from_le_bytes(*b"GGUF");
pub const GGUF_VERSION: u32 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
}
