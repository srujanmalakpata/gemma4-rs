#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    F32,
    F16,
    BF16,
    Q8_0,
}

impl TensorType {
    pub(crate) fn from_ggml(ty: u32) -> Option<Self> {
        match ty {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            8 => Some(Self::Q8_0),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    pub(crate) fn byte_len(self, numel: usize) -> Option<usize> {
        match self {
            Self::F32 => numel.checked_mul(4),
            Self::F16 | Self::BF16 => numel.checked_mul(2),
            Self::Q8_0 if numel.is_multiple_of(32) => numel.checked_div(32)?.checked_mul(34),
            Self::Q8_0 => None,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TensorDescriptor {
    pub(crate) name: String,
    pub(crate) dimensions: Vec<u64>,
    pub(crate) ty: TensorType,
    pub(crate) offset: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct TensorInfo<'a> {
    pub name: &'a str,
    pub dimensions: &'a [u64],
    pub ty: TensorType,
    pub offset: u64,
}

impl TensorInfo<'_> {
    pub fn numel(&self) -> usize {
        self.dimensions
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim as usize))
            .expect("tensor dimensions validated while parsing")
    }
}
