use std::{
    collections::HashMap,
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    str,
};

use thiserror::Error;

use crate::{
    header::{Header, GGUF_MAGIC, GGUF_VERSION},
    tensor::{TensorDescriptor, TensorInfo, TensorType},
};

const ALIGNMENT: u64 = 32;

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("I/O error reading `{path}`: {source}")]
    IoPath {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("invalid GGUF magic {found:#010x}")]
    InvalidMagic { found: u32 },
    #[error("unsupported GGUF version {found}; expected {expected}")]
    UnsupportedVersion { found: u32, expected: u32 },
    #[error("unsupported GGUF metadata type {ty} for key `{key}`")]
    UnsupportedMetadataType { key: String, ty: u32 },
    #[error("unsupported GGUF tensor type {ty} for tensor `{name}`")]
    UnsupportedTensorType { name: String, ty: u32 },
    #[error("invalid UTF-8 in {context}")]
    InvalidUtf8 { context: &'static str },
    #[error("tensor `{name}` was not found")]
    MissingTensor { name: String },
    #[error("invalid tensor `{name}`: {reason}")]
    InvalidTensor { name: String, reason: String },
    #[error("GGUF value is too large for this platform")]
    ValueTooLarge,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
enum MetadataValue {
    U32(u32),
    U64(u64),
    F32(f32),
    Bool(bool),
    String(String),
    Strings(Vec<String>),
    F32s(Vec<f32>),
    I32s(Vec<i32>),
    Array,
}

pub struct GgufFile<R> {
    reader: R,
    #[allow(dead_code)]
    header: Header,
    metadata: HashMap<String, MetadataValue>,
    tensors: HashMap<String, TensorDescriptor>,
    tensor_order: Vec<String>,
    data_offset: u64,
}

impl GgufFile<File> {
    pub fn open_path(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let path = path.as_ref().to_path_buf();
        let reader = File::open(&path).map_err(|source| GgufError::IoPath {
            path: path.clone(),
            source,
        })?;
        Self::open_read(reader).map_err(|error| match error {
            GgufError::Io(source) => GgufError::IoPath { path, source },
            error => error,
        })
    }
}

impl<R: Read + Seek> GgufFile<R> {
    pub fn open_read(mut reader: R) -> Result<Self, GgufError> {
        let magic = read_u32(&mut reader)?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic { found: magic });
        }

        let version = read_u32(&mut reader)?;
        if version != GGUF_VERSION {
            return Err(GgufError::UnsupportedVersion {
                found: version,
                expected: GGUF_VERSION,
            });
        }

        let header = Header {
            version,
            tensor_count: read_u64(&mut reader)?,
            metadata_count: read_u64(&mut reader)?,
        };

        let mut metadata = HashMap::new();
        for _ in 0..header.metadata_count {
            let key = read_string(&mut reader, "metadata key")?;
            let ty = read_u32(&mut reader)?;
            let value = read_metadata_value(&mut reader, &key, ty)?;
            metadata.insert(key, value);
        }

        let mut tensors = HashMap::new();
        let mut tensor_order = Vec::with_capacity(to_usize(header.tensor_count)?);
        for _ in 0..header.tensor_count {
            let name = read_string(&mut reader, "tensor name")?;
            let ndim = to_usize(read_u32(&mut reader)? as u64)?;
            let mut dimensions = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                let dimension = read_u64(&mut reader)?;
                if dimension > usize::MAX as u64 {
                    return Err(GgufError::ValueTooLarge);
                }
                dimensions.push(dimension);
            }
            let raw_ty = read_u32(&mut reader)?;
            let ty =
                TensorType::from_ggml(raw_ty).ok_or_else(|| GgufError::UnsupportedTensorType {
                    name: name.clone(),
                    ty: raw_ty,
                })?;
            let offset = read_u64(&mut reader)?;

            let descriptor = TensorDescriptor {
                name: name.clone(),
                dimensions,
                ty,
                offset,
            };
            validate_tensor(&descriptor)?;
            tensor_order.push(name.clone());
            tensors.insert(name, descriptor);
        }

        let position = reader.stream_position()?;
        let data_offset = align_up(position, ALIGNMENT)?;
        reader.seek(SeekFrom::Start(data_offset))?;

        Ok(Self {
            reader,
            header,
            metadata,
            tensors,
            tensor_order,
            data_offset,
        })
    }

    pub fn metadata_str(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(MetadataValue::String(value)) => Some(value),
            _ => None,
        }
    }

    pub fn metadata_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key) {
            Some(MetadataValue::U32(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn metadata_usize(&self, key: &str) -> Option<usize> {
        match self.metadata.get(key) {
            Some(MetadataValue::U32(value)) => Some(*value as usize),
            Some(MetadataValue::U64(value)) => usize::try_from(*value).ok(),
            _ => None,
        }
    }

    pub fn metadata_strings(&self, key: &str) -> Option<&[String]> {
        match self.metadata.get(key) {
            Some(MetadataValue::Strings(values)) => Some(values),
            _ => None,
        }
    }

    pub fn metadata_f32s(&self, key: &str) -> Option<&[f32]> {
        match self.metadata.get(key) {
            Some(MetadataValue::F32s(values)) => Some(values),
            _ => None,
        }
    }

    pub fn metadata_i32s(&self, key: &str) -> Option<&[i32]> {
        match self.metadata.get(key) {
            Some(MetadataValue::I32s(values)) => Some(values),
            _ => None,
        }
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensor_order.iter().map(String::as_str).collect()
    }

    pub fn tensor(&self, name: &str) -> Result<TensorInfo<'_>, GgufError> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::MissingTensor {
                name: name.to_owned(),
            })?;

        Ok(TensorInfo {
            name: &tensor.name,
            dimensions: &tensor.dimensions,
            ty: tensor.ty,
            offset: tensor.offset,
        })
    }

    pub fn read_tensor_bytes(&mut self, name: &str) -> Result<Vec<u8>, GgufError> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::MissingTensor {
                name: name.to_owned(),
            })?;
        let byte_len = tensor_byte_len(tensor)?;
        let start = self
            .data_offset
            .checked_add(tensor.offset)
            .ok_or(GgufError::ValueTooLarge)?;

        self.reader.seek(SeekFrom::Start(start))?;
        let mut bytes = vec![0; byte_len];
        self.reader.read_exact(&mut bytes)?;
        Ok(bytes)
    }
}

fn read_metadata_value<R: Read>(
    reader: &mut R,
    key: &str,
    ty: u32,
) -> Result<MetadataValue, GgufError> {
    match ty {
        4 => Ok(MetadataValue::U32(read_u32(reader)?)),
        6 => Ok(MetadataValue::F32(read_f32(reader)?)),
        7 => Ok(MetadataValue::Bool(read_bool(reader)?)),
        8 => Ok(MetadataValue::String(read_string(
            reader,
            "metadata string",
        )?)),
        9 => {
            let element_ty = read_u32(reader)?;
            let count = read_u64(reader)?;
            read_metadata_array(reader, key, element_ty, count)
        }
        10 => Ok(MetadataValue::U64(read_u64(reader)?)),
        _ => Err(GgufError::UnsupportedMetadataType {
            key: key.to_owned(),
            ty,
        }),
    }
}

fn read_metadata_array<R: Read>(
    reader: &mut R,
    key: &str,
    element_ty: u32,
    count: u64,
) -> Result<MetadataValue, GgufError> {
    match element_ty {
        5 => (0..count)
            .map(|_| read_i32(reader))
            .collect::<Result<Vec<_>, _>>()
            .map(MetadataValue::I32s),
        6 => (0..count)
            .map(|_| read_f32(reader))
            .collect::<Result<Vec<_>, _>>()
            .map(MetadataValue::F32s),
        8 => (0..count)
            .map(|_| read_string(reader, "metadata array string"))
            .collect::<Result<Vec<_>, _>>()
            .map(MetadataValue::Strings),
        _ => {
            for _ in 0..count {
                skip_metadata_value(reader, key, element_ty)?;
            }
            Ok(MetadataValue::Array)
        }
    }
}

fn skip_metadata_value<R: Read>(reader: &mut R, key: &str, ty: u32) -> Result<(), GgufError> {
    match ty {
        0 | 1 => skip_bytes(reader, 1),
        2 | 3 => skip_bytes(reader, 2),
        4 | 5 | 6 => skip_bytes(reader, 4),
        7 => read_bool(reader).map(|_| ()),
        8 => read_string(reader, "metadata array string").map(|_| ()),
        10 | 11 | 12 => skip_bytes(reader, 8),
        _ => Err(GgufError::UnsupportedMetadataType {
            key: key.to_owned(),
            ty,
        }),
    }
}

fn validate_tensor(tensor: &TensorDescriptor) -> Result<(), GgufError> {
    tensor_byte_len(tensor).map(|_| ())
}

fn tensor_byte_len(tensor: &TensorDescriptor) -> Result<usize, GgufError> {
    let numel = tensor
        .dimensions
        .iter()
        .try_fold(1usize, |count, &dimension| {
            count.checked_mul(to_usize(dimension).ok()?)
        })
        .ok_or_else(|| GgufError::InvalidTensor {
            name: tensor.name.clone(),
            reason: "element count overflows usize".to_owned(),
        })?;

    tensor
        .ty
        .byte_len(numel)
        .ok_or_else(|| GgufError::InvalidTensor {
            name: tensor.name.clone(),
            reason: match tensor.ty {
                TensorType::Q8_0 => "Q8_0 tensors require a multiple of 32 elements".to_owned(),
                _ => "byte count overflows usize".to_owned(),
            },
        })
}

fn read_string<R: Read>(reader: &mut R, context: &'static str) -> Result<String, GgufError> {
    let length = to_usize(read_u64(reader)?)?;
    let mut bytes = vec![0; length];
    reader.read_exact(&mut bytes)?;
    str::from_utf8(&bytes)
        .map(str::to_owned)
        .map_err(|_| GgufError::InvalidUtf8 { context })
}

fn read_bool<R: Read>(reader: &mut R) -> Result<bool, GgufError> {
    match read_u8(reader)? {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(GgufError::InvalidTensor {
            name: "metadata bool".to_owned(),
            reason: "must be 0 or 1".to_owned(),
        }),
    }
}

fn read_u8<R: Read>(reader: &mut R) -> Result<u8, GgufError> {
    let mut bytes = [0; 1];
    reader.read_exact(&mut bytes)?;
    Ok(bytes[0])
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32, GgufError> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i32<R: Read>(reader: &mut R) -> Result<i32, GgufError> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64, GgufError> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f32<R: Read>(reader: &mut R) -> Result<f32, GgufError> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn skip_bytes<R: Read>(reader: &mut R, length: usize) -> Result<(), GgufError> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes[..length])?;
    Ok(())
}

fn to_usize(value: u64) -> Result<usize, GgufError> {
    usize::try_from(value).map_err(|_| GgufError::ValueTooLarge)
}

fn align_up(value: u64, alignment: u64) -> Result<u64, GgufError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or(GgufError::ValueTooLarge)
}
