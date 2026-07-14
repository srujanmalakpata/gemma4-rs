use std::path::Path;

use half::f16;
use llm_backend_cpu::dequant::dequant_q8_0;
use llm_eng::weight_store::{TensorBlob, WeightStore};
use llm_gguf::{
    reader::{GgufError, GgufFile},
    tensor::TensorType,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WeightLoadError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error("expected a Gemma 4 GGUF, but `general.architecture` was {found:?}")]
    WrongArchitecture { found: Option<String> },
    #[error("cannot dequantize tensor `{name}`: {reason}")]
    Dequantization { name: String, reason: String },
    #[error("unsupported tensor type for `{name}`")]
    UnsupportedTensor { name: String },
}

/// Loads text-model tensors into RAM. Multimodal tensors are deliberately ignored.
pub fn load_gemma4_e2b(path: impl AsRef<Path>) -> Result<WeightStore, WeightLoadError> {
    let mut gguf = GgufFile::open_path(path)?;
    let architecture = gguf.metadata_str("general.architecture").map(str::to_owned);
    if !architecture
        .as_deref()
        .is_some_and(|value| value.to_ascii_lowercase().contains("gemma4"))
    {
        return Err(WeightLoadError::WrongArchitecture {
            found: architecture,
        });
    }

    let mut weights = WeightStore::default();
    let tensor_names = gguf
        .tensor_names()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    for name in tensor_names {
        if is_multimodal_tensor(&name) {
            continue;
        }
        let ty = gguf.tensor(&name)?.ty;
        let bytes = gguf.read_tensor_bytes(&name)?;
        let values = match ty {
            TensorType::F32 => bytes
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("exact chunk size")))
                .collect(),
            TensorType::F16 => bytes
                .chunks_exact(2)
                .map(|bytes| {
                    f16::from_le_bytes(bytes.try_into().expect("exact chunk size")).to_f32()
                })
                .collect(),
            TensorType::Q8_0 => {
                dequant_q8_0(&bytes).map_err(|error| WeightLoadError::Dequantization {
                    name: name.clone(),
                    reason: error.to_string(),
                })?
            }
        };
        weights.insert(name, TensorBlob::F32(values));
    }
    Ok(weights)
}

fn is_multimodal_tensor(name: &str) -> bool {
    ["vision.", "audio.", "image.", "mm.", "multimodal.", "clip."]
        .iter()
        .any(|prefix| name.starts_with(prefix))
}
