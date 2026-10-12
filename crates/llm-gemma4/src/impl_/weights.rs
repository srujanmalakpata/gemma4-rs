use std::{path::Path, sync::Arc};

use half::f16;
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
    #[error("unsupported tensor type for `{name}`")]
    UnsupportedTensor { name: String },
}

/// Loads text-model tensors into RAM in their stored formats: Q8_0 and BF16
/// matrices stay quantized, F32 and F16 tensors become f32. Multimodal
/// tensors are deliberately ignored.
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
        let info = gguf.tensor(&name)?;
        let ty = info.ty;
        // GGUF lists the fastest-varying dimension first: `[ncols, nrows, ...]`.
        let ncols = info.dimensions.first().copied().unwrap_or(1) as usize;
        let nrows = info.numel() / ncols.max(1);
        let bytes = gguf.read_tensor_bytes(&name)?;
        let blob = match ty {
            TensorType::F32 => TensorBlob::F32(Arc::new(
                bytes
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("exact chunk size")))
                    .collect(),
            )),
            TensorType::F16 => TensorBlob::F32(Arc::new(
                bytes
                    .chunks_exact(2)
                    .map(|bytes| {
                        f16::from_le_bytes(bytes.try_into().expect("exact chunk size")).to_f32()
                    })
                    .collect(),
            )),
            TensorType::BF16 => TensorBlob::Bf16 {
                data: Arc::new(
                    bytes
                        .chunks_exact(2)
                        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
                        .collect(),
                ),
                nrows,
                ncols,
            },
            TensorType::Q8_0 => TensorBlob::Q8 { data: Arc::new(bytes), nrows, ncols },
        };
        weights.insert(name, blob);
    }
    Ok(weights)
}

fn is_multimodal_tensor(name: &str) -> bool {
    ["vision.", "audio.", "image.", "mm.", "multimodal.", "clip."]
        .iter()
        .any(|prefix| name.starts_with(prefix))
}
