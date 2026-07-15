//! Gemma 4 E2B architecture choices, independent of execution details.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionKind {
    Local { window: usize },
    Global,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobalPRope {
    pub p: f32,
    pub base: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gemma4HyperParams {
    pub n_layers: usize,
    pub vocab_size: usize,
    pub d_model: usize,
    pub ple_dim: usize,
    pub sliding_window: usize,
    pub kv_slots: usize,
    pub local_rope_base: f32,
    pub global_p_rope: GlobalPRope,
}

impl Gemma4HyperParams {
    pub fn e2b() -> Self {
        Self {
            n_layers: 35,
            vocab_size: 262_144,
            d_model: 1_536,
            ple_dim: 256,
            sliding_window: 512,
            kv_slots: 20,
            local_rope_base: 10_000.0,
            global_p_rope: GlobalPRope {
                p: 0.25,
                base: 1_000_000.0,
            },
        }
    }

    /// Reads portable GGUF architecture metadata, retaining E2B defaults for
    /// optional fields that older conversions omit.
    pub fn from_gguf_metadata<R: std::io::Read + std::io::Seek>(
        gguf: &llm_gguf::reader::GgufFile<R>,
    ) -> Self {
        let defaults = Self::e2b();
        let arch = gguf.metadata_str("general.architecture").unwrap_or("gemma4");
        let key = |suffix: &str| format!("{arch}.{suffix}");
        Self {
            n_layers: gguf.metadata_usize(&key("block_count")).unwrap_or(defaults.n_layers),
            d_model: gguf.metadata_usize(&key("embedding_length")).unwrap_or(defaults.d_model),
            vocab_size: gguf.metadata_strings("tokenizer.ggml.tokens").map_or(defaults.vocab_size, <[String]>::len),
            sliding_window: gguf.metadata_usize(&key("sliding_window")).unwrap_or(defaults.sliding_window),
            kv_slots: gguf.metadata_usize(&key("shared_kv_layers")).unwrap_or(defaults.kv_slots),
            ..defaults
        }
    }
}

pub fn layer_attention_kind(hp: &Gemma4HyperParams, layer: usize) -> AttentionKind {
    if layer + 1 == hp.n_layers || layer % 5 == 4 {
        AttentionKind::Global
    } else {
        AttentionKind::Local {
            window: hp.sliding_window,
        }
    }
}

/// Extension point for adding the per-layer PLE residual during model execution.
///
/// Task 8 will define how `ple` is projected into `hidden`.
pub fn ple_residual(_hidden: &mut [f32], _ple: &[f32]) {}
