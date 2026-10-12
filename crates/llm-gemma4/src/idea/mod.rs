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
    pub n_heads: usize,
    pub n_kv_heads: usize,
    /// Width of each attention head on sliding-window layers.
    pub head_dim_local: usize,
    /// Width of each attention head on global layers.
    pub head_dim_global: usize,
    pub ple_dim: usize,
    pub sliding_window: usize,
    /// Number of final layers that reuse an earlier layer's K/V cache.
    pub shared_kv_layers: usize,
    pub local_rope_base: f32,
    pub global_p_rope: GlobalPRope,
    pub rms_norm_eps: f32,
    pub final_logit_softcap: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HyperParamError(pub String);

impl std::fmt::Display for HyperParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for HyperParamError {}

impl Gemma4HyperParams {
    pub fn e2b() -> Self {
        Self {
            n_layers: 35,
            vocab_size: 262_144,
            d_model: 1_536,
            n_heads: 8,
            n_kv_heads: 1,
            head_dim_local: 256,
            head_dim_global: 512,
            ple_dim: 256,
            sliding_window: 512,
            shared_kv_layers: 20,
            local_rope_base: 10_000.0,
            global_p_rope: GlobalPRope {
                p: 0.25,
                base: 1_000_000.0,
            },
            rms_norm_eps: 1e-6,
            final_logit_softcap: 30.0,
        }
    }

    /// Reads the architecture from GGUF metadata (llama.cpp's `gemma4.*`
    /// keys), keeping E2B defaults for keys a file omits, and checks the
    /// file's sliding-window layer pattern against [`layer_attention_kind`].
    pub fn from_gguf_metadata<R: std::io::Read + std::io::Seek>(
        gguf: &llm_gguf::reader::GgufFile<R>,
    ) -> Result<Self, HyperParamError> {
        let defaults = Self::e2b();
        let arch = gguf.metadata_str("general.architecture").unwrap_or("gemma4");
        let key = |suffix: &str| format!("{arch}.{suffix}");
        let usize_or = |suffix: &str, default: usize| gguf.metadata_usize(&key(suffix)).unwrap_or(default);
        let f32_or = |suffix: &str, default: f32| gguf.metadata_f32(&key(suffix)).unwrap_or(default);
        let hp = Self {
            n_layers: usize_or("block_count", defaults.n_layers),
            d_model: usize_or("embedding_length", defaults.d_model),
            vocab_size: gguf
                .metadata_strings("tokenizer.ggml.tokens")
                .map_or(defaults.vocab_size, <[String]>::len),
            n_heads: usize_or("attention.head_count", defaults.n_heads),
            n_kv_heads: usize_or("attention.head_count_kv", defaults.n_kv_heads),
            head_dim_local: usize_or("attention.key_length_swa", defaults.head_dim_local),
            head_dim_global: usize_or("attention.key_length", defaults.head_dim_global),
            ple_dim: usize_or("embedding_length_per_layer_input", defaults.ple_dim),
            sliding_window: usize_or("attention.sliding_window", defaults.sliding_window),
            shared_kv_layers: usize_or("attention.shared_kv_layers", defaults.shared_kv_layers),
            local_rope_base: f32_or("rope.freq_base_swa", defaults.local_rope_base),
            global_p_rope: GlobalPRope {
                base: f32_or("rope.freq_base", defaults.global_p_rope.base),
                ..defaults.global_p_rope
            },
            rms_norm_eps: f32_or("attention.layer_norm_rms_epsilon", defaults.rms_norm_eps),
            final_logit_softcap: f32_or("final_logit_softcapping", defaults.final_logit_softcap),
        };
        hp.validate()?;
        if let Some(pattern) = gguf.metadata_bools(&key("attention.sliding_window_pattern")) {
            let expected: Vec<bool> = (0..hp.n_layers)
                .map(|layer| matches!(layer_attention_kind(&hp, layer), AttentionKind::Local { .. }))
                .collect();
            if pattern != expected.as_slice() {
                return Err(HyperParamError(format!(
                    "sliding-window pattern {pattern:?} is not Gemma 4's 4 local : 1 global schedule"
                )));
            }
        }
        Ok(hp)
    }

    pub fn validate(&self) -> Result<(), HyperParamError> {
        let fail = |message: &str| Err(HyperParamError(message.to_owned()));
        if self.n_layers == 0 || self.d_model == 0 || self.vocab_size == 0 {
            return fail("layer count, model width and vocabulary must be non-zero");
        }
        if self.n_heads == 0 || self.n_kv_heads == 0 || !self.n_heads.is_multiple_of(self.n_kv_heads) {
            return fail("query heads must be a non-zero multiple of KV heads");
        }
        if self.head_dim_local == 0 || self.head_dim_global == 0 {
            return fail("head widths must be non-zero");
        }
        Ok(())
    }

    /// Width of each attention head on `layer`.
    pub fn head_dim(&self, layer: usize) -> usize {
        match layer_attention_kind(self, layer) {
            AttentionKind::Local { .. } => self.head_dim_local,
            AttentionKind::Global => self.head_dim_global,
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
