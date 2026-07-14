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
