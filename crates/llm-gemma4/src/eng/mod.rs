//! Execution policies for Gemma 4's hybrid attention.
//!
//! `ContextMemory::view` currently exposes prefix slices, so callers fetch their
//! available context and apply these ranges when creating attention inputs.

use std::ops::Range;

use crate::idea::{layer_attention_kind, Gemma4HyperParams};

pub fn windowed_len(seq_len: usize, window: usize) -> Range<usize> {
    seq_len.saturating_sub(window)..seq_len
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalWindowPolicy {
    window: usize,
}

impl LocalWindowPolicy {
    pub const fn new(window: usize) -> Self {
        Self { window }
    }

    pub fn range(self, seq_len: usize) -> Range<usize> {
        windowed_len(seq_len, self.window)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GlobalPolicy;

impl GlobalPolicy {
    pub fn range(self, seq_len: usize) -> Range<usize> {
        0..seq_len
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvShareError {
    LayerOutOfRange { layer: usize, n_layers: usize },
    TooManySharedLayers { shared: usize, n_layers: usize },
    NoEarlierLayerOfSameKind { layer: usize },
}

impl std::fmt::Display for KvShareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LayerOutOfRange { layer, n_layers } => {
                write!(f, "layer {layer} must be within n_layers {n_layers}")
            }
            Self::TooManySharedLayers { shared, n_layers } => {
                write!(f, "{shared} KV-shared layers leave no layer of {n_layers} to own a cache")
            }
            Self::NoEarlierLayerOfSameKind { layer } => {
                write!(f, "KV-shared layer {layer} has no earlier layer of the same attention kind")
            }
        }
    }
}

impl std::error::Error for KvShareError {}

/// Returns the layer whose K/V cache `layer` attends over.
///
/// Gemma 4's last `shared_kv_layers` layers compute no keys or values. Each
/// reuses the cache of the last earlier layer with the same attention kind;
/// for E2B (35 layers, 20 shared) layers 15-34 read layer 13 if they are
/// sliding-window layers and layer 14 if they are global. Every other layer
/// uses its own cache. Matches llama.cpp's Gemma 4 `layer_reuse_cb` and
/// transformers' `kv_shared_layer_index`.
pub fn kv_source_layer(hp: &Gemma4HyperParams, layer: usize) -> Result<usize, KvShareError> {
    let n_layers = hp.n_layers;
    if layer >= n_layers {
        return Err(KvShareError::LayerOutOfRange { layer, n_layers });
    }
    let first_shared = n_layers
        .checked_sub(hp.shared_kv_layers)
        .filter(|&first| first > 0)
        .ok_or(KvShareError::TooManySharedLayers { shared: hp.shared_kv_layers, n_layers })?;
    if layer < first_shared {
        return Ok(layer);
    }
    let kind = layer_attention_kind(hp, layer);
    (0..first_shared)
        .rev()
        .find(|&earlier| {
            std::mem::discriminant(&layer_attention_kind(hp, earlier)) == std::mem::discriminant(&kind)
        })
        .ok_or(KvShareError::NoEarlierLayerOfSameKind { layer })
}
