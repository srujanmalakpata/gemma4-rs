//! Execution policies for Gemma 4's hybrid attention.
//!
//! `ContextMemory::view` currently exposes prefix slices, so callers fetch their
//! available context and apply these ranges when creating attention inputs.

use std::ops::Range;

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
pub enum KvSlotError {
    ZeroLayers,
    ZeroSlots,
    LayerOutOfRange { layer: usize, n_layers: usize },
}

impl std::fmt::Display for KvSlotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroLayers => write!(f, "n_layers must be non-zero"),
            Self::ZeroSlots => write!(f, "n_slots must be non-zero"),
            Self::LayerOutOfRange { layer, n_layers } => {
                write!(f, "layer {layer} must be within n_layers {n_layers}")
            }
        }
    }
}

impl std::error::Error for KvSlotError {}

/// Assigns layers to proportionally spaced KV-share groups.
///
/// For E2B, use `kv_slot(layer, 35, 20)`.
pub fn kv_slot(layer: usize, n_layers: usize, n_slots: usize) -> Result<usize, KvSlotError> {
    if n_layers == 0 {
        return Err(KvSlotError::ZeroLayers);
    }
    if n_slots == 0 {
        return Err(KvSlotError::ZeroSlots);
    }
    if layer >= n_layers {
        return Err(KvSlotError::LayerOutOfRange { layer, n_layers });
    }
    Ok(layer * n_slots / n_layers)
}

/// The lowest-numbered layer assigned to a slot writes that slot's K/V.
pub fn is_kv_owner(layer: usize, n_layers: usize, n_slots: usize) -> Result<bool, KvSlotError> {
    let slot = kv_slot(layer, n_layers, n_slots)?;
    Ok((0..layer).all(|other| kv_slot(other, n_layers, n_slots) != Ok(slot)))
}
