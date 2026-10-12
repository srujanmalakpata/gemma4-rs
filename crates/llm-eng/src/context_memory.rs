//! Both strategies retain appended K/V pairs in v1. The driver rebuilds
//! FullRecompute inputs on each pass; later engines may replace its storage.
//! Model code always attends over the slices returned by `view`; windowing
//! belongs in the engine through truncation or masking.

use crate::EngError;

/// Cached keys and values for one layer, oldest first.
pub type KvView<'a, K, V> = (&'a [K], &'a [V]);

pub trait ContextMemory {
    type K;
    type V;

    fn append(&mut self, layer: usize, k: Self::K, v: Self::V) -> Result<(), EngError>;
    fn view(&self, layer: usize, len: usize) -> Result<KvView<'_, Self::K, Self::V>, EngError>;
    fn clear(&mut self);
}

#[derive(Default)]
pub struct KvCache<K, V> {
    layers: Vec<Layer<K, V>>,
}

#[derive(Default)]
pub struct FullRecompute<K, V> {
    layers: Vec<Layer<K, V>>,
}

struct Layer<K, V> {
    keys: Vec<K>,
    values: Vec<V>,
}

impl<K, V> Default for Layer<K, V> {
    fn default() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
        }
    }
}

fn append<K, V>(layers: &mut Vec<Layer<K, V>>, layer: usize, k: K, v: V) -> Result<(), EngError> {
    if layers.len() <= layer {
        layers.resize_with(layer + 1, Layer::default);
    }
    layers[layer].keys.push(k);
    layers[layer].values.push(v);
    Ok(())
}

fn view<K, V>(layers: &[Layer<K, V>], layer: usize, len: usize) -> Result<KvView<'_, K, V>, EngError> {
    let Some(layer) = layers.get(layer) else {
        return if len == 0 {
            Ok((&[], &[]))
        } else {
            Err(EngError::MissingLayer(layer))
        };
    };

    if len > layer.keys.len() {
        return Err(EngError::ContextTooLong {
            got: len,
            max: layer.keys.len(),
        });
    }
    Ok((&layer.keys[..len], &layer.values[..len]))
}

impl<K, V> ContextMemory for KvCache<K, V> {
    type K = K;
    type V = V;

    fn append(&mut self, layer: usize, k: K, v: V) -> Result<(), EngError> {
        append(&mut self.layers, layer, k, v)
    }

    fn view(&self, layer: usize, len: usize) -> Result<KvView<'_, K, V>, EngError> {
        view(&self.layers, layer, len)
    }

    fn clear(&mut self) {
        self.layers.clear();
    }
}

impl<K, V> ContextMemory for FullRecompute<K, V> {
    type K = K;
    type V = V;

    fn append(&mut self, layer: usize, k: K, v: V) -> Result<(), EngError> {
        append(&mut self.layers, layer, k, v)
    }

    fn view(&self, layer: usize, len: usize) -> Result<KvView<'_, K, V>, EngError> {
        view(&self.layers, layer, len)
    }

    fn clear(&mut self) {
        self.layers.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_missing_layer_returns_missing_layer_error() {
        let cache = KvCache::<f32, f32>::default();
        assert!(matches!(
            cache.view(0, 1),
            Err(EngError::MissingLayer(0))
        ));
    }
}
