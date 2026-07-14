use crate::EngError;

pub trait ContextMemory {
    type K;
    type V;

    fn append(&mut self, layer: usize, k: Self::K, v: Self::V) -> Result<(), EngError>;
    fn view(&self, layer: usize, len: usize) -> Result<(&[Self::K], &[Self::V]), EngError>;
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

fn view<K, V>(layers: &[Layer<K, V>], layer: usize, len: usize) -> Result<(&[K], &[V]), EngError> {
    let Some(layer) = layers.get(layer) else {
        return if len == 0 {
            Ok((&[], &[]))
        } else {
            Err(EngError::ContextTooLong { got: len, max: 0 })
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

    fn view(&self, layer: usize, len: usize) -> Result<(&[K], &[V]), EngError> {
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

    fn view(&self, layer: usize, len: usize) -> Result<(&[K], &[V]), EngError> {
        view(&self.layers, layer, len)
    }

    fn clear(&mut self) {
        self.layers.clear();
    }
}

// Both strategies retain appended K/V pairs in v1. The driver rebuilds
// FullRecompute inputs on each pass; later engines may replace its storage.
// Model code always attends over the slices returned by `view`; windowing
// belongs in the engine through truncation or masking.
