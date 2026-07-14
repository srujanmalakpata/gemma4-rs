use crate::types::{ActId, TokenId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerOp {
    Prefill,
    Decode { position: usize },
}

pub trait TransformerSpec {
    type Weights;
    type Activations;
    type Error;

    fn embed(
        &mut self,
        w: &Self::Weights,
        tokens: &[TokenId],
        acts: &mut Self::Activations,
    ) -> Result<ActId, Self::Error>;

    fn layer_count(&self, w: &Self::Weights) -> usize;

    fn layer(
        &mut self,
        w: &Self::Weights,
        layer: usize,
        x: ActId,
        acts: &mut Self::Activations,
        op: LayerOp,
    ) -> Result<ActId, Self::Error>;

    fn logits<'a>(
        &mut self,
        w: &Self::Weights,
        x: ActId,
        acts: &'a mut Self::Activations,
    ) -> Result<&'a [f32], Self::Error>;
}

pub fn prefill<E: TransformerSpec>(
    eng: &mut E,
    w: &E::Weights,
    tokens: &[TokenId],
    acts: &mut E::Activations,
) -> Result<Vec<f32>, E::Error> {
    let mut x = eng.embed(w, tokens, acts)?;

    for layer in 0..eng.layer_count(w) {
        x = eng.layer(w, layer, x, acts, LayerOp::Prefill)?;
    }

    Ok(eng.logits(w, x, acts)?.to_vec())
}

pub fn decode_step<E: TransformerSpec>(
    eng: &mut E,
    w: &E::Weights,
    token: TokenId,
    position: usize,
    acts: &mut E::Activations,
) -> Result<Vec<f32>, E::Error> {
    let mut x = eng.embed(w, &[token], acts)?;

    for layer in 0..eng.layer_count(w) {
        x = eng.layer(w, layer, x, acts, LayerOp::Decode { position })?;
    }

    Ok(eng.logits(w, x, acts)?.to_vec())
}
