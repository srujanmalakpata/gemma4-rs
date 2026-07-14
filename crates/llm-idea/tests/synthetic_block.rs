use llm_idea::transformer::{LayerOp, TransformerSpec};
use llm_idea::types::{ActId, TokenId};

/// Tiny driver that records the idea-layer call sequence.
struct TraceEngine {
    log: Vec<&'static str>,
}

struct TraceWeights;
struct TraceActs {
    logits: Vec<f32>,
}

impl TransformerSpec for TraceEngine {
    type Weights = TraceWeights;
    type Activations = TraceActs;
    type Error = String;

    fn embed(
        &mut self,
        _w: &Self::Weights,
        _tokens: &[TokenId],
        acts: &mut Self::Activations,
    ) -> Result<ActId, Self::Error> {
        self.log.push("embed");
        acts.logits = vec![0.0; 4];
        Ok(ActId(0))
    }

    fn layer_count(&self, _w: &Self::Weights) -> usize {
        1
    }

    fn layer(
        &mut self,
        _w: &Self::Weights,
        layer: usize,
        x: ActId,
        acts: &mut Self::Activations,
        op: LayerOp,
    ) -> Result<ActId, Self::Error> {
        let _ = (layer, x, acts, op);
        self.log.push("layer");
        Ok(ActId(0))
    }

    fn logits<'a>(
        &mut self,
        _w: &Self::Weights,
        x: ActId,
        acts: &'a mut Self::Activations,
    ) -> Result<&'a [f32], Self::Error> {
        let _ = x;
        self.log.push("logits");
        Ok(&acts.logits)
    }
}

#[test]
fn prefill_walks_embed_layers_logits_without_kv_hooks() {
    let mut eng = TraceEngine { log: vec![] };
    let w = TraceWeights;
    let mut acts = TraceActs { logits: vec![] };
    let tokens = [TokenId(1), TokenId(2)];

    let out = llm_idea::transformer::prefill(&mut eng, &w, &tokens, &mut acts).unwrap();

    assert_eq!(out.len(), 4);
    assert_eq!(eng.log, ["embed", "layer", "logits"]);
}
