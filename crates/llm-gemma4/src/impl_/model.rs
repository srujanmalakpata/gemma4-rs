use llm_backend_cpu::{buffer::F32Buf, port::CpuPort};
use llm_eng::{
    compute::ComputePort,
    context_memory::{ContextMemory, KvCache},
    weight_store::{TensorBlob, WeightStore},
    EngError,
};
use llm_idea::{
    transformer::{LayerOp, TransformerSpec},
    types::{ActId, TokenId},
};

use crate::{
    eng::{GlobalPolicy, LocalWindowPolicy, kv_slot},
    idea::{AttentionKind, Gemma4HyperParams, layer_attention_kind},
};

const EPS: f32 = 1e-6;

#[derive(Default)]
pub struct CpuActivations {
    values: Vec<F32Buf>,
}

impl CpuActivations {
    fn push(&mut self, value: F32Buf) -> ActId {
        let id = ActId(self.values.len() as u32);
        self.values.push(value);
        id
    }

    fn get(&self, id: ActId) -> Result<&F32Buf, EngError> {
        self.values
            .get(id.0 as usize)
            .ok_or_else(|| EngError::Compute(format!("missing activation {}", id.0)))
    }
}

pub struct Gemma4CpuModel<C = KvCache<Vec<f32>, Vec<f32>>> {
    hp: Gemma4HyperParams,
    weights: WeightStore,
    context: C,
    port: CpuPort,
    prefill_position: usize,
    last_logits: Vec<f32>,
}

impl<C> Gemma4CpuModel<C>
where
    C: ContextMemory<K = Vec<f32>, V = Vec<f32>> + Default,
{
    pub fn synthetic(seed: u64, hp: Gemma4HyperParams) -> Self {
        let mut state = seed;
        let mut weights = WeightStore::default();
        let d = hp.d_model;
        let mut insert = |name: String, len: usize| {
            weights.insert(
                name,
                TensorBlob::F32(
                    (0..len)
                        .map(|_| {
                            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                            ((state >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 0.2
                        })
                        .collect(),
                ),
            );
        };

        insert("tok_embeddings".into(), hp.vocab_size * d);
        insert("final_norm".into(), d);
        insert("lm_head".into(), d * hp.vocab_size);
        for layer in 0..hp.n_layers {
            for name in ["attn_norm", "q", "k", "v", "o", "ffn_norm", "gate", "up", "down"] {
                insert(
                    format!("layers.{layer}.{name}"),
                    if name.ends_with("norm") { d } else { d * d },
                );
            }
        }

        Self {
            hp,
            weights,
            context: C::default(),
            port: CpuPort,
            prefill_position: 0,
            last_logits: Vec::new(),
        }
    }

    pub fn clear_context(&mut self) {
        self.context.clear();
        self.prefill_position = 0;
        self.last_logits.clear();
    }

    pub fn finish_prefill_token(&mut self) {
        self.prefill_position += 1;
    }

    pub fn last_logits(&self) -> &[f32] {
        &self.last_logits
    }

    fn tensor(&self, name: &str) -> Result<F32Buf, EngError> {
        match self.weights.get(name)? {
            TensorBlob::F32(values) => Ok(values.clone().into()),
            TensorBlob::Q8 { .. } => Err(EngError::Compute(format!("synthetic model requires f32 `{name}`"))),
        }
    }

    fn project(&self, x: &F32Buf, name: &str, out: usize) -> Result<F32Buf, EngError> {
        self.port.matmul(x, &self.tensor(name)?, 1, out, self.hp.d_model)
    }

    fn layer_position(&self, op: LayerOp) -> usize {
        match op {
            LayerOp::Prefill => self.prefill_position,
            LayerOp::Decode { position } => position,
        }
    }
}

impl<C> TransformerSpec for Gemma4CpuModel<C>
where
    C: ContextMemory<K = Vec<f32>, V = Vec<f32>> + Default,
{
    type Weights = ();
    type Activations = CpuActivations;
    type Error = EngError;

    fn embed(
        &mut self,
        _: &Self::Weights,
        tokens: &[TokenId],
        acts: &mut Self::Activations,
    ) -> Result<ActId, Self::Error> {
        let token = tokens
            .last()
            .ok_or_else(|| EngError::Msg("cannot embed an empty token sequence".into()))?
            .0 as usize;
        if token >= self.hp.vocab_size {
            return Err(EngError::Compute(format!("token {token} exceeds vocabulary")));
        }
        let table = self.tensor("tok_embeddings")?;
        let d = self.hp.d_model;
        Ok(acts.push(table.as_slice()[token * d..(token + 1) * d].to_vec().into()))
    }

    fn layer_count(&self, _: &Self::Weights) -> usize {
        self.hp.n_layers
    }

    fn layer(
        &mut self,
        _: &Self::Weights,
        layer: usize,
        x: ActId,
        acts: &mut Self::Activations,
        op: LayerOp,
    ) -> Result<ActId, Self::Error> {
        let d = self.hp.d_model;
        let position = self.layer_position(op);
        let input = acts.get(x)?.clone();
        let normalized = self.port.rms_norm(&input, &self.tensor(&format!("layers.{layer}.attn_norm"))?, EPS)?;
        let mut q = self.project(&normalized, &format!("layers.{layer}.q"), d)?;
        let mut k = self.project(&normalized, &format!("layers.{layer}.k"), d)?;
        let v = self.project(&normalized, &format!("layers.{layer}.v"), d)?;
        let (theta, rope_dims) = match layer_attention_kind(&self.hp, layer) {
            AttentionKind::Local { .. } => (self.hp.local_rope_base, d),
            AttentionKind::Global => {
                let dims = ((d as f32 * self.hp.global_p_rope.p) as usize / 2) * 2;
                (self.hp.global_p_rope.base, dims)
            }
        };
        if rope_dims > 0 {
            self.port.rope(&mut q, position, theta, rope_dims)?;
            self.port.rope(&mut k, position, theta, rope_dims)?;
        }

        let slot = kv_slot(layer, self.hp.n_layers, self.hp.kv_slots)
            .map_err(|error| EngError::Compute(error.to_string()))?;
        self.context.append(slot, k.as_slice().to_vec(), v.as_slice().to_vec())?;
        let (keys, values) = self.context.view(slot, position + 1)?;
        let range = match layer_attention_kind(&self.hp, layer) {
            AttentionKind::Local { window } => LocalWindowPolicy::new(window).range(keys.len()),
            AttentionKind::Global => GlobalPolicy.range(keys.len()),
        };
        let scale = (d as f32).sqrt().recip();
        let scores: F32Buf = keys[range.clone()]
            .iter()
            .map(|key| q.as_slice().iter().zip(key).map(|(a, b)| a * b).sum::<f32>() * scale)
            .collect::<Vec<_>>()
            .into();
        let probabilities = self.port.softmax(&scores)?;
        let mut attended = vec![0.0; d];
        for (probability, value) in probabilities.as_slice().iter().zip(&values[range]) {
            for (out, value) in attended.iter_mut().zip(value) {
                *out += probability * value;
            }
        }
        let attention = self.project(&attended.into(), &format!("layers.{layer}.o"), d)?;
        let residual = self.port.add(&input, &attention)?;
        let ffn_input = self.port.rms_norm(&residual, &self.tensor(&format!("layers.{layer}.ffn_norm"))?, EPS)?;
        let gate = self.project(&ffn_input, &format!("layers.{layer}.gate"), d)?;
        let up = self.project(&ffn_input, &format!("layers.{layer}.up"), d)?;
        let gated: F32Buf = gate
            .as_slice()
            .iter()
            .zip(up.as_slice())
            .map(|(&gate, &up)| (gate / (1.0 + (-gate).exp())) * up)
            .collect::<Vec<_>>()
            .into();
        let ffn = self.project(&gated, &format!("layers.{layer}.down"), d)?;
        Ok(acts.push(self.port.add(&residual, &ffn)?))
    }

    fn logits<'a>(
        &mut self,
        _: &Self::Weights,
        x: ActId,
        acts: &'a mut Self::Activations,
    ) -> Result<&'a [f32], Self::Error> {
        let normalized = self.port.rms_norm(acts.get(x)?, &self.tensor("final_norm")?, EPS)?;
        let logits = self.project(&normalized, "lm_head", self.hp.vocab_size)?;
        self.last_logits = logits.as_slice().to_vec();
        let logits = acts.push(logits);
        Ok(acts.get(logits)?.as_slice())
    }
}
