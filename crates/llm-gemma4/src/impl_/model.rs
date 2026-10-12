use llm_backend_cpu::{
    buffer::F32Buf,
    dequant::dequant_q8_0,
    matvec::{matvec_bf16, matvec_f32, matvec_q8_0, quantize_q8_0, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_VALUES},
    ops::{gelu_tanh, rms_norm, rope_neox},
    port::CpuPort,
};
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
    eng::{kv_source_layer, GlobalPolicy, LocalWindowPolicy},
    idea::{layer_attention_kind, proportional_rope_freq_factors, AttentionKind, Gemma4HyperParams},
};

const TOKEN_EMBD: &str = "token_embd.weight";
const OUTPUT: &str = "output.weight";
const OUTPUT_NORM: &str = "output_norm.weight";
/// Per-pair RoPE frequency divisors for global layers.
const ROPE_FREQS: &str = "rope_freqs.weight";
/// Per-layer embeddings (PLE): a token table with `ple_dim` values per
/// layer, a projection of the main embedding into the same shape, and the
/// norm applied to that projection.
const PLE_TOKEN_EMBD: &str = "per_layer_token_embd.weight";
const PLE_MODEL_PROJ: &str = "per_layer_model_proj.weight";
const PLE_PROJ_NORM: &str = "per_layer_proj_norm.weight";

/// GGUF name of a per-layer weight, e.g. `blk.3.attn_q.weight`.
fn blk(layer: usize, name: &str) -> String {
    format!("blk.{layer}.{name}.weight")
}

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
        self.values.get(id.0 as usize).ok_or_else(|| EngError::Compute(format!("missing activation {}", id.0)))
    }
}

pub struct Gemma4CpuModel<C = KvCache<Vec<f32>, Vec<f32>>> {
    hp: Gemma4HyperParams,
    weights: WeightStore,
    context: C,
    port: CpuPort,
    prefill_position: usize,
    last_logits: Vec<f32>,
    eos_token_ids: Vec<u32>,
    /// The current token's per-layer inputs, `n_layers` rows of `ple_dim`.
    per_layer_inputs: Vec<f32>,
}

impl<C> Gemma4CpuModel<C>
where
    C: ContextMemory<K = Vec<f32>, V = Vec<f32>> + Default,
{
    /// A model with small random f32 weights under the real GGUF tensor
    /// names, for tests.
    pub fn synthetic(seed: u64, hp: Gemma4HyperParams) -> Self {
        hp.validate().expect("synthetic hyperparameters");
        let mut state = seed;
        let mut weights = WeightStore::default();
        weights.insert(ROPE_FREQS, TensorBlob::F32(proportional_rope_freq_factors(hp.head_dim_global, hp.global_p_rope.p)));
        let d = hp.d_model;
        let d_ff = 2 * d;
        // Matrices get small values around 0, norm scales values around 1.
        let mut insert = |name: String, len: usize| {
            let center = if name.contains("norm") || name.contains("scale") { 1.0 } else { 0.0 };
            weights.insert(name, TensorBlob::F32((0..len).map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                center + ((state >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 0.2
            }).collect()));
        };
        insert(TOKEN_EMBD.into(), hp.vocab_size * d);
        insert(OUTPUT_NORM.into(), d);
        let ple_width = hp.n_layers * hp.ple_dim;
        if hp.ple_dim > 0 {
            insert(PLE_TOKEN_EMBD.into(), hp.vocab_size * ple_width);
            insert(PLE_MODEL_PROJ.into(), ple_width * d);
            insert(PLE_PROJ_NORM.into(), hp.ple_dim);
        }
        for layer in 0..hp.n_layers {
            if hp.ple_dim > 0 {
                insert(blk(layer, "inp_gate"), hp.ple_dim * d);
                insert(blk(layer, "proj"), d * hp.ple_dim);
                insert(blk(layer, "post_norm"), d);
            }
            insert(blk(layer, "layer_output_scale"), 1);
            let head_dim = hp.head_dim(layer);
            let (q_dim, kv_dim) = (hp.n_heads * head_dim, hp.n_kv_heads * head_dim);
            for name in ["attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm"] { insert(blk(layer, name), d); }
            for name in ["attn_q_norm", "attn_k_norm"] { insert(blk(layer, name), head_dim); }
            insert(blk(layer, "attn_q"), q_dim * d);
            insert(blk(layer, "attn_k"), kv_dim * d);
            insert(blk(layer, "attn_v"), kv_dim * d);
            insert(blk(layer, "attn_output"), d * q_dim);
            insert(blk(layer, "ffn_gate"), d_ff * d);
            insert(blk(layer, "ffn_up"), d_ff * d);
            insert(blk(layer, "ffn_down"), d * d_ff);
        }
        Self::new(weights, hp)
    }

    pub fn from_weights(weights: WeightStore, hp: Gemma4HyperParams) -> Result<Self, EngError> {
        hp.validate().map_err(|error| EngError::Compute(error.to_string()))?;
        Ok(Self::new(weights, hp))
    }

    fn new(weights: WeightStore, hp: Gemma4HyperParams) -> Self {
        Self { hp, weights, context: C::default(), port: CpuPort, prefill_position: 0, last_logits: Vec::new(), eos_token_ids: Vec::new(), per_layer_inputs: Vec::new() }
    }

    /// Tokens that stop generation (`<eos>`, and `<turn|>` for chat).
    pub fn with_eos_token_ids(mut self, eos_token_ids: impl IntoIterator<Item = u32>) -> Self {
        self.eos_token_ids = eos_token_ids.into_iter().collect();
        self
    }
    pub(crate) fn is_eos(&self, token: u32) -> bool { self.eos_token_ids.contains(&token) }
    pub fn clear_context(&mut self) { self.context.clear(); self.prefill_position = 0; self.last_logits.clear(); }
    pub fn finish_prefill_token(&mut self) { self.prefill_position += 1; }
    pub fn last_logits(&self) -> &[f32] { &self.last_logits }

    /// Borrows a weight in its stored format; weights can be gigabytes, so
    /// they are never copied.
    fn blob(&self, name: &str) -> Result<&TensorBlob, EngError> {
        self.weights.get(name)
    }
    /// Borrows an f32 weight (norm scales and synthetic test weights).
    fn tensor(&self, name: &str) -> Result<&[f32], EngError> {
        match self.blob(name)? {
            TensorBlob::F32(values) => Ok(values),
            _ => Err(EngError::Compute(format!("weight `{name}` is not stored as f32"))),
        }
    }
    fn optional_tensor(&self, name: &str) -> Option<&[f32]> { self.tensor(name).ok() }
    /// Number of output rows of a linear weight.
    fn rows(&self, name: &str, cols: usize) -> Result<usize, EngError> {
        Ok(match self.blob(name)? {
            TensorBlob::F32(values) => values.len() / cols,
            TensorBlob::Q8 { nrows, .. } | TensorBlob::Bf16 { nrows, .. } => *nrows,
        })
    }
    /// One row of a matrix as f32, e.g. a token's embedding.
    fn row(&self, name: &str, index: usize, cols: usize) -> Result<Vec<f32>, EngError> {
        match self.blob(name)? {
            TensorBlob::F32(values) => values
                .get(index * cols..(index + 1) * cols)
                .map(<[f32]>::to_vec)
                .ok_or_else(|| EngError::Compute(format!("row {index} is outside `{name}`"))),
            TensorBlob::Q8 { data, nrows, ncols } if *ncols == cols && index < *nrows => {
                let row_bytes = cols / Q8_0_BLOCK_VALUES * Q8_0_BLOCK_BYTES;
                dequant_q8_0(&data[index * row_bytes..(index + 1) * row_bytes])
                    .map_err(|error| EngError::Compute(error.to_string()))
            }
            _ => Err(EngError::Compute(format!("cannot read row {index} of `{name}`"))),
        }
    }
    /// GGUF linear tensors are row-major `[out, in]`; this computes `W @ x`.
    fn project(&self, x: &[f32], name: &str, out: usize) -> Result<F32Buf, EngError> {
        let shape_error = || EngError::Compute(format!("weight `{name}` is not {out}x{}", x.len()));
        let values = match self.blob(name)? {
            TensorBlob::F32(w) => matvec_f32(w, out, x).map_err(|_| shape_error())?,
            TensorBlob::Q8 { data, nrows, ncols } => {
                if (*nrows, *ncols) != (out, x.len()) { return Err(shape_error()); }
                matvec_q8_0(data, out, &quantize_q8_0(x)?)?
            }
            TensorBlob::Bf16 { data, nrows, ncols } => {
                if (*nrows, *ncols) != (out, x.len()) { return Err(shape_error()); }
                matvec_bf16(data, out, x)?
            }
        };
        Ok(values.into())
    }
    /// The output head; Gemma ties it to the token embeddings.
    fn output_name(&self) -> &'static str {
        if self.weights.get(OUTPUT).is_ok() { OUTPUT } else { TOKEN_EMBD }
    }
    fn layer_position(&self, op: LayerOp) -> usize { match op { LayerOp::Prefill => self.prefill_position, LayerOp::Decode { position } => position } }

    /// Gemma 4's per-layer inputs for `token`, given its scaled embedding:
    /// `(norm(W_proj @ embedding / sqrt(d_model)) + table[token] * sqrt(ple_dim)) / sqrt(2)`,
    /// with the norm applied to each layer's `ple_dim` slice.
    fn per_layer_inputs(&self, token: usize, embedding: &[f32]) -> Result<Vec<f32>, EngError> {
        let ple_dim = self.hp.ple_dim;
        let width = self.hp.n_layers * ple_dim;
        let projection_scale = (self.hp.d_model as f32).sqrt().recip();
        let projected = self.project(embedding, PLE_MODEL_PROJ, width)?;
        let table_scale = (ple_dim as f32).sqrt();
        let table_row = self.row(PLE_TOKEN_EMBD, token, width)?;
        let input_scale = 2.0_f32.sqrt().recip();
        let mut inputs = Vec::with_capacity(width);
        for (projected, table) in projected.as_slice().chunks_exact(ple_dim).zip(table_row.chunks_exact(ple_dim)) {
            let scaled: Vec<f32> = projected.iter().map(|value| value * projection_scale).collect();
            let normalized = self.norm(&scaled, PLE_PROJ_NORM)?;
            inputs.extend(normalized.as_slice().iter().zip(table).map(|(p, t)| (p + t * table_scale) * input_scale));
        }
        Ok(inputs)
    }

    /// RMSNorm with a stored weight.
    fn norm(&self, x: &[f32], name: &str) -> Result<F32Buf, EngError> {
        Ok(rms_norm(x, Some(self.tensor(name)?), self.hp.rms_norm_eps)?.into())
    }

    /// Normalizes each `head_dim`-wide head of `x` on its own (Gemma 4's
    /// q/k/v norms), scaled by the weight `name` if given.
    fn norm_heads(&self, x: &mut [f32], head_dim: usize, name: Option<&str>) -> Result<(), EngError> {
        let weight = name.map(|name| self.tensor(name)).transpose()?;
        for head in x.chunks_exact_mut(head_dim) {
            let normalized = rms_norm(head, weight, self.hp.rms_norm_eps)?;
            head.copy_from_slice(&normalized);
        }
        Ok(())
    }

    /// Applies NEOX RoPE over the full width of each `head_dim`-wide head
    /// in `x`. Global layers use the `rope_freqs` divisors (proportional
    /// RoPE); sliding-window layers rotate every pair.
    fn rope_heads(&self, x: &mut [f32], head_dim: usize, position: usize, kind: AttentionKind) -> Result<(), EngError> {
        let (base, factors) = match kind {
            AttentionKind::Local { .. } => (self.hp.local_rope_base, None),
            AttentionKind::Global => (self.hp.global_p_rope.base, Some(self.tensor(ROPE_FREQS)?)),
        };
        for head in x.chunks_exact_mut(head_dim) {
            rope_neox(head, position, base, head_dim, factors)?;
        }
        Ok(())
    }
}

/// Grouped-query attention for one query position: each query head attends
/// over `keys`/`values` (one entry per position, `n_kv_heads * head_dim`
/// wide) through the KV head it shares with `n_heads / n_kv_heads` others.
fn attend(
    port: &CpuPort,
    q: &[f32],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    n_heads: usize,
    n_kv_heads: usize,
    scale: f32,
) -> Result<Vec<f32>, EngError> {
    let head_dim = q.len() / n_heads;
    let mut attended = vec![0.0; q.len()];
    for (head, (qh, out)) in q.chunks_exact(head_dim).zip(attended.chunks_exact_mut(head_dim)).enumerate() {
        let kv = head * n_kv_heads / n_heads * head_dim..(head * n_kv_heads / n_heads + 1) * head_dim;
        let scores: F32Buf = keys
            .iter()
            .map(|key| qh.iter().zip(&key[kv.clone()]).map(|(a, b)| a * b).sum::<f32>() * scale)
            .collect::<Vec<_>>()
            .into();
        let probabilities = port.softmax(&scores)?;
        for (probability, value) in probabilities.as_slice().iter().zip(values) {
            for (out, value) in out.iter_mut().zip(&value[kv.clone()]) { *out += probability * value; }
        }
    }
    Ok(attended)
}

impl<C> TransformerSpec for Gemma4CpuModel<C>
where
    C: ContextMemory<K = Vec<f32>, V = Vec<f32>> + Default,
{
    type Weights = ();
    type Activations = CpuActivations;
    type Error = EngError;

    fn embed(&mut self, _: &(), tokens: &[TokenId], acts: &mut CpuActivations) -> Result<ActId, EngError> {
        let token = tokens.last().ok_or_else(|| EngError::Msg("cannot embed an empty token sequence".into()))?.0 as usize;
        if token >= self.hp.vocab_size { return Err(EngError::Compute(format!("token {token} exceeds vocabulary"))); }
        // Gemma scales embeddings by sqrt(d_model) before the first layer.
        let scale = (self.hp.d_model as f32).sqrt();
        let embedding: Vec<f32> = self.row(TOKEN_EMBD, token, self.hp.d_model)?.iter().map(|value| value * scale).collect();
        self.per_layer_inputs = if self.hp.ple_dim > 0 { self.per_layer_inputs(token, &embedding)? } else { Vec::new() };
        Ok(acts.push(embedding.into()))
    }
    fn layer_count(&self, _: &()) -> usize { self.hp.n_layers }
    fn layer(&mut self, _: &(), layer: usize, x: ActId, acts: &mut CpuActivations, op: LayerOp) -> Result<ActId, EngError> {
        let d = self.hp.d_model;
        let (n_heads, n_kv_heads) = (self.hp.n_heads, self.hp.n_kv_heads);
        let head_dim = self.hp.head_dim(layer);
        let kind = layer_attention_kind(&self.hp, layer);
        let position = self.layer_position(op);
        let input = acts.get(x)?.clone();
        let normalized = self.norm(input.as_slice(), &blk(layer, "attn_norm"))?;
        let mut q = self.project(normalized.as_slice(), &blk(layer, "attn_q"), n_heads * head_dim)?;
        self.norm_heads(q.as_mut_slice(), head_dim, Some(&blk(layer, "attn_q_norm")))?;
        self.rope_heads(q.as_mut_slice(), head_dim, position, kind)?;

        let source = kv_source_layer(&self.hp, layer).map_err(|e| EngError::Compute(e.to_string()))?;
        if source == layer {
            let mut k = self.project(normalized.as_slice(), &blk(layer, "attn_k"), n_kv_heads * head_dim)?;
            let mut v = self.project(normalized.as_slice(), &blk(layer, "attn_v"), n_kv_heads * head_dim)?;
            self.norm_heads(k.as_mut_slice(), head_dim, Some(&blk(layer, "attn_k_norm")))?;
            self.norm_heads(v.as_mut_slice(), head_dim, None)?;
            self.rope_heads(k.as_mut_slice(), head_dim, position, kind)?;
            self.context.append(layer, k.as_slice().to_vec(), v.as_slice().to_vec())?;
        }
        let (keys, values) = self.context.view(source, position + 1)?;
        let range = match kind { AttentionKind::Local { window } => LocalWindowPolicy::new(window).range(keys.len()), AttentionKind::Global => GlobalPolicy.range(keys.len()) };
        // q and k are RMS-normalized, so Gemma 4 uses no 1/sqrt(head_dim) scaling.
        let attended = attend(&self.port, q.as_slice(), &keys[range.clone()], &values[range], n_heads, n_kv_heads, 1.0)?;
        let attention = self.project(&attended, &blk(layer, "attn_output"), d)?;
        let attention = self.norm(attention.as_slice(), &blk(layer, "post_attention_norm"))?;
        let residual = self.port.add(&input, &attention)?;

        let ffn_input = self.norm(residual.as_slice(), &blk(layer, "ffn_norm"))?;
        let d_ff = self.rows(&blk(layer, "ffn_gate"), d)?;
        let gate = self.project(ffn_input.as_slice(), &blk(layer, "ffn_gate"), d_ff)?;
        let up = self.project(ffn_input.as_slice(), &blk(layer, "ffn_up"), d_ff)?;
        let gated: Vec<f32> = gate.as_slice().iter().zip(up.as_slice()).map(|(&g, &u)| gelu_tanh(g) * u).collect();
        let ffn = self.project(&gated, &blk(layer, "ffn_down"), d)?;
        let ffn = self.norm(ffn.as_slice(), &blk(layer, "post_ffw_norm"))?;
        let mut output = self.port.add(&residual, &ffn)?;

        // Per-layer embedding: gate the hidden state down to `ple_dim`, mix in
        // this layer's slice of the token's per-layer input, project back up.
        let ple_dim = self.hp.ple_dim;
        if ple_dim > 0 {
            let gate = self.project(output.as_slice(), &blk(layer, "inp_gate"), ple_dim)?;
            let layer_input = &self.per_layer_inputs[layer * ple_dim..(layer + 1) * ple_dim];
            let gated: Vec<f32> = gate.as_slice().iter().zip(layer_input).map(|(&g, &input)| gelu_tanh(g) * input).collect();
            let projected = self.project(&gated, &blk(layer, "proj"), d)?;
            let projected = self.norm(projected.as_slice(), &blk(layer, "post_norm"))?;
            output = self.port.add(&output, &projected)?;
        }
        if let Some(&[scale]) = self.optional_tensor(&blk(layer, "layer_output_scale")) {
            output.as_mut_slice().iter_mut().for_each(|value| *value *= scale);
        }
        Ok(acts.push(output))
    }
    fn logits<'a>(&mut self, _: &(), x: ActId, acts: &'a mut CpuActivations) -> Result<&'a [f32], EngError> {
        let normalized = self.norm(acts.get(x)?.as_slice(), OUTPUT_NORM)?;
        let logits = self.project(normalized.as_slice(), self.output_name(), self.hp.vocab_size)?;
        // Final logit soft-capping: cap * tanh(logit / cap).
        let cap = self.hp.final_logit_softcap;
        let logits: F32Buf = logits.as_slice().iter().map(|logit| (logit * cap.recip()).tanh() * cap).collect::<Vec<_>>().into();
        self.last_logits = logits.as_slice().to_vec();
        let logits = acts.push(logits);
        Ok(acts.get(logits)?.as_slice())
    }
}
