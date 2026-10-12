use llm_backend_cpu::{
    buffer::F32Buf,
    dequant::dequant_q8_0,
    matvec::{matvec_bf16, matvec_f32, matvec_q8_0, quantize_q8_0, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_VALUES},
    ops::rms_norm_gemma_f32,
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
    idea::{layer_attention_kind, AttentionKind, Gemma4HyperParams},
};

const EPS: f32 = 1e-6;

const TOKEN_EMBD: &str = "token_embd.weight";
const OUTPUT: &str = "output.weight";
const OUTPUT_NORM: &str = "output_norm.weight";

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
    current_token: Option<usize>,
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
        let d = hp.d_model;
        let d_ff = 2 * d;
        let mut insert = |name: String, len: usize| {
            weights.insert(name, TensorBlob::F32((0..len).map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                ((state >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 0.2
            }).collect()));
        };
        insert(TOKEN_EMBD.into(), hp.vocab_size * d);
        insert(OUTPUT_NORM.into(), d);
        for layer in 0..hp.n_layers {
            let head_dim = hp.head_dim(layer);
            let (q_dim, kv_dim) = (hp.n_heads * head_dim, hp.n_kv_heads * head_dim);
            for name in ["attn_norm", "ffn_norm"] { insert(blk(layer, name), d); }
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
        Self { hp, weights, context: C::default(), port: CpuPort, prefill_position: 0, last_logits: Vec::new(), eos_token_ids: Vec::new(), current_token: None }
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

    /// Applies RoPE to each `head_dim`-wide head in `x`.
    fn rope_heads(&self, x: &mut [f32], head_dim: usize, position: usize, theta: f32, rope_dims: usize) -> Result<(), EngError> {
        for head in x.chunks_exact_mut(head_dim) {
            let mut chunk: F32Buf = head.to_vec().into();
            self.port.rope(&mut chunk, position, theta, rope_dims)?;
            head.copy_from_slice(chunk.as_slice());
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
        self.current_token = Some(token);
        let embedding = self.row(TOKEN_EMBD, token, self.hp.d_model)?;
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
        // Gemma RMSNorm uses its learned weights as a delta: effective scale is `1 + weight`.
        let normalized = rms_norm_gemma_f32(&input, self.tensor(&blk(layer, "attn_norm"))?, EPS)?;
        let (theta, rope_dims) = match kind {
            AttentionKind::Local { .. } => (self.hp.local_rope_base, head_dim),
            AttentionKind::Global => (self.hp.global_p_rope.base, ((head_dim as f32 * self.hp.global_p_rope.p) as usize / 2) * 2),
        };
        let mut q = self.project(normalized.as_slice(), &blk(layer, "attn_q"), n_heads * head_dim)?;
        self.rope_heads(q.as_mut_slice(), head_dim, position, theta, rope_dims)?;

        let source = kv_source_layer(&self.hp, layer).map_err(|e| EngError::Compute(e.to_string()))?;
        if source == layer {
            let mut k = self.project(normalized.as_slice(), &blk(layer, "attn_k"), n_kv_heads * head_dim)?;
            let v = self.project(normalized.as_slice(), &blk(layer, "attn_v"), n_kv_heads * head_dim)?;
            self.rope_heads(k.as_mut_slice(), head_dim, position, theta, rope_dims)?;
            self.context.append(layer, k.as_slice().to_vec(), v.as_slice().to_vec())?;
        }
        let (keys, values) = self.context.view(source, position + 1)?;
        let range = match kind { AttentionKind::Local { window } => LocalWindowPolicy::new(window).range(keys.len()), AttentionKind::Global => GlobalPolicy.range(keys.len()) };
        let scale = (head_dim as f32).sqrt().recip();
        let attended = attend(&self.port, q.as_slice(), &keys[range.clone()], &values[range], n_heads, n_kv_heads, scale)?;
        let attention = self.project(&attended, &blk(layer, "attn_output"), d)?;
        let residual = self.port.add(&input, &attention)?;

        let ffn_input = rms_norm_gemma_f32(&residual, self.tensor(&blk(layer, "ffn_norm"))?, EPS)?;
        let d_ff = self.rows(&blk(layer, "ffn_gate"), d)?;
        let gate = self.project(ffn_input.as_slice(), &blk(layer, "ffn_gate"), d_ff)?;
        let up = self.project(ffn_input.as_slice(), &blk(layer, "ffn_up"), d_ff)?;
        let gated: Vec<f32> = gate.as_slice().iter().zip(up.as_slice()).map(|(&g, &u)| g / (1.0 + (-g).exp()) * u).collect();
        let ffn = self.project(&gated, &blk(layer, "ffn_down"), d)?;
        let mut output = self.port.add(&residual, &ffn)?;
        // ponytail: only accepts directly lookupable `[vocab, d_model]` PLE
        // tensors; add projection variants once a real GGUF exposes one.
        if self.hp.ple_dim > 0 {
            if let (Some(ple), Some(token)) = (self.optional_tensor(&blk(layer, "ple")), self.current_token) {
                if ple.len() == self.hp.vocab_size * d {
                    for (out, value) in output.as_mut_slice().iter_mut().zip(&ple[token * d..(token + 1) * d]) { *out += value; }
                }
            }
        }
        Ok(acts.push(output))
    }
    fn logits<'a>(&mut self, _: &(), x: ActId, acts: &'a mut CpuActivations) -> Result<&'a [f32], EngError> {
        let normalized = rms_norm_gemma_f32(acts.get(x)?, self.tensor(OUTPUT_NORM)?, EPS)?;
        let logits = self.project(normalized.as_slice(), self.output_name(), self.hp.vocab_size)?;
        self.last_logits = logits.as_slice().to_vec();
        let logits = acts.push(logits);
        Ok(acts.get(logits)?.as_slice())
    }
}
