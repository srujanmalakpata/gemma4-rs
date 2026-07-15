use llm_backend_cpu::{buffer::F32Buf, ops::rms_norm_gemma_f32, port::CpuPort};
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
    eng::{is_kv_owner, kv_slot, GlobalPolicy, LocalWindowPolicy},
    idea::{layer_attention_kind, AttentionKind, Gemma4HyperParams},
};

const EPS: f32 = 1e-6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadConfig {
    pub n_heads: usize,
    pub n_kv_heads: usize,
}

impl HeadConfig {
    fn validate(self, d_model: usize) -> Result<usize, EngError> {
        if self.n_heads == 0 || self.n_kv_heads == 0 || self.n_heads % self.n_kv_heads != 0
            || d_model % self.n_heads != 0
        {
            return Err(EngError::Compute("invalid GQA head configuration".into()));
        }
        Ok(d_model / self.n_heads)
    }

    /// Reads Gemma's GQA layout from GGUF metadata, with the Gemma 256-wide
    /// head convention as a fallback for old files that omit those keys.
    pub fn from_gguf_metadata<R: std::io::Read + std::io::Seek>(
        gguf: &llm_gguf::reader::GgufFile<R>,
        d_model: usize,
    ) -> Result<Self, EngError> {
        let arch = gguf.metadata_str("general.architecture").unwrap_or("gemma4");
        let key = |suffix: &str| format!("{arch}.{suffix}");
        let n_heads = gguf
            .metadata_usize(&key("attention.head_count"))
            .or_else(|| infer_head_count(gguf, "blk.0.attn_q.weight", d_model));
        let n_kv_heads = gguf
            .metadata_usize(&key("attention.head_count_kv"))
            .or_else(|| {
                gguf.metadata_u32s(&key("attention.head_count_kv"))
                    .and_then(|values| values.first().copied())
                    .map(|value| value as usize)
            })
            .or_else(|| infer_head_count(gguf, "blk.0.attn_k.weight", d_model));
        let heads = Self {
            n_heads: n_heads.ok_or_else(|| EngError::Compute("missing Gemma attention head count".into()))?,
            n_kv_heads: n_kv_heads
                .ok_or_else(|| EngError::Compute("missing Gemma KV attention head count".into()))?,
        };
        heads.validate(d_model)?;
        Ok(heads)
    }
}

fn infer_head_count<R: std::io::Read + std::io::Seek>(
    gguf: &llm_gguf::reader::GgufFile<R>,
    tensor_name: &str,
    d_model: usize,
) -> Option<usize> {
    let dimensions = gguf.tensor(tensor_name).ok()?.dimensions;
    let output = dimensions
        .iter()
        .copied()
        .find(|&dimension| dimension as usize != d_model)
        .unwrap_or(d_model as u64) as usize;
    // Gemma's published architectures use 256-wide attention heads. Tensor
    // dimensions alone cannot uniquely determine head count without that convention.
    output.checked_div(256).filter(|&count| count > 0)
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
    heads: HeadConfig,
    weights: WeightStore,
    context: C,
    port: CpuPort,
    prefill_position: usize,
    last_logits: Vec<f32>,
    eos_token_id: Option<u32>,
    current_token: Option<usize>,
}

impl<C> Gemma4CpuModel<C>
where
    C: ContextMemory<K = Vec<f32>, V = Vec<f32>> + Default,
{
    pub fn synthetic(seed: u64, hp: Gemma4HyperParams) -> Self {
        Self::synthetic_with_heads(seed, hp, HeadConfig { n_heads: 1, n_kv_heads: 1 })
    }

    pub fn synthetic_with_heads(seed: u64, hp: Gemma4HyperParams, heads: HeadConfig) -> Self {
        let mut state = seed;
        let mut weights = WeightStore::default();
        let d = hp.d_model;
        let head_dim = heads.validate(d).expect("synthetic head configuration");
        let kv_dim = heads.n_kv_heads * head_dim;
        let mut insert = |name: String, len: usize| {
            weights.insert(name, TensorBlob::F32((0..len).map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                ((state >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 0.2
            }).collect()));
        };
        insert("tok_embeddings".into(), hp.vocab_size * d);
        insert("final_norm".into(), d);
        insert("lm_head".into(), hp.vocab_size * d);
        for layer in 0..hp.n_layers {
            for name in ["attn_norm", "ffn_norm"] { insert(format!("layers.{layer}.{name}"), d); }
            for name in ["q", "o", "gate", "up", "down"] {
                insert(format!("layers.{layer}.{name}"), d * d);
            }
            for name in ["k", "v"] { insert(format!("layers.{layer}.{name}"), kv_dim * d); }
        }
        Self { hp, heads, weights, context: C::default(), port: CpuPort, prefill_position: 0, last_logits: Vec::new(), eos_token_id: None, current_token: None }
    }

    pub fn from_weights(weights: WeightStore, hp: Gemma4HyperParams, heads: HeadConfig) -> Result<Self, EngError> {
        heads.validate(hp.d_model)?;
        // Binding remains lazy so tied output embeddings work without copying tensors.
        Ok(Self { hp, heads, weights, context: C::default(), port: CpuPort, prefill_position: 0, last_logits: Vec::new(), eos_token_id: None, current_token: None })
    }

    pub fn with_eos_token_id(mut self, eos_token_id: Option<u32>) -> Self {
        self.eos_token_id = eos_token_id;
        self
    }
    pub(crate) fn eos_token_id(&self) -> Option<u32> { self.eos_token_id }
    pub fn clear_context(&mut self) { self.context.clear(); self.prefill_position = 0; self.last_logits.clear(); }
    pub fn finish_prefill_token(&mut self) { self.prefill_position += 1; }
    pub fn last_logits(&self) -> &[f32] { &self.last_logits }

    fn names(name: &str) -> Vec<String> {
        if let Some(rest) = name.strip_prefix("layers.") {
            let (layer, suffix) = rest.split_once('.').expect("layer name");
            let gguf = match suffix {
                "attn_norm" => "attn_norm.weight", "q" => "attn_q.weight", "k" => "attn_k.weight",
                "v" => "attn_v.weight", "o" => "attn_output.weight", "ffn_norm" => "ffn_norm.weight",
                "gate" => "ffn_gate.weight", "up" => "ffn_up.weight", "down" => "ffn_down.weight",
                "ple" => "ple.weight", _ => suffix,
            };
            vec![name.to_owned(), format!("blk.{layer}.{gguf}")]
        } else {
            match name {
                "tok_embeddings" => vec![name.into(), "token_embd.weight".into()],
                "final_norm" => vec![name.into(), "output_norm.weight".into()],
                "lm_head" => vec![name.into(), "output.weight".into(), "token_embd.weight".into()],
                _ => vec![name.into()],
            }
        }
    }
    fn tensor(&self, name: &str) -> Result<F32Buf, EngError> {
        for candidate in Self::names(name) {
            if let Ok(TensorBlob::F32(values)) = self.weights.get(&candidate) { return Ok(values.clone().into()); }
        }
        Err(EngError::MissingWeight(name.into()))
    }
    fn optional_tensor(&self, name: &str) -> Option<F32Buf> { self.tensor(name).ok() }
    /// GGUF linear tensors are row-major `[out, in]`; this computes `x @ Wᵀ`.
    fn project(&self, x: &[f32], name: &str, out: usize) -> Result<F32Buf, EngError> {
        let w = self.tensor(name)?;
        if w.len() != x.len() * out { return Err(EngError::Compute(format!("weight `{name}` has {}, expected {}x{}", w.len(), out, x.len()))); }
        Ok(w.as_slice().chunks_exact(x.len()).map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum()).collect::<Vec<f32>>().into())
    }
    fn layer_position(&self, op: LayerOp) -> usize { match op { LayerOp::Prefill => self.prefill_position, LayerOp::Decode { position } => position } }
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
        let table = self.tensor("tok_embeddings")?;
        let d = self.hp.d_model;
        Ok(acts.push(table.as_slice()[token * d..(token + 1) * d].to_vec().into()))
    }
    fn layer_count(&self, _: &()) -> usize { self.hp.n_layers }
    fn layer(&mut self, _: &(), layer: usize, x: ActId, acts: &mut CpuActivations, op: LayerOp) -> Result<ActId, EngError> {
        let d = self.hp.d_model;
        let head_dim = self.heads.validate(d)?;
        let kv_dim = self.heads.n_kv_heads * head_dim;
        let position = self.layer_position(op);
        let input = acts.get(x)?.clone();
        // Gemma RMSNorm uses its learned weights as a delta: effective scale is `1 + weight`.
        let normalized = rms_norm_gemma_f32(&input, &self.tensor(&format!("layers.{layer}.attn_norm"))?, EPS)?;
        let mut q = self.project(normalized.as_slice(), &format!("layers.{layer}.q"), d)?;
        let mut k = self.project(normalized.as_slice(), &format!("layers.{layer}.k"), kv_dim)?;
        let v = self.project(normalized.as_slice(), &format!("layers.{layer}.v"), kv_dim)?;
        let (theta, rope_dims) = match layer_attention_kind(&self.hp, layer) {
            AttentionKind::Local { .. } => (self.hp.local_rope_base, head_dim),
            AttentionKind::Global => (self.hp.global_p_rope.base, ((head_dim as f32 * self.hp.global_p_rope.p) as usize / 2) * 2),
        };
        for head in 0..self.heads.n_heads {
            let mut chunk: F32Buf = q.as_slice()[head * head_dim..(head + 1) * head_dim].to_vec().into();
            self.port.rope(&mut chunk, position, theta, rope_dims)?;
            q.as_mut_slice()[head * head_dim..(head + 1) * head_dim].copy_from_slice(chunk.as_slice());
        }
        for head in 0..self.heads.n_kv_heads {
            let mut chunk: F32Buf = k.as_slice()[head * head_dim..(head + 1) * head_dim].to_vec().into();
            self.port.rope(&mut chunk, position, theta, rope_dims)?;
            k.as_mut_slice()[head * head_dim..(head + 1) * head_dim].copy_from_slice(chunk.as_slice());
        }
        let slot = kv_slot(layer, self.hp.n_layers, self.hp.kv_slots).map_err(|e| EngError::Compute(e.to_string()))?;
        if is_kv_owner(layer, self.hp.n_layers, self.hp.kv_slots).map_err(|e| EngError::Compute(e.to_string()))? {
            self.context.append(slot, k.as_slice().to_vec(), v.as_slice().to_vec())?;
        }
        let (keys, values) = self.context.view(slot, position + 1)?;
        let range = match layer_attention_kind(&self.hp, layer) { AttentionKind::Local { window } => LocalWindowPolicy::new(window).range(keys.len()), AttentionKind::Global => GlobalPolicy.range(keys.len()) };
        let scale = (head_dim as f32).sqrt().recip();
        let mut attended = vec![0.0; d];
        for head in 0..self.heads.n_heads {
            let kv_head = head * self.heads.n_kv_heads / self.heads.n_heads;
            let qh = &q.as_slice()[head * head_dim..(head + 1) * head_dim];
            let scores: F32Buf = keys[range.clone()].iter().map(|key| qh.iter().zip(&key[kv_head * head_dim..(kv_head + 1) * head_dim]).map(|(a,b)| a*b).sum::<f32>() * scale).collect::<Vec<_>>().into();
            let probabilities = self.port.softmax(&scores)?;
            for (probability, value) in probabilities.as_slice().iter().zip(&values[range.clone()]) {
                for (out, value) in attended[head * head_dim..(head + 1) * head_dim].iter_mut().zip(&value[kv_head * head_dim..(kv_head + 1) * head_dim]) { *out += probability * value; }
            }
        }
        let attention = self.project(&attended, &format!("layers.{layer}.o"), d)?;
        let residual = self.port.add(&input, &attention)?;
        let ffn_input = rms_norm_gemma_f32(&residual, &self.tensor(&format!("layers.{layer}.ffn_norm"))?, EPS)?;
        let gate = self.tensor(&format!("layers.{layer}.gate"))?;
        let d_ff = gate.len() / d;
        let gate = self.project(ffn_input.as_slice(), &format!("layers.{layer}.gate"), d_ff)?;
        let up = self.project(ffn_input.as_slice(), &format!("layers.{layer}.up"), d_ff)?;
        let gated: Vec<f32> = gate.as_slice().iter().zip(up.as_slice()).map(|(&g, &u)| g / (1.0 + (-g).exp()) * u).collect();
        let ffn = self.project(&gated, &format!("layers.{layer}.down"), d)?;
        let mut output = self.port.add(&residual, &ffn)?;
        // ponytail: only accepts directly lookupable `[vocab, d_model]` PLE
        // tensors; add projection variants once a real GGUF exposes one.
        if self.hp.ple_dim > 0 {
            if let (Some(ple), Some(token)) = (self.optional_tensor(&format!("layers.{layer}.ple")), self.current_token) {
                if ple.len() == self.hp.vocab_size * d {
                    for (out, value) in output.as_mut_slice().iter_mut().zip(&ple.as_slice()[token * d..(token + 1) * d]) { *out += value; }
                }
            }
        }
        Ok(acts.push(output))
    }
    fn logits<'a>(&mut self, _: &(), x: ActId, acts: &'a mut CpuActivations) -> Result<&'a [f32], EngError> {
        let normalized = rms_norm_gemma_f32(acts.get(x)?, &self.tensor("final_norm")?, EPS)?;
        let logits = self.project(normalized.as_slice(), "lm_head", self.hp.vocab_size)?;
        self.last_logits = logits.as_slice().to_vec();
        let logits = acts.push(logits);
        Ok(acts.get(logits)?.as_slice())
    }
}
