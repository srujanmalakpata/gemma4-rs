# LLM Inference Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a learnable Rust LLM inference engine that loads Gemma 4 E2B (text path, GGUF), runs CPU prefill+decode, and keeps mathematical transformer contracts free of KV-cache / device details.

**Architecture:** Three layers — `llm-idea` (math contracts, opaque handles), `llm-eng` (ports: ContextMemory, WeightStore, Schedule, ComputePort), and impl crates (`llm-gguf`, `llm-backend-cpu`, `llm-gemma4` mini-engine with recursive idea/eng/impl modules, `llm-cli`). KV vs full-recompute both implement `ContextMemory`; CPU is the first `ComputePort`.

**Tech Stack:** Rust 2024 edition (stable via devenv), Cargo workspace, SentencePiece tokenizer data from GGUF metadata, GGUF Q8_0 decode-to-f32 on use, RMSNorm / GQA / hybrid local-global attention / PLE / RoPE+p-RoPE as required by E2B.

**Locked decisions (were deferred in the design spec):**

| Topic | Choice |
|-------|--------|
| Bootstrap | `devenv` with Rust + cargo; managed toolchain only inside devenv |
| PLE residency | Always load PLE tables into RAM in v1 |
| Context cap | Default `max_context = 4096`; longer prompts return `Err` (no silent truncate) |
| Tokenizer | Tokens + merges/scores from GGUF `tokenizer.ggml.*` keys; chat template from Gemma 4 IT (`<\|turn\|>…`) |
| Pinned GGUF (optional golden) | `bartowski/google_gemma-4-E2B-it-GGUF` → `google_gemma-4-E2B-it-Q8_0.gguf` (~5GB); CI skips if absent |
| Quant path | v1: Q8_0 (+ F32/F16 if present) dequantized to f32 buffers on read; refuse other quants with a clear error |
| Sampling defaults | `temperature=1.0`, `top_p=0.95`, `top_k=64` |

**E2B idea params (text path):** 35 layers, vocab 262144, `d_model=1536`, PLE width 256, sliding window 512, local:global 4:1 (final layer global), KV-cache share 20/35, local RoPE base 10k, global p-RoPE `p=0.25` base 1M. Exact GQA head counts and tensor names are taken from the loaded GGUF metadata / tensor map (do not hard-fail if a published GGUF uses slightly different head names — bind by known key patterns).

---

## File structure

```text
devenv.nix / devenv.yaml / .gitignore (extend)
Cargo.toml                          # workspace
crates/
  llm-idea/                         # pure contracts
    src/lib.rs
    src/types.rs                    # TokenId, LayerId, handles
    src/transformer.rs              # TransformerSpec + layer loop
    src/attention.rs                # full-context attention contract
    src/ffn.rs
    tests/synthetic_block.rs
  llm-eng/                          # ports + schedule
    src/lib.rs
    src/context_memory.rs           # ContextMemory trait + KvCache + FullRecompute
    src/weight_store.rs
    src/schedule.rs                 # Prefill then decode
    src/compute.rs                  # ComputePort trait
    src/sample.rs                   # temp / top-p / top-k
    src/error.rs
    tests/kv_vs_recompute.rs
  llm-gguf/
    src/lib.rs
    src/header.rs
    src/tensor.rs
    src/reader.rs
    tests/header_parse.rs           # fixture: tiny synthetic gguf bytes
  llm-backend-cpu/
    src/lib.rs
    src/buffer.rs                   # f32 buffers
    src/ops.rs                      # matmul, rmsnorm, softmax, rope
    src/dequant.rs                  # Q8_0 → f32
    src/port.rs                     # ComputePort impl
    tests/ops_smoke.rs
  llm-gemma4/
    src/lib.rs
    src/idea/mod.rs                 # Gemma4Spec diffs (hybrid attn, PLE hooks)
    src/eng/mod.rs                  # wire ContextMemory policies (window vs global)
    src/impl_/mod.rs                # GGUF bind + tokenizer + generate
    src/impl_/tokenizer.rs
    src/impl_/chat_template.rs
    src/impl_/weights.rs
    src/impl_/generate.rs
    tests/chat_template.rs
  llm-cli/
    src/main.rs
    Cargo.toml
tests/
  (none at workspace root — keep tests next to crates)
models/                             # gitignored; local GGUF drops here
```

Dependency rule enforced by Cargo: `llm-idea` has zero deps on other workspace crates; `llm-eng` → `llm-idea`; impl crates → eng/idea as needed; `llm-cli` depends on all wiring crates.

---

### Task 1: devenv + Cargo workspace skeleton

**Files:**
- Create: `devenv.nix`, `devenv.yaml`, `Cargo.toml`, `crates/llm-idea/Cargo.toml`, `crates/llm-idea/src/lib.rs`, `crates/llm-eng/Cargo.toml`, `crates/llm-eng/src/lib.rs`, `crates/llm-gguf/Cargo.toml`, `crates/llm-gguf/src/lib.rs`, `crates/llm-backend-cpu/Cargo.toml`, `crates/llm-backend-cpu/src/lib.rs`, `crates/llm-gemma4/Cargo.toml`, `crates/llm-gemma4/src/lib.rs`, `crates/llm-cli/Cargo.toml`, `crates/llm-cli/src/main.rs`
- Modify: `.gitignore`

- [ ] **Step 1: Write minimal devenv**

`devenv.nix`:
```nix
{ pkgs, ... }:
{
  languages.rust.enable = true;
  languages.rust.channel = "stable";

  packages = with pkgs; [
    rust-analyzer
  ];

  enterShell = ''
    echo "llm-inference: Rust $(rustc --version)"
  '';
}
```

`devenv.yaml`:
```yaml
inputs:
  nixpkgs:
    url: github:cachix/devenv-nixpkgs/rolling
```

- [ ] **Step 2: Write workspace Cargo.toml and stub crates**

Root `Cargo.toml`:
```toml
[workspace]
resolver = "2"
members = [
  "crates/llm-idea",
  "crates/llm-eng",
  "crates/llm-gguf",
  "crates/llm-backend-cpu",
  "crates/llm-gemma4",
  "crates/llm-cli",
]

[workspace.package]
edition = "2021"
license = "MIT"
version = "0.1.0"

[workspace.dependencies]
llm-idea = { path = "crates/llm-idea" }
llm-eng = { path = "crates/llm-eng" }
llm-gguf = { path = "crates/llm-gguf" }
llm-backend-cpu = { path = "crates/llm-backend-cpu" }
llm-gemma4 = { path = "crates/llm-gemma4" }
thiserror = "2"
anyhow = "1"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
clap = { version = "4", features = ["derive"] }
```

Each library crate `Cargo.toml` uses `version.workspace = true`, `edition.workspace = true`. Start with empty `pub fn ping() -> &'static str { "ok" }` in each `lib.rs`. `llm-cli` is a `[[bin]]` with `fn main() { println!("llm-cli stub"); }`.

Extend `.gitignore`:
```
.superpowers/
target/
.DS_Store
*.gguf
.devenv/
.result/
models/
```

- [ ] **Step 3: Enter devenv and verify workspace builds**

Run: `devenv shell -- cargo build`
Expected: all crates compile.

- [ ] **Step 4: Commit**

```bash
git add devenv.nix devenv.yaml Cargo.toml crates .gitignore Cargo.lock
git commit -m "$(cat <<'EOF'
chore: bootstrap devenv and Cargo workspace crates

EOF
)"
```

---

### Task 2: `llm-idea` core types and `TransformerSpec`

**Files:**
- Create: `crates/llm-idea/src/types.rs`, `crates/llm-idea/src/transformer.rs`, `crates/llm-idea/src/attention.rs`, `crates/llm-idea/src/ffn.rs`
- Modify: `crates/llm-idea/src/lib.rs`
- Test: `crates/llm-idea/tests/synthetic_block.rs`

- [ ] **Step 1: Write the failing synthetic-block test**

```rust
// crates/llm-idea/tests/synthetic_block.rs
use llm_idea::transformer::{LayerOp, TransformerSpec};
use llm_idea::types::{ActId, TokenId, WeightId};

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
        tokens: &[TokenId],
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

    fn logits(
        &mut self,
        _w: &Self::Weights,
        x: ActId,
        acts: &mut Self::Activations,
    ) -> Result<&[f32], Self::Error> {
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `devenv shell -- cargo test -p llm-idea --test synthetic_block -- --nocapture`
Expected: FAIL (missing modules / traits).

- [ ] **Step 3: Implement idea types + transformer loop (no KV, no devices)**

`types.rs`:
```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TokenId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LayerId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ActId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightId(pub u32);
```

`transformer.rs` — define `LayerOp { Prefill, Decode { position: usize } }` and trait:

```rust
pub trait TransformerSpec {
    type Weights;
    type Activations;
    type Error;
    fn embed(&mut self, w: &Self::Weights, tokens: &[TokenId], acts: &mut Self::Activations)
        -> Result<ActId, Self::Error>;
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
```

Add stub `attention.rs` / `ffn.rs` documenting that attention is over **full supplied context** at the idea layer (windowing is an eng concern via what context is supplied). Export from `lib.rs`. **Do not** mention KV cache in this crate.

- [ ] **Step 4: Run test to verify it passes**

Run: `devenv shell -- cargo test -p llm-idea --test synthetic_block`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/llm-idea
git commit -m "$(cat <<'EOF'
feat(llm-idea): add TransformerSpec prefill/decode without KV or devices

EOF
)"
```

---

### Task 3: `llm-eng` ports — ContextMemory, WeightStore, Schedule, ComputePort

**Files:**
- Create: `crates/llm-eng/src/error.rs`, `crates/llm-eng/src/compute.rs`, `crates/llm-eng/src/weight_store.rs`, `crates/llm-eng/src/context_memory.rs`, `crates/llm-eng/src/schedule.rs`, `crates/llm-eng/src/sample.rs`
- Modify: `crates/llm-eng/Cargo.toml`, `crates/llm-eng/src/lib.rs`
- Test: `crates/llm-eng/tests/kv_vs_recompute.rs`

- [ ] **Step 1: Write failing KV-vs-recompute test (numeric fixture)**

```rust
// crates/llm-eng/tests/kv_vs_recompute.rs
use llm_eng::context_memory::{ContextMemory, FullRecompute, KvCache};
use llm_eng::compute::ComputePort;

/// Toy attention: score = q·k, output = score * v (scalars).
fn toy_attend(q: f32, ks: &[f32], vs: &[f32]) -> f32 {
    assert_eq!(ks.len(), vs.len());
    let mut acc = 0.0;
    for i in 0..ks.len() {
        acc += (q * ks[i]) * vs[i];
    }
    acc
}

#[test]
fn kv_cache_matches_full_recompute_on_scalar_fixture() {
    let qs = [1.0_f32, 2.0, 3.0];
    let ks = [0.5_f32, -1.0, 2.0];
    let vs = [1.0_f32, 2.0, 0.5];

    let mut recompute = FullRecompute::<f32, f32>::default();
    let mut kv = KvCache::<f32, f32>::default();

    let mut out_r = Vec::new();
    let mut out_k = Vec::new();
    for t in 0..3 {
        recompute.append(0, ks[t], vs[t]).unwrap();
        kv.append(0, ks[t], vs[t]).unwrap();
        let (k_r, v_r) = recompute.view(0, t + 1).unwrap();
        let (k_k, v_k) = kv.view(0, t + 1).unwrap();
        out_r.push(toy_attend(qs[t], k_r, v_r));
        out_k.push(toy_attend(qs[t], k_k, v_k));
    }
    for i in 0..3 {
        assert!((out_r[i] - out_k[i]).abs() < 1e-6);
    }
}
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `devenv shell -- cargo test -p llm-eng --test kv_vs_recompute`
Expected: FAIL (modules missing).

- [ ] **Step 3: Implement eng ports**

`error.rs`:
```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngError {
    #[error("context length {got} exceeds configured max {max}")]
    ContextTooLong { got: usize, max: usize },
    #[error("missing weight `{0}`")]
    MissingWeight(String),
    #[error("compute: {0}")]
    Compute(String),
    #[error("{0}")]
    Msg(String),
}
```

`context_memory.rs` — trait + both strategies:

```rust
pub trait ContextMemory {
    type K;
    type V;
    fn append(&mut self, layer: usize, k: Self::K, v: Self::V) -> Result<(), EngError>;
    fn view(&self, layer: usize, len: usize) -> Result<(&[Self::K], &[Self::V]), EngError>;
    fn clear(&mut self);
}

#[derive(Default)]
pub struct KvCache<K, V> {
    layers: Vec<(Vec<K>, Vec<V>)>,
}

#[derive(Default)]
pub struct FullRecompute<K, V> {
    // Same storage shape in v1; semantics differ at the compute driver:
    // FullRecompute callers must rebuild K/V from scratch each step when used
    // for correctness tests. For the scalar fixture, storage equality is enough
    // to prove the port swap.
    layers: Vec<(Vec<K>, Vec<V>)>,
}
```

Implement `append`/`view`/`clear` for both (grow `layers` on demand). Document in module docs: idea always attends over whatever `view` returns; windowing = eng truncating or masking the view for local layers.

`compute.rs`:
```rust
pub trait ComputePort {
    type Buffer;
    fn matmul(&self, a: &Self::Buffer, b: &Self::Buffer, m: usize, n: usize, k: usize)
        -> Result<Self::Buffer, EngError>;
    fn rms_norm(&self, x: &Self::Buffer, weight: &Self::Buffer, eps: f32)
        -> Result<Self::Buffer, EngError>;
    fn softmax(&self, x: &Self::Buffer) -> Result<Self::Buffer, EngError>;
    fn rope(&self, x: &mut Self::Buffer, pos: usize, theta: f32, dims: usize)
        -> Result<(), EngError>;
    fn add(&self, a: &Self::Buffer, b: &Self::Buffer) -> Result<Self::Buffer, EngError>;
}
```

`weight_store.rs`: map `String → WeightHandle` trait object or `enum TensorBlob { F32(Vec<f32>), Q8 { … } }` with `get(name) -> Result<&TensorBlob, EngError>`.

`schedule.rs`:
```rust
pub enum Phase { Prefill, Decode { position: usize } }

pub struct SingleSequence {
    pub max_context: usize,
    pub phase: Phase,
}
```

`sample.rs`: implement `sample_logits(logits: &[f32], temperature, top_p, top_k, rng) -> u32` with the locked defaults available as `SamplingConfig::gemma4_defaults()`.

Wire exports in `lib.rs`. Add `thiserror` dependency.

- [ ] **Step 4: Run test — expect PASS**

Run: `devenv shell -- cargo test -p llm-eng --test kv_vs_recompute`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/llm-eng
git commit -m "$(cat <<'EOF'
feat(llm-eng): add ContextMemory, ComputePort, and KV vs recompute fixture

EOF
)"
```

---

### Task 4: `llm-gguf` reader (header + tensors + Q8_0 bytes)

**Files:**
- Create: `crates/llm-gguf/src/header.rs`, `crates/llm-gguf/src/tensor.rs`, `crates/llm-gguf/src/reader.rs`, `crates/llm-gguf/tests/fixtures/tiny.gguf` (generated in test or checked in as bytes)
- Modify: `crates/llm-gguf/src/lib.rs`, `Cargo.toml`
- Test: `crates/llm-gguf/tests/header_parse.rs`

- [ ] **Step 1: Write failing header parse test using an in-test writer**

```rust
// crates/llm-gguf/tests/header_parse.rs
use llm_gguf::reader::GgufFile;
use std::io::Cursor;

#[test]
fn parses_magic_version_and_tensor_count_from_minimal_file() {
    let bytes = llm_gguf::testutil::write_minimal_q8_tensor("demo.weight", &[1i8, 2, 3, 4]);
    let file = GgufFile::open_read(Cursor::new(bytes)).unwrap();
    assert_eq!(file.tensor_names(), ["demo.weight"]);
    let t = file.tensor("demo.weight").unwrap();
    assert_eq!(t.numel(), 4);
}
```

Put `write_minimal_q8_tensor` behind `#[cfg(test)] pub mod testutil` in the crate so we do not depend on external fixture files yet.

- [ ] **Step 2: Run — expect FAIL**

Run: `devenv shell -- cargo test -p llm-gguf --test header_parse`
Expected: FAIL.

- [ ] **Step 3: Implement GGUF parser**

Follow [GGUF spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md):
- Magic `GGUF`, version u32, tensor_count u64, kv_count u64
- KV metadata (enough types for u32/u64/f32/bool/string/array)
- Tensor infos: name, ndims, dims, type, offset
- Align to 32 bytes before data blob

Support tensor types needed for v1: `F32`, `F16`, `Q8_0`. Map unknown types to `GgufError::UnsupportedTensorType { name, ty }` with context.

`GgufFile` API:
```rust
pub struct GgufFile<R> { /* … */ }
impl<R: Read + Seek> GgufFile<R> {
    pub fn open_read(reader: R) -> Result<Self, GgufError>;
    pub fn metadata_str(&self, key: &str) -> Option<&str>;
    pub fn metadata_u32(&self, key: &str) -> Option<u32>;
    pub fn tensor_names(&self) -> Vec<&str>;
    pub fn tensor(&self, name: &str) -> Result<TensorInfo<'_>, GgufError>;
    pub fn read_tensor_bytes(&mut self, name: &str) -> Result<Vec<u8>, GgufError>;
}
```

Fail fast with path/tensor/arch context — accept an optional `path: PathBuf` stored for Display.

- [ ] **Step 4: Run — expect PASS**

Run: `devenv shell -- cargo test -p llm-gguf`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/llm-gguf
git commit -m "$(cat <<'EOF'
feat(llm-gguf): parse GGUF header, metadata, and Q8_0/F32/F16 tensors

EOF
)"
```

---

### Task 5: `llm-backend-cpu` — f32 buffers + Q8_0 dequant + ComputePort

**Files:**
- Create: `crates/llm-backend-cpu/src/buffer.rs`, `crates/llm-backend-cpu/src/dequant.rs`, `crates/llm-backend-cpu/src/ops.rs`, `crates/llm-backend-cpu/src/port.rs`
- Modify: `crates/llm-backend-cpu/Cargo.toml`, `src/lib.rs`
- Test: `crates/llm-backend-cpu/tests/ops_smoke.rs`

- [ ] **Step 1: Write failing ops smoke tests**

```rust
use llm_backend_cpu::dequant::dequant_q8_0;
use llm_backend_cpu::ops::{matmul_f32, rms_norm_f32};
use llm_backend_cpu::buffer::F32Buf;

#[test]
fn matmul_2x2() {
    let a = F32Buf::from(vec![1.0, 2.0, 3.0, 4.0]); // 2x2 row-major
    let b = F32Buf::from(vec![5.0, 6.0, 7.0, 8.0]);
    let c = matmul_f32(&a, &b, 2, 2, 2).unwrap();
    assert_eq!(c.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn q8_0_block_dequant_roundtrip_scale() {
    // One Q8_0 block: f16 scale + 32 i8 values. Use scale=1.0 and ints 0..31.
    let mut raw = Vec::new();
    raw.extend_from_slice(&1.0f32.to_le_bytes()[..2]); // ponytail: write real f16 bits in impl test helper
    // Prefer a test helper that builds a valid block with half::f16::from_f32(1.0)
    let buf = dequant_q8_0(&valid_q8_block_scale1_sequential()).unwrap();
    assert_eq!(buf.len(), 32);
    assert!((buf[0] - 0.0).abs() < 1e-3);
    assert!((buf[31] - 31.0).abs() < 1e-2);
}
```

Add `half` crate dependency for f16.

- [ ] **Step 2: Run — expect FAIL**

Run: `devenv shell -- cargo test -p llm-backend-cpu --test ops_smoke`
Expected: FAIL.

- [ ] **Step 3: Implement CPU backend**

- `F32Buf(Vec<f32>)` with `as_slice` / `as_mut_slice` / `len`
- `dequant_q8_0(bytes) -> Vec<f32>`: block size 34 (2 + 32); scale = f16; each value `i8 as f32 * scale`
- Naive O(mnk) `matmul_f32` (mark `// ponytail: O(n³) f32 matmul — replace with blocked/SIMD or Metal port later`)
- `rms_norm_f32`, `softmax_f32`, `rope_f32` (even dims pair rotate), `add_f32`
- `CpuPort` implements `llm_eng::compute::ComputePort<Buffer = F32Buf>`

- [ ] **Step 4: Run — expect PASS**

Run: `devenv shell -- cargo test -p llm-backend-cpu`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/llm-backend-cpu
git commit -m "$(cat <<'EOF'
feat(llm-backend-cpu): f32 ComputePort with Q8_0 dequant

EOF
)"
```

---

### Task 6: `llm-gemma4` idea + eng refinements (hybrid attention + PLE hooks)

**Files:**
- Create: `crates/llm-gemma4/src/idea/mod.rs`, `crates/llm-gemma4/src/eng/mod.rs`
- Modify: `crates/llm-gemma4/Cargo.toml`, `src/lib.rs`
- Test: `crates/llm-gemma4/tests/layer_pattern.rs`

- [ ] **Step 1: Write failing layer-pattern test**

```rust
use llm_gemma4::idea::{AttentionKind, Gemma4HyperParams, layer_attention_kind};

#[test]
fn e2b_uses_4_local_then_1_global_and_final_global() {
    let hp = Gemma4HyperParams::e2b();
    assert_eq!(hp.n_layers, 35);
    assert_eq!(hp.sliding_window, 512);
    // layers 0..3 local, 4 global, … last global
    assert_eq!(layer_attention_kind(&hp, 0), AttentionKind::Local { window: 512 });
    assert_eq!(layer_attention_kind(&hp, 3), AttentionKind::Local { window: 512 });
    assert_eq!(layer_attention_kind(&hp, 4), AttentionKind::Global);
    assert_eq!(layer_attention_kind(&hp, 34), AttentionKind::Global);
}
```

- [ ] **Step 2: Run — expect FAIL**

Run: `devenv shell -- cargo test -p llm-gemma4 --test layer_pattern`
Expected: FAIL.

- [ ] **Step 3: Implement idea/eng stubs**

`Gemma4HyperParams::e2b()` returns the locked constants. `layer_attention_kind`: for E2B, pattern repeats 4 local + 1 global; layer `n_layers-1` forced global.

`idea` module defines extension hooks used later by the concrete `TransformerSpec` impl:
- `ple_residual(layer, token_ids) -> ActId` (idea-level: extra residual contribution)
- attention kind selection

`eng` module:
- `LocalWindowPolicy` that, for local layers, returns `ContextMemory::view` truncated to last `window` positions (or errors if policy misconfigured)
- `GlobalPolicy` returns full view
- KV-share groups: document E2B 20/35 — implement shared slot map `layer -> kv_slot` where layers in a share group read/write the same slot (derive evenly from `share_ratio` or from GGUF metadata when present)

- [ ] **Step 4: Run — expect PASS**

Run: `devenv shell -- cargo test -p llm-gemma4 --test layer_pattern`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/llm-gemma4
git commit -m "$(cat <<'EOF'
feat(llm-gemma4): encode E2B hybrid attention pattern and eng policies

EOF
)"
```

---

### Task 7: GGUF weight binding + tokenizer + chat template

**Files:**
- Create: `crates/llm-gemma4/src/impl_/mod.rs`, `weights.rs`, `tokenizer.rs`, `chat_template.rs`
- Test: `crates/llm-gemma4/tests/chat_template.rs`

- [ ] **Step 1: Write failing chat-template test**

```rust
use llm_gemma4::impl_::chat_template::format_user_prompt;

#[test]
fn wraps_user_turn_with_gemma4_markers() {
    let s = format_user_prompt("Hello");
    assert!(s.contains("<|turn|>user"));
    assert!(s.contains("Hello"));
    assert!(s.contains("<turn|>"));
    assert!(s.contains("<|turn|>model"));
}
```

- [ ] **Step 2: Run — expect FAIL**

Expected: FAIL.

- [ ] **Step 3: Implement template + tokenizer + weight bind**

Chat template (IT, no thinking by default):
```text
<|turn|>user
{prompt}<turn|>
<|turn|>model
```
Document that callers must add BOS via tokenizer `add_bos=true`, never as literal `"[BOS]"` text (per Gemma 4 table 11).

Tokenizer:
- Read `tokenizer.ggml.tokens`, `tokenizer.ggml.scores`, `tokenizer.ggml.token_type` from GGUF metadata when present
- Implement greedy SentencePiece-style longest-match encode using the score table (keep it simple; mark `// ponytail: greedy SP decode — upgrade to full SentencePiece proto if mismatch vs golden`)
- Decode: byte-fallback aware detokenize joining tokens

Weights:
- `load_gemma4_e2b(path) -> WeightStore`:
  - Open GGUF
  - Require arch metadata matches gemma4 / e2b-ish (`general.architecture` contains expected string; else `Err`)
  - Skip multimodal tensor name prefixes (`vision.`, `audio.`, etc.)
  - For each text tensor: if Q8_0/F16/F32, dequant/store as f32 in RAM (PLE included); unsupported type → hard error naming the tensor
  - Cap: if estimated RAM absurdly high, still OK for Q8_0 E2B on 16GB — no paging

- [ ] **Step 4: Run unit tests**

Run: `devenv shell -- cargo test -p llm-gemma4 --test chat_template`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/llm-gemma4
git commit -m "$(cat <<'EOF'
feat(llm-gemma4): chat template, GGUF tokenizer bind, RAM weight load

EOF
)"
```

---

### Task 8: Concrete generate loop (prefill + decode on CPU)

**Files:**
- Create: `crates/llm-gemma4/src/impl_/generate.rs`, `crates/llm-gemma4/src/impl_/model.rs`
- Test: `crates/llm-gemma4/tests/synthetic_generate.rs` (tiny random weights, not full E2B)

- [ ] **Step 1: Write failing synthetic generate test**

Build a **shrunk** Gemma-like config (`n_layers=2`, `d_model=32`, vocab=32`, one local + one global) with random f32 weights in a `WeightStore`. Run `generate(..., max_new_tokens=3)` and assert:
- returns 3 token ids
- using `KvCache` vs `FullRecompute` yields logits within `1e-4` relative mid-step (hook a `logits_only` probe on last step)

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `Gemma4CpuModel: TransformerSpec`**

Layer body (text-only):
1. RMSNorm
2. QKV projections (GQA)
3. RoPE / p-RoPE by attention kind
4. Attend over `ContextMemory::view` for the layer’s KV slot (append K/V each position)
5. Out proj + residual
6. RMSNorm + FFN (gated Gelu/SiLU per GGUF key naming — detect `gate`/`up`/`down`)
7. PLE residual add when PLE tensors present for that layer
8. Final norm + lm_head → logits

`generate`:
```rust
pub struct GenerateRequest<'a> {
    pub prompt_tokens: &'a [TokenId],
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
    pub max_context: usize,
}

pub fn generate<C: ContextMemory, P: ComputePort>(...) -> Result<Vec<TokenId>, EngError>
```

Flow: check `prompt_tokens.len() <= max_context` else `ContextTooLong`; `prefill`; sample; loop `decode_step` until EOS or limit; on each step if `position + 1 > max_context` return error.

EOS: from metadata `tokenizer.ggml.eos_token_id` or token string `<turn|>` / `<|eot|>` whichever exists in the file.

- [ ] **Step 4: Run synthetic generate tests**

Run: `devenv shell -- cargo test -p llm-gemma4 --test synthetic_generate`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/llm-gemma4
git commit -m "$(cat <<'EOF'
feat(llm-gemma4): CPU prefill/decode generate with KV and recompute paths

EOF
)"
```

---

### Task 9: `llm-cli` — load → tokenize → generate → print

**Files:**
- Modify: `crates/llm-cli/Cargo.toml`, `crates/llm-cli/src/main.rs`

- [ ] **Step 1: Write CLI**

```rust
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "llm")]
struct Args {
    /// Path to Gemma 4 E2B GGUF
    model: PathBuf,
    /// User prompt
    #[arg(short, long)]
    prompt: String,
    #[arg(long, default_value_t = 64)]
    max_tokens: usize,
    #[arg(long, default_value_t = 4096)]
    max_context: usize,
    #[arg(long)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();
    if args.verbose {
        tracing_subscriber::fmt().with_env_filter("debug").init();
    } else {
        tracing_subscriber::fmt().with_env_filter("info").init();
    }
    if let Err(e) = run(args) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
```

`run`: load weights → format chat template → tokenize (add BOS) → generate → detokenize streaming print tokens as produced (print each decode step). No panic on missing file — map I/O errors to exit 1 with path.

- [ ] **Step 2: Build CLI**

Run: `devenv shell -- cargo build -p llm-cli`
Expected: SUCCESS.

- [ ] **Step 3: Smoke without model (expect clean error)**

Run: `devenv shell -- cargo run -p llm-cli -- --model models/missing.gguf --prompt hi`
Expected: exit ≠ 0, stderr contains `models/missing.gguf`, no backtrace panic.

- [ ] **Step 4: Commit**

```bash
git add crates/llm-cli
git commit -m "$(cat <<'EOF'
feat(llm-cli): load GGUF, generate tokens, fail cleanly on bad paths

EOF
)"
```

---

### Task 10: Optional real-model integration + README

**Files:**
- Create: `README.md`, `crates/llm-gemma4/tests/e2b_smoke.rs`
- Modify: `.gitignore` (already ignores `*.gguf` / `models/`)

- [ ] **Step 1: Integration test gated on env**

```rust
#[test]
fn e2b_generates_a_few_tokens_when_model_present() {
    let path = match std::env::var("LLM_E2B_GGUF") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("skip: set LLM_E2B_GGUF to run");
            return;
        }
    };
    // load + generate 8 tokens from "Say hi."
    // assert at least 1 token and no error
}
```

- [ ] **Step 2: README with run instructions**

Document:
```bash
devenv shell
# download (example):
# huggingface-cli download bartowski/google_gemma-4-E2B-it-GGUF \
#   --include "google_gemma-4-E2B-it-Q8_0.gguf" --local-dir models/
cargo run -p llm-cli -- --model models/google_gemma-4-E2B-it-Q8_0.gguf \
  --prompt "Write a haiku about caches." --max-tokens 64
LLM_E2B_GGUF=models/google_gemma-4-E2B-it-Q8_0.gguf cargo test -p llm-gemma4 --test e2b_smoke
```

State architecture layers + how to swap `ContextMemory` / `ComputePort`.

- [ ] **Step 3: Full workspace test (no model)**

Run: `devenv shell -- cargo test --workspace`
Expected: all non-gated tests PASS.

- [ ] **Step 4: Commit**

```bash
git add README.md crates/llm-gemma4/tests/e2b_smoke.rs
git commit -m "$(cat <<'EOF'
docs: README and optional E2B GGUF smoke test

EOF
)"
```

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| `llm-idea` free of KV/device/I/O | Task 2 |
| Eng ports: ContextMemory, WeightStore, Schedule, ComputePort | Task 3 |
| KV vs full recompute correctness fixture | Task 3 |
| GGUF parse + fail-fast unsupported tensors | Task 4 |
| CPU ComputePort f32 + quant decode-on-use | Task 5 |
| Gemma4 mini-engine idea/eng/impl split | Tasks 6–8 |
| Hybrid local/global + PLE + p-RoPE hooks | Tasks 6, 8 |
| No silent context truncate | Tasks 3, 8 |
| Tokenizer + chat template | Task 7 |
| CLI load → generate → print | Task 9 |
| Optional golden GGUF integration | Task 10 |
| devenv bootstrap | Task 1 |
| Sampling defaults | Task 3 (`sample.rs`) + Task 8/9 wiring |

## Self-review notes

- No TBDs left for tokenizer, GGUF pin, PLE residency, or bootstrap.
- Types named consistently: `TokenId`, `ActId`, `ContextMemory`, `ComputePort`, `EngError`.
- Multimodal / MoE / Metal explicitly out of scope — skipped tensors / refused arches.

---
