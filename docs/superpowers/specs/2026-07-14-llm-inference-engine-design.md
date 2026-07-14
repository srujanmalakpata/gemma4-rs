# LLM Inference Engine — Design Spec

**Date:** 2026-07-14  
**Status:** Draft for review  
**Stack:** Rust  
**v1 target:** Gemma 4 E2B, text-in/text-out, GGUF, CPU backend

## Goal

Build a learnable LLM inference engine that separates:

1. the **mathematical idea** of a transformer / model,
2. the **engineering** choices that realize that idea under real resource limits,
3. the **concrete implementations** (kernels, file formats, devices),

so core logic stays stable while backends, caches, and model specializations evolve. Primary uses: learn inference deeply, run efficient local experiments, and swap pieces without rewriting the abstract engine.

## Non-goals (v1)

- Multimodal (image/audio) input
- MoE routing (Gemma 4 26B A4B)
- Metal / CUDA backends (ports exist; not implemented)
- Continuous batching / multi-request serving
- Hugging Face safetensors loader
- Production serving quality / safety filters

## Guiding analogy

| Computer science | This engine |
|------------------|-------------|
| Abstract memory / compute | Transformer and model **idea** specs |
| RAM, cache, SSD, schedulers | **Engineering**: ContextMemory, WeightStore, Schedule, ComputePort |
| This DRAM stick / this GPU | **Implementation**: CPU kernels, GGUF layout, device buffers |

KV cache is **not** part of the mathematical core. Mathematically, attention is over the full context. A KV cache is one engineering strategy to avoid recomputing past K/V — analogous to a CPU cache vs “memory” as a concept.

## Architecture overview

### Three layers

```text
llm-idea          mathematical contracts (no I/O, no devices, no caches)
     ↓
llm-eng           systems ports (memory strategies, weight residency, schedule, compute port)
     ↓
impl crates       CPU backend, GGUF I/O, CLI wiring, concrete buffers
```

**Dependency rule:** idea never depends on eng or impl. Eng depends on idea. Impl depends on eng + idea as needed. CLI wires everything.

### Recursive model mini-engines

There is a **global** abstract transformer engine. Each model family is a **mini-engine** that refines that base the same way a microarchitecture refines an ISA:

```text
TransformerSpec  (universal: residual stream, layer loop, attn+ffn laws)
       │
       ├── Gemma4Spec   idea diffs → eng → impl   (v1)
       ├── LlamaSpec    …                         (later)
       └── MoESpec      …                         (later)
```

A model module is not “just weights.” It declares which base hooks change (attention kind, PLE, RoPE variant, layer pattern), then applies the **same idea → eng → impl** split inside itself. Recursion is intentional; depth is bounded — v1 only builds Base + Gemma4 E2B.

## Crate layout (Approach 1)

| Crate | Layer | Responsibility |
|-------|-------|----------------|
| `llm-idea` | Idea | `TransformerSpec` traits/types: embed, layer, attention over full context, FFN, logits. Opaque activation/weight handles — no tensor library. |
| `llm-eng` | Eng | Ports: `ContextMemory`, `WeightStore`, `Schedule`, `ComputePort`. Shared across model mini-engines. |
| `llm-gemma4` | Mini-engine | One crate with internal `idea` / `eng` / `impl` modules (same recursive split, no extra crates in v1). Refinements: hybrid local/global attention, PLE, p-RoPE as required by E2B, text-only graph; eng wiring; GGUF tensor binding; tokenizer + chat template for this family. |
| `llm-gguf` | Impl | Parse GGUF → metadata + tensors. Concrete for v1; abstract `Checkpoint` only when a second format appears. |
| `llm-backend-cpu` | Impl | First `ComputePort` implementation (f32 path; quantized decode-on-use can follow). |
| `llm-cli` | Impl | `load → tokenize → generate → print`. |

Optional later: `llm-backend-metal`, `llm-backend-cuda` implementing the same `ComputePort`.

### Gemma 4 E2B idea-level diffs (v1)

From the Gemma 4 family (text path only):

- Dense decoder stack (~35 layers for E2B), large vocab (~262K).
- Hybrid attention: interleaved local sliding-window (~512) and global layers; final layer global.
- Per-Layer Embeddings (PLE): each layer may add a per-token embedding lookup (large tables, cheap lookup) — expressed in the **idea** as an extra residual contribution; **eng** decides residency/paging.
- Long context capability up to 128K for E2B — eng may cap lower for laptop RAM in v1; no silent truncation without an error.

Multimodal towers are out of scope; do not load or execute vision/audio paths in v1.

## Engineering ports

| Port | Meaning | v1 concrete choice |
|------|---------|-------------------|
| `ContextMemory` | How past context is supplied to attention | KV cache strategy; **full recompute** remains a valid alternate for correctness tests |
| `WeightStore` | How parameters live in memory | In-RAM tensors from GGUF |
| `Schedule` | Prefill vs decode, request shape | Single sequence: prefill then token-by-token decode |
| `ComputePort` | Where element ops / matmuls run | CPU backend |

Gemma4 eng specializes how hybrid local/global attention talks to `ContextMemory` (windowed vs global retention policies).

## Data flow (v1 generate)

1. CLI opens GGUF; Gemma4 mini-engine fills `WeightStore` and builds the specialized graph.
2. Prompt → `TokenId`s via the Gemma4 mini-engine tokenizer / chat template.
3. **Prefill:** Idea evaluates embed + N layers over the full prompt. Eng records K/V into `ContextMemory` when using the KV strategy. ComputePort runs ops on CPU.
4. **Decode loop:** Idea evaluates layers for the new token only; eng appends K_t, V_t; sample logits → next token; stream/detokenize until EOS or limit.
5. Sampling defaults align with Gemma 4 guidance where practical (`temperature=1.0`, `top_p=0.95`, `top_k=64`); exact wiring belongs in eng/CLI config.

Invariant: swapping `ContextMemory` (KV vs recompute) or `ComputePort` (CPU → Metal later) must not rewrite `TransformerSpec`.

## Error handling

| Layer | Examples | Policy |
|-------|----------|--------|
| Impl / I/O | Missing file, corrupt GGUF, unknown tensor, unsupported quant | Fail fast with path / tensor key / arch context |
| Eng | Context longer than configured window, OOM | Explicit `Result`; no silent truncate in v1 |
| Idea / Gemma4 contracts | Shape / position / layer contract breaks | Debug asserts in tests; `Result` at eng/impl boundaries |
| CLI | User-facing failures | Exit code + short message; verbose via `RUST_LOG` / `--verbose` |

No panics for expected user mistakes (bad path, wrong model file).

## Testing

1. **Idea:** tiny synthetic weights — one block attention + residual; known numeric fixture on recompute path.
2. **Eng:** same fixture with KV `ContextMemory` vs full recompute; logits within tolerance.
3. **Gemma4 / GGUF:** header/metadata parse; optional golden short-prompt logits when a pinned public E2B GGUF is available.
4. **CLI integration:** generate a few tokens when weights are present; CI may skip if no model file.

## v1 success criteria

- Load a Gemma 4 E2B GGUF (text path).
- Run prompt → autoregressive decode on CPU and print tokens.
- Demonstrate that core idea stays free of KV cache / device details.
- Leave clear ports so Metal/CUDA, alternate ContextMemory, and later model mini-engines can land without rewriting `TransformerSpec`.

## Future experiments (seams only)

- Attention / decode algorithms (sliding policies, speculative decoding)
- Numerics (int8/int4 kernels, Metal)
- Serving (batching, continuous batching)

These plug into eng/impl or new model mini-engines; they do not enlarge the mathematical core unless the math itself changes.

## Deferred to implementation plan (not design gaps)

- Concrete tokenizer dependency and exact instruct chat-template string handling
- Which public E2B GGUF quant file to pin for integration tests
- PLE table residency: always in RAM for v1 (default) vs any paging
- Project bootstrap: `devenv` + Rust toolchain (repo is currently empty)
