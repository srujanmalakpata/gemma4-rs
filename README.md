# llm-inference-engine

A Rust CPU inference engine for Google's Gemma 4 E2B in GGUF format, written without ML
libraries or an inference runtime (dependencies: `half`, `rand`, `clap`, `anyhow`, `thiserror`, `tracing`,
`tracing-subscriber`). It has its own GGUF v3 reader, Gemma 4 BPE tokenizer, Q8_0 and BF16 mat-vec kernels,
and the Gemma 4 decoder: 8 query heads sharing 1 KV head, 256-wide heads on sliding-window layers
and 512-wide on global layers, NEOX RoPE with proportional frequencies on global layers, q/k/v
and post-norms, K/V shared by the last 20 layers, per-layer embeddings (PLE), and final logit
soft-capping. Text only.

## What has been verified

All on 2026-10-12, Apple M5 (10 cores, 16 GB), macOS, Rust 1.96.0. Model file: bartowski's
`google_gemma-4-E2B-it-Q8_0.gguf` (4,967,497,184 bytes, sha256 `bb145c0e…d78732`).
Reference: llama.cpp release `b11564` (commit `08ffba2`), built CPU-only.

| Check | Result | How |
|---|---|---|
| Tokenizer matches llama.cpp | 20 of 20 strings give identical token ids (special tokens, byte fallback, CJK, emoji, code, newline runs) | `tools/compare/run.sh`, section Tokenizer of [RESULTS.md](tools/compare/RESULTS.md) |
| Logits match llama.cpp (reference check) | pass, 6 of 6 prompts: same top-1 token at all 119 compared positions, max abs diff at most 0.0494, mean KL at most 2.2e-6. Mean abs diff (0.0033-0.0043) misses the original fixed limit of 0.002 but is within 2x the measured float noise floor (below) | same, section "reference check" |
| Logits with default (quantized) arithmetic | same top-1 token at 118 of 119 positions, mean KL 5.6e-3. For scale, llama.cpp's own Q8_0 run against its F32-weight run: 118 of 119, mean KL 7.4e-3 | same, section "default arithmetic" (informational) |
| Real model metadata, `rope_freqs`, a short generation | pass | `LLM_E2B_GGUF=<path> cargo test --release -p llm-gemma4 --test e2b_smoke` |
| Unit and synthetic-model tests | 46 pass (3 of them do nothing unless `LLM_E2B_GGUF` is set) | `cargo test --workspace` |
| Clippy | no warnings | `cargo clippy --workspace --all-targets -- -D warnings` |

What the logit checks mean:

- **Reference check (pass/fail).** llama.cpp runs a copy of the model whose weights are stored as
  F32 (made with `llama-quantize --allow-requantize ... F32`, which dequantizes Q8_0 and BF16
  exactly), and the engine runs the Q8_0 file with `--f32-activations --fp16-gelu`. Neither side
  rounds activations to 8 bits, so the two should agree to float rounding. Short prompts are
  compared at every position (262,144 logits each); the 628-token prompt, which is longer than the
  512-token sliding window, at every 64th position inside the window, every 8th past it, and the
  last. Tolerances committed before the first run were max |diff| <= 0.05, mean |diff| <= 0.002
  and the same top-1 token everywhere. That run met the max and top-1 limits but missed the mean
  one on every prompt. Rerunning the engine with its sums in f64 instead of f32 moved its own
  logits by as much (mean 0.0050, max 0.046 on the first prompt), so 0.002 was below this model's
  float noise. Each run now measures that floor (`--f64-sums`), and a prompt passes if it is within
  the fixed limits or within 2x the floor, with the same top-1 token everywhere. RESULTS.md keeps a
  column with the verdict under the original limits ("no" for every prompt).
- **Default arithmetic (informational).** Normally both engines quantize activations to Q8_0
  before each Q8_0 mat-vec (faster, int8 dot products). Gemma 4's normalized activations have large
  outliers, so a tiny difference flips some 8-bit roundings and the differences grow through 35
  layers. The first comparison used tolerances set for this mode (mean |diff| <= 0.02, max <= 0.25)
  and failed them (mean 0.20-0.28, max up to 3.4, same top-1 token at 94 of 95 positions); the
  commit history records how that was traced to quantization noise. This mode is now reported as
  top-1 agreement and KL divergence, alongside llama.cpp's own distance from the F32 reference.

Not verified:

- Against Hugging Face transformers or Google's reference implementation. The checks show that
  the engine computes what llama.cpp computes for this file; a llama.cpp deviation from the original
  model would be shared.
- Other Gemma 4 sizes, other GGUF files or quantization types (only F32, F16, BF16 and Q8_0
  tensors load), contexts longer than 628 tokens, or sampling quality.
- On Linux: CI runs the tests and clippy on pull requests, but has not run yet.

## Speed

On the M5 above, `llm-dump` (default arithmetic) ran at 58.0 ms per token over the 628-token
prompt and 59.6 ms over a 16-token one, about 17 tokens per second, after a 1.6 s load. That
includes the full 262,144-wide output layer at every token, and generation does the same work per
token. This is one measurement on one machine, not a benchmark suite.

The prompt is processed one token at a time (no batched prefill), and mat-vecs are split across
threads with `std::thread::scope`. llama.cpp is much faster on prompts because it batches them:
7.5 s against 46 s for the 628-token prompt, both including loading (RESULTS.md, run times).

## Usage

Download the model (about 5 GB) outside the repo or into the ignored `models/` directory:

```sh
hf download bartowski/google_gemma-4-E2B-it-GGUF --include '*Q8_0.gguf' --local-dir models
```

Generate (a release build matters; a debug build is far slower):

```sh
cargo run --release -p llm-cli -- \
  --model models/google_gemma-4-E2B-it-Q8_0.gguf \
  --prompt "What is the capital of France? Answer in one sentence." \
  --max-tokens 24
```

On the machine above this printed `The capital of France is Paris.` in about 4.1 s including
loading, in each of two runs.

The prompt is wrapped in Gemma 4's chat template and sampled with Gemma's defaults (temperature 1,
top-p 0.95, top-k 64), so output varies between runs. Generation stops at `<eos>` or `<turn|>`.

Re-run the comparison with llama.cpp (needs cmake and a C++ compiler; the first run clones and
builds llama.cpp `b11564` under `~/.cache/llm-compare` and writes an 18.6 GB F32 copy of the model
next to the model file):

```sh
tools/compare/run.sh models/google_gemma-4-E2B-it-Q8_0.gguf
```

The repo also has a [devenv](https://devenv.sh/) shell (`devenv shell`) with a Rust toolchain.

## Layout

- `crates/llm-gguf`: GGUF v3 reader (all metadata types; F32, F16, BF16 and Q8_0 tensors).
- `crates/llm-backend-cpu`: kernels (RMSNorm, NEOX RoPE, GELU, Q8_0/BF16/F32 mat-vec).
- `crates/llm-gemma4`: Gemma 4 hyperparameters and layer schedule (`idea`), KV sharing and attention
  windows (`eng`), and the tokenizer, weight loader, model and generation loop (`impl_`).
- `crates/llm-eng`, `crates/llm-idea`: engine and transformer traits, KV cache, sampler.
- `crates/llm-cli`: `llm-cli` (generation) and `llm-dump` (logits for the comparison).
- `tools/compare`: the llama.cpp comparison and its latest results.

## History

The first version (July 2026) was written with Cursor from the plan in `docs/superpowers`; it
had only been tested on random synthetic weights and could not load the real model. The fixes that
made it run Gemma 4 E2B (October 2026) were made with Claude Code; each commit message says what
was wrong and how it was checked.
