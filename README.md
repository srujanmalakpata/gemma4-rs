# llm-inference-engine

A small Rust CPU inference path for text-only Gemma 4 E2B GGUFs. It reads GGUF
metadata and tokenizer data, dequantizes Q8_0 weights into RAM, then runs a
Gemma-style transformer with GQA, RoPE, sliding-window/global attention, and a
K/V cache. The CPU kernels are deliberately simple reference implementations.

## Setup

This repository uses [devenv](https://devenv.sh/):

```sh
devenv shell
```

Download the Q8_0 model into the ignored `models/` directory:

```sh
hf download bartowski/google_gemma-4-E2B-it-GGUF \
  --include '*Q8_0.gguf' \
  --local-dir models
```

Run generation (substitute the downloaded filename):

```sh
devenv shell -- cargo run -p llm-cli -- \
  --model models/google_gemma-4-e2b-it-Q8_0.gguf \
  --prompt "Explain why leaves change color in autumn." \
  --max-tokens 64
```

Use `--verbose` for debug logs and `--max-context` to set the context ceiling.

## Optional E2B smoke test

The E2B smoke test is skipped unless its model path is explicitly supplied:

```sh
LLM_E2B_GGUF=models/google_gemma-4-e2b-it-Q8_0.gguf \
  devenv shell -- cargo test -p llm-gemma4 --test e2b_smoke -- --nocapture
```
