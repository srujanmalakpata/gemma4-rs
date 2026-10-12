#!/usr/bin/env bash
# Compares this engine with llama.cpp on a Gemma 4 E2B Q8_0 GGUF and
# rewrites tools/compare/RESULTS.md.
#
#   tools/compare/run.sh MODEL.gguf [LLAMA_DIR] [-- extra compare.py args]
#
# On first use it clones llama.cpp at the pinned tag into LLAMA_DIR (default
# ~/.cache/llm-compare/llama.cpp-TAG, outside this repo), builds it CPU-only
# and compiles tools/compare/llama_dump.cpp against it. It also writes the
# F32-weight reference copy of the model next to MODEL (MODEL minus .gguf,
# plus .f32-reference.gguf; about 18.6 GB for E2B). Later runs reuse both.
set -euo pipefail

LLAMA_TAG=b11564
MODEL=${1:?usage: run.sh MODEL.gguf [LLAMA_DIR] [-- compare.py args]}
shift
LLAMA_DIR=${HOME}/.cache/llm-compare/llama.cpp-${LLAMA_TAG}
if [[ $# -gt 0 && $1 != "--" ]]; then
  LLAMA_DIR=$1
  shift
fi
[[ ${1:-} == "--" ]] && shift
REFERENCE=${MODEL%.gguf}.f32-reference.gguf

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)

if [[ ! -d $LLAMA_DIR ]]; then
  git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
if [[ ! -x $LLAMA_DIR/build-cpu/bin/llama-quantize ]]; then
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build-cpu" -DGGML_METAL=OFF -DGGML_BLAS=OFF \
    -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build "$LLAMA_DIR/build-cpu" -j 8 --target llama llama-quantize
fi
c++ -std=c++17 -O2 -I"$LLAMA_DIR/include" -I"$LLAMA_DIR/ggml/include" "$here/llama_dump.cpp" \
  -L"$LLAMA_DIR/build-cpu/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$LLAMA_DIR/build-cpu/bin" \
  -o "$LLAMA_DIR/build-cpu/bin/llama_dump"

if [[ ! -f $REFERENCE ]]; then
  echo "writing the F32-weight reference model to $REFERENCE" >&2
  "$LLAMA_DIR/build-cpu/bin/llama-quantize" --allow-requantize "$MODEL" "$REFERENCE" F32 8
fi

cargo build --release --manifest-path "$root/Cargo.toml" -p llm-cli
python3 "$here/compare.py" --model "$MODEL" --reference-model "$REFERENCE" \
  --llama-dump "$LLAMA_DIR/build-cpu/bin/llama_dump" --llama-version "$LLAMA_TAG" "$@"
