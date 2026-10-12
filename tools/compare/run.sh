#!/usr/bin/env bash
# Compares this engine with llama.cpp on a Gemma 4 E2B GGUF and rewrites
# tools/compare/RESULTS.md.
#
#   tools/compare/run.sh MODEL.gguf [LLAMA_DIR] [-- extra compare.py args]
#
# On first use it clones llama.cpp at the pinned tag into LLAMA_DIR (default
# ~/.cache/llm-compare/llama.cpp-TAG, outside this repo), builds it CPU-only
# and compiles tools/compare/llama_dump.cpp against it. Later runs reuse it.
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

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)

if [[ ! -d $LLAMA_DIR ]]; then
  git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
if ! compgen -G "$LLAMA_DIR/build-cpu/bin/libllama.*" > /dev/null; then
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build-cpu" -DGGML_METAL=OFF -DGGML_BLAS=OFF \
    -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build "$LLAMA_DIR/build-cpu" -j 8 --target llama
fi
c++ -std=c++17 -O2 -I"$LLAMA_DIR/include" -I"$LLAMA_DIR/ggml/include" "$here/llama_dump.cpp" \
  -L"$LLAMA_DIR/build-cpu/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$LLAMA_DIR/build-cpu/bin" \
  -o "$LLAMA_DIR/build-cpu/bin/llama_dump"

cargo build --release --manifest-path "$root/Cargo.toml" -p llm-cli
python3 "$here/compare.py" --model "$MODEL" --llama-dump "$LLAMA_DIR/build-cpu/bin/llama_dump" \
  --llama-version "$LLAMA_TAG" "$@"
