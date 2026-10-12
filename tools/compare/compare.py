#!/usr/bin/env python3
"""Compare this engine with llama.cpp on the same Gemma 4 E2B GGUF.

Runs `llama_dump` (tools/compare/llama_dump.cpp, built against llama.cpp) and
`llm-dump` (this repo) on the same inputs and checks:

1. Tokenizer: identical token ids for every string in TOKENIZER_CASES.
2. Logits: for every prompt in LOGIT_PROMPTS, both run the *same* token ids
   (llama.cpp's) and every position's logits are compared against the
   tolerances below.

Writes a Markdown report (default tools/compare/RESULTS.md) and exits non-zero
if any check fails. Standard library only.

    python3 tools/compare/compare.py --model MODEL.gguf \
        --llama-dump PATH/TO/llama_dump [--engine-dump target/release/llm-dump]
"""

import argparse
import array
import math
import os
import platform
import subprocess
import sys
import tempfile
import time

# Tolerances, fixed before the first real-model run. Logits are post-softcap
# (|logit| <= 30). They allow for llama.cpp's own approximations (GELU via an
# fp16 lookup table, bf16-rounded activations for the BF16 tensor) and for
# float summation order; anything structural (a wrong norm, RoPE layout,
# mask) shows up as differences of whole logits and wrong top-1 tokens.
MAX_ABS_DIFF = 0.25  # largest |engine - llama.cpp| over all positions and vocab
MEAN_ABS_DIFF = 0.02  # mean |difference| over all positions and vocab
TOP1_AGREEMENT = 1.0  # fraction of positions whose argmax token matches

TOKENIZER_CASES = [
    "Hello",
    "Hello world",
    " leading space",
    "two  spaces and   three",
    "trailing space ",
    "tabs\tand\ttabs",
    "line one\nline two\n\n\nline five",
    "\n\nstarts with newlines",
    "Numbers: 12345 67.89 1,000,000",
    "Punctuation!? (brackets) [square] {curly} <angle>",
    "café naïve résumé",
    "日本語のテキスト",
    "Emoji 🙂🚀 and symbols ∑ ≤ ≥",
    "def f(x):\n    return x ** 2  # square",
    "https://example.com/path?query=1&b=2",
    "<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n",
    "text with <|turn> in the middle and <turn|> again",
    "<|think|>reasoning<|channel>thought\n<channel|>",
    "a" * 50,
    "Supercalifragilisticexpialidocious antidisestablishmentarianism",
]

LOGIT_PROMPTS = [
    "<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n",
    "<|turn>user\nWrite a haiku about autumn leaves.<turn|>\n<|turn>model\n",
    "<|turn>user\nExplain in one sentence why the sky is blue.<turn|>\n<|turn>model\n",
    "<|turn>user\nTranslate to French: The cat sleeps on the warm windowsill.<turn|>\n<|turn>model\n",
    "<|turn>user\ndef fibonacci(n):\n    # complete this Python function<turn|>\n<|turn>model\n",
]


def read_u32(path):
    values = array.array("I")
    with open(path, "rb") as f:
        values.frombytes(f.read())
    return list(values)


def read_f32(path):
    values = array.array("f")
    with open(path, "rb") as f:
        values.frombytes(f.read())
    return values


def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr}")
    return result


def dump(tool, model, text, prefix, extra):
    prompt_path = prefix + ".txt"
    with open(prompt_path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    started = time.monotonic()
    run([tool, model, prompt_path, prefix] + extra)
    return time.monotonic() - started


def compare_tokenizer(args, work):
    rows = []
    for index, text in enumerate(TOKENIZER_CASES):
        ref = os.path.join(work, f"tok{index}-llama")
        eng = os.path.join(work, f"tok{index}-engine")
        dump(args.llama_dump, args.model, text, ref, [str(args.threads), "--tokens-only"])
        dump(args.engine_dump, args.model, text, eng, ["--tokens-only"])
        expected, actual = read_u32(ref + ".tokens"), read_u32(eng + ".tokens")
        rows.append((text, expected, actual, expected == actual))
    return rows


def logit_stats(expected, actual, n_tokens):
    n_vocab = len(expected) // n_tokens
    max_abs, total_abs, top1_same, worst_margin = 0.0, 0.0, 0, math.inf
    for position in range(n_tokens):
        row = slice(position * n_vocab, (position + 1) * n_vocab)
        e, a = expected[row], actual[row]
        diffs = [abs(x - y) for x, y in zip(e, a)]
        max_abs = max(max_abs, max(diffs))
        total_abs += sum(diffs)
        e_top = max(range(n_vocab), key=e.__getitem__)
        a_top = max(range(n_vocab), key=a.__getitem__)
        top1_same += e_top == a_top
        if e_top != a_top:
            ranked = sorted(e, reverse=True)
            worst_margin = min(worst_margin, ranked[0] - ranked[1])
    return {
        "n_tokens": n_tokens,
        "max_abs": max_abs,
        "mean_abs": total_abs / (n_tokens * n_vocab),
        "top1": top1_same / n_tokens,
        "worst_margin": worst_margin,
    }


def compare_logits(args, work):
    rows = []
    for index, text in enumerate(LOGIT_PROMPTS):
        ref = os.path.join(work, f"logit{index}-llama")
        eng = os.path.join(work, f"logit{index}-engine")
        ref_seconds = dump(args.llama_dump, args.model, text, ref, [str(args.threads)])
        eng_seconds = dump(
            args.engine_dump, args.model, text, eng, ["--token-file", ref + ".tokens"]
        )
        tokens = read_u32(ref + ".tokens")
        stats = logit_stats(read_f32(ref + ".logits"), read_f32(eng + ".logits"), len(tokens))
        stats.update(text=text, ref_seconds=ref_seconds, eng_seconds=eng_seconds)
        stats["pass"] = (
            stats["max_abs"] <= MAX_ABS_DIFF
            and stats["mean_abs"] <= MEAN_ABS_DIFF
            and stats["top1"] >= TOP1_AGREEMENT
        )
        rows.append(stats)
    return rows


def show(text):
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t").replace("|", "\\|")


def report(args, tokenizer_rows, logit_rows):
    lines = [
        "# Engine vs llama.cpp on Gemma 4 E2B",
        "",
        "Generated by `tools/compare/compare.py`; do not edit by hand.",
        "",
        f"- Model: `{os.path.basename(args.model)}` ({os.path.getsize(args.model):,} bytes)",
        f"- Reference: llama.cpp `{args.llama_version}`, CPU only, f32 KV cache, flash attention off",
        f"- Machine: {platform.machine()}, {platform.system()} {platform.release()}",
        "",
        "## Tokenizer",
        "",
        f"{sum(r[3] for r in tokenizer_rows)} of {len(tokenizer_rows)} strings give identical token ids.",
        "",
        "| Text | Tokens | Match |",
        "|---|---:|---|",
    ]
    for text, expected, actual, same in tokenizer_rows:
        shown = show(text if len(text) <= 60 else text[:57] + "...")
        detail = "yes" if same else f"**no**: llama.cpp {expected}, engine {actual}"
        lines.append(f"| `{shown}` | {len(expected)} | {detail} |")
    if logit_rows:
        lines += [
            "",
            "## Logits",
            "",
            "Both sides run llama.cpp's token ids for each prompt (BOS plus the chat-formatted",
            "text) and every position's full 262,144-entry logit row is compared.",
            "",
            f"Tolerances: max |diff| <= {MAX_ABS_DIFF}, mean |diff| <= {MEAN_ABS_DIFF},",
            f"top-1 token agreement >= {TOP1_AGREEMENT:.0%} of positions.",
            "",
            "| Prompt | Positions | Max abs diff | Mean abs diff | Top-1 agree | llama.cpp s | Engine s | Pass |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for r in logit_rows:
            lines.append(
                f"| `{show(r['text'])}` | {r['n_tokens']} | {r['max_abs']:.4f} | {r['mean_abs']:.5f} "
                f"| {r['top1']:.0%} | {r['ref_seconds']:.1f} | {r['eng_seconds']:.1f} "
                f"| {'yes' if r['pass'] else '**no**'} |"
            )
        lines += [
            "",
            "Times are wall-clock for a whole run of each tool (load plus all positions), not a",
            "benchmark.",
        ]
    return "\n".join(lines) + "\n"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--llama-dump", required=True)
    parser.add_argument("--llama-version", default="unknown", help="llama.cpp tag, for the report")
    parser.add_argument("--engine-dump", default=os.path.join(root, "target", "release", "llm-dump"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--report", default=os.path.join(here, "RESULTS.md"))
    parser.add_argument("--skip-logits", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as work:
        tokenizer_rows = compare_tokenizer(args, work)
        logit_rows = [] if args.skip_logits else compare_logits(args, work)

    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report(args, tokenizer_rows, logit_rows))
    failures = sum(not r[3] for r in tokenizer_rows) + sum(not r["pass"] for r in logit_rows)
    print(f"wrote {args.report}; {failures} failing checks")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
