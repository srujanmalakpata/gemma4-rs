#!/usr/bin/env python3
"""Compare this engine with llama.cpp on the same Gemma 4 E2B GGUF.

Runs `llama_dump` (tools/compare/llama_dump.cpp, built against llama.cpp) and
`llm-dump` (this repo) on the same inputs:

1. Tokenizer: identical token ids for every string in TOKENIZER_CASES
   (pass/fail).
2. Reference check (pass/fail): for every prompt in LOGIT_PROMPTS, llama.cpp
   runs a copy of the model with all weights stored as F32 (made with
   `llama-quantize --allow-requantize ... F32`, which dequantizes Q8_0 and
   BF16 exactly), and the engine runs the Q8_0 file with f32 activations and
   ggml's fp16 GELU table. Neither side rounds activations to 8 bits, so the
   logits should agree to float rounding. The engine also runs once more with
   its sums taken in f64; the difference between its two runs is the float
   noise floor. A prompt passes if every position has the same top-1 token
   and the max and mean |diff| are each within the fixed REFERENCE_* tolerance
   or within NOISE_FLOOR_FACTOR times the measured floor.
3. Default arithmetic (informational): llama.cpp on the Q8_0 file and the
   engine with its default settings both quantize activations to Q8_0 before
   each Q8_0 mat-vec. Gemma 4's activations have outliers, so tiny
   differences flip some 8-bit roundings and grow through the layers; this
   section reports how far apart that leaves the two, and how far each is
   from the F32 reference.

All logit runs use llama.cpp's token ids, so the model math is compared on
identical input. Writes a Markdown report (default tools/compare/RESULTS.md)
and exits non-zero if a pass/fail check fails. Standard library only.

    python3 tools/compare/compare.py --model E2B-Q8_0.gguf \\
        --reference-model E2B-F32.gguf --llama-dump PATH/TO/llama_dump
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

# Reference-check tolerances, fixed before the check's first run. Logits are
# post-softcap (|logit| <= 30). With no 8-bit activation rounding on either
# side, what is left is float summation order, libm differences and the
# occasional fp16 rounding flip inside the GELU table.
REFERENCE_MAX_ABS_DIFF = 0.05  # largest |engine - llama.cpp| over all positions and vocab
REFERENCE_MEAN_ABS_DIFF = 0.002  # mean |difference| over all positions and vocab
REFERENCE_TOP1_AGREEMENT = 1.0  # fraction of positions whose argmax token matches
# Added after the first run: every prompt had max |diff| 0.029-0.049 and the
# same top-1 tokens, but mean |diff| 0.0035-0.0043, over the fixed 0.002. The
# engine's own f32-sum and f64-sum runs differed by about as much (mean 0.0050,
# max 0.046 on the first prompt), so 0.002 was below this model's float noise.
# A difference within this factor of the floor measured in the same run passes.
NOISE_FLOOR_FACTOR = 2.0

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

def chat(message):
    return f"<|turn>user\n{message}<turn|>\n<|turn>model\n"


with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "long_passage.txt"), encoding="utf-8") as f:
    LONG_PASSAGE = f.read().strip()

LOGIT_PROMPTS = [
    chat("What is the capital of France?"),
    chat("Write a haiku about autumn leaves."),
    chat("Explain in one sentence why the sky is blue."),
    chat("Translate to French: The cat sleeps on the warm windowsill."),
    chat("def fibonacci(n):\n    # complete this Python function"),
    # Longer than the 512-token sliding window, so local layers drop old keys.
    chat("Read this passage, then say in one sentence why the wells failed.\n\n" + LONG_PASSAGE),
]

SLIDING_WINDOW = 512


def compared_positions(n_tokens):
    """Every position of a short prompt. For a long one (pure-Python statistics
    over 262,144 logits per position are slow): every 64th position inside the
    sliding window, every 8th past it, and the last."""
    if n_tokens <= 64:
        return list(range(n_tokens))
    inside = range(0, min(n_tokens, SLIDING_WINDOW), 64)
    past = range(SLIDING_WINDOW, n_tokens, 8)
    return sorted(set(inside) | set(past) | {n_tokens - 1})


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


def log_softmax(row):
    top = max(row)
    log_total = math.log(math.fsum(math.exp(x - top) for x in row)) + top
    return [x - log_total for x in row]


def compare_rows(expected, actual, n_tokens, positions):
    """Logit statistics over `positions`; `expected` is the baseline."""
    n_vocab = len(expected) // n_tokens
    max_abs, total_abs, top1_same, kls = 0.0, 0.0, 0, []
    for position in positions:
        row = slice(position * n_vocab, (position + 1) * n_vocab)
        e, a = expected[row], actual[row]
        diffs = [abs(x - y) for x, y in zip(e, a)]
        max_abs = max(max_abs, max(diffs))
        total_abs += math.fsum(diffs)
        top1_same += max(range(n_vocab), key=e.__getitem__) == max(range(n_vocab), key=a.__getitem__)
        log_p, log_q = log_softmax(e), log_softmax(a)
        kls.append(math.fsum(math.exp(p) * (p - q) for p, q in zip(log_p, log_q)))
    return {
        "max_abs": max_abs,
        "mean_abs": total_abs / (len(positions) * n_vocab),
        "top1": top1_same,
        "kl_mean": sum(kls) / len(positions),
        "kl_max": max(kls),
    }


def compare_logits(args, work):
    rows = []
    for index, text in enumerate(LOGIT_PROMPTS):
        prefix = os.path.join(work, f"logit{index}")
        timing = {}
        timing["llama_q8"] = dump(args.llama_dump, args.model, text, prefix + "-llama-q8", [str(args.threads)])
        tokens_file = prefix + "-llama-q8.tokens"
        timing["llama_f32"] = dump(
            args.llama_dump, args.reference_model, text, prefix + "-llama-f32", [str(args.threads)]
        )
        if read_u32(prefix + "-llama-f32.tokens") != read_u32(tokens_file):
            sys.exit(f"the two llama.cpp runs tokenized prompt {index} differently")
        timing["engine_default"] = dump(
            args.engine_dump, args.model, text, prefix + "-engine-default", ["--token-file", tokens_file]
        )
        timing["engine_reference"] = dump(
            args.engine_dump,
            args.model,
            text,
            prefix + "-engine-reference",
            ["--token-file", tokens_file, "--f32-activations", "--fp16-gelu"],
        )
        timing["engine_f64"] = dump(
            args.engine_dump,
            args.model,
            text,
            prefix + "-engine-f64",
            ["--token-file", tokens_file, "--f32-activations", "--fp16-gelu", "--f64-sums"],
        )
        n_tokens = len(read_u32(tokens_file))
        positions = compared_positions(n_tokens)
        logits = {name: read_f32(f"{prefix}-{name.replace('_', '-')}.logits") for name in timing}
        reference = compare_rows(logits["llama_f32"], logits["engine_reference"], n_tokens, positions)
        floor = compare_rows(logits["engine_reference"], logits["engine_f64"], n_tokens, positions)
        reference["floor"] = floor
        reference["within_fixed"] = (
            reference["max_abs"] <= REFERENCE_MAX_ABS_DIFF and reference["mean_abs"] <= REFERENCE_MEAN_ABS_DIFF
        )
        reference["pass"] = (
            reference["max_abs"] <= max(REFERENCE_MAX_ABS_DIFF, NOISE_FLOOR_FACTOR * floor["max_abs"])
            and reference["mean_abs"] <= max(REFERENCE_MEAN_ABS_DIFF, NOISE_FLOOR_FACTOR * floor["mean_abs"])
            and reference["top1"] >= REFERENCE_TOP1_AGREEMENT * len(positions)
        )
        rows.append(
            {
                "text": text,
                "n_tokens": n_tokens,
                "n_compared": len(positions),
                "timing": timing,
                "reference": reference,
                "engine_vs_llama_q8": compare_rows(logits["llama_q8"], logits["engine_default"], n_tokens, positions),
                "llama_q8_vs_f32": compare_rows(logits["llama_f32"], logits["llama_q8"], n_tokens, positions),
                "engine_vs_f32": compare_rows(logits["llama_f32"], logits["engine_default"], n_tokens, positions),
            }
        )
    return rows


def show(text):
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t").replace("|", "\\|")


def short(text):
    message = text[len("<|turn>user\n"):].split("<turn|>")[0] if text.startswith("<|turn>user\n") else text
    if LONG_PASSAGE in message:
        message = message.replace(LONG_PASSAGE, f"<long_passage.txt, {len(LONG_PASSAGE.split())} words>")
    return show(message)


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
    if not logit_rows:
        return "\n".join(lines) + "\n"

    positions = sum(r["n_compared"] for r in logit_rows)
    lines += [
        "",
        "## Logits: reference check (pass/fail)",
        "",
        f"`{os.path.basename(args.reference_model)}` holds the same weights stored as F32.",
        "llama.cpp runs it; the engine runs the Q8_0 file with `--f32-activations --fp16-gelu`.",
        "Neither side rounds activations to 8 bits. Each prompt is the chat-formatted user",
        "message below plus BOS. All 262,144 logits are compared at every position of the short",
        "prompts; for the long one, at every 64th position inside the 512-token sliding window,",
        "every 8th past it, and the last.",
        "",
        "Float noise floor: the same engine run with its Q8_0 mat-vec sums taken in f64 instead",
        "of f32 (`--f64-sums`); the floor columns compare those two engine runs.",
        "",
        f"Fixed tolerances, set before the first run: max |diff| <= {REFERENCE_MAX_ABS_DIFF},",
        f"mean |diff| <= {REFERENCE_MEAN_ABS_DIFF}, same top-1 token at every position. The first run",
        "missed the mean tolerance on every prompt, and the engine's own floor turned out to be",
        f"larger than it, so since then a max or mean within {NOISE_FLOOR_FACTOR:g}x the floor measured in the",
        "same run also passes. The \"Fixed\" column shows the original verdict.",
        "",
        "| User message | Tokens | Compared | Max abs diff | Floor max | Mean abs diff | Floor mean "
        "| Top-1 same | Mean KL | Fixed | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in logit_rows:
        s = r["reference"]
        f = s["floor"]
        lines.append(
            f"| `{short(r['text'])}` | {r['n_tokens']} | {r['n_compared']} | {s['max_abs']:.4f} "
            f"| {f['max_abs']:.4f} | {s['mean_abs']:.5f} | {f['mean_abs']:.5f} "
            f"| {s['top1']}/{r['n_compared']} | {s['kl_mean']:.1e} "
            f"| {'yes' if s['within_fixed'] else 'no'} | {'yes' if s['pass'] else '**no**'} |"
        )
    lines += [
        "",
        "## Logits: default arithmetic (informational)",
        "",
        "Both engines' normal settings quantize activations to Q8_0 before every Q8_0",
        "mat-vec. Gemma 4's normalized activations have outliers (norm weights reach 123), so",
        "a block's single 8-bit scale is large and a tiny upstream difference flips some",
        "roundings; these differences grow through 35 layers. KL divergence (nats, from the",
        "first-named run's distribution) measures the effect on next-token probabilities and,",
        f"unlike raw logit differences, ignores uniform shifts. Totals over {positions} positions:",
        "",
        "| Comparison | Top-1 same | Mean KL | Max KL | Mean abs diff |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in [
        ("engine_vs_llama_q8", "llama.cpp Q8_0 vs engine default"),
        ("llama_q8_vs_f32", "llama.cpp F32 weights vs llama.cpp Q8_0"),
        ("engine_vs_f32", "llama.cpp F32 weights vs engine default"),
    ]:
        stats = [r[key] for r in logit_rows]
        top1 = sum(s["top1"] for s in stats)
        kl_mean = sum(s["kl_mean"] * r["n_compared"] for s, r in zip(stats, logit_rows)) / positions
        mean_abs = sum(s["mean_abs"] * r["n_compared"] for s, r in zip(stats, logit_rows)) / positions
        lines.append(
            f"| {label} | {top1}/{positions} | {kl_mean:.1e} | {max(s['kl_max'] for s in stats):.1e} "
            f"| {mean_abs:.3f} |"
        )
    lines += [
        "",
        "## Run times",
        "",
        "Wall-clock seconds for one whole run of each tool (load plus every position, one",
        "token at a time for the engine), not a benchmark.",
        "",
        "| User message | llama.cpp Q8_0 | llama.cpp F32 | Engine default | Engine reference | Engine f64 sums |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in logit_rows:
        t = r["timing"]
        lines.append(
            f"| `{short(r['text'])}` | {t['llama_q8']:.1f} | {t['llama_f32']:.1f} "
            f"| {t['engine_default']:.1f} | {t['engine_reference']:.1f} | {t['engine_f64']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="Q8_0 GGUF")
    parser.add_argument("--reference-model", help="the same model with F32 weights (needed for logits)")
    parser.add_argument("--llama-dump", required=True)
    parser.add_argument("--llama-version", default="unknown", help="llama.cpp tag, for the report")
    parser.add_argument("--engine-dump", default=os.path.join(root, "target", "release", "llm-dump"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--report", default=os.path.join(here, "RESULTS.md"))
    parser.add_argument("--skip-logits", action="store_true")
    args = parser.parse_args()
    if not args.skip_logits and not args.reference_model:
        parser.error("--reference-model is required unless --skip-logits is given")

    with tempfile.TemporaryDirectory() as work:
        tokenizer_rows = compare_tokenizer(args, work)
        logit_rows = [] if args.skip_logits else compare_logits(args, work)

    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report(args, tokenizer_rows, logit_rows))
    failures = sum(not r[3] for r in tokenizer_rows) + sum(not r["reference"]["pass"] for r in logit_rows)
    print(f"wrote {args.report}; {failures} failing checks")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
