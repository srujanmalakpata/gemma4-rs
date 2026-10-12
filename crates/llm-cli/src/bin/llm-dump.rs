//! Engine side of the logit comparison in `tools/compare`. Mirrors
//! `tools/compare/llama_dump.cpp`: tokenizes PROMPT_FILE (BOS added, special
//! tokens parsed), or reads the ids from `--token-file`, then writes
//! OUT_PREFIX.tokens (little-endian u32 ids) and OUT_PREFIX.logits
//! (little-endian f32, one row of `vocab` logits per position).

use std::{fs, path::PathBuf, time::Instant};

use anyhow::{bail, Context, Result};
use clap::Parser;
use llm_eng::context_memory::KvCache;
use llm_gemma4::{
    generate::sequence_logits,
    idea::Gemma4HyperParams,
    impl_::{tokenizer::Tokenizer, weights::load_gemma4_e2b},
    model::{Gemma4CpuModel, Numerics},
};
use llm_gguf::reader::GgufFile;
use llm_idea::types::TokenId;

#[derive(Debug, Parser)]
#[command(name = "llm-dump", about = "Write token ids and per-position logits for a prompt")]
struct Args {
    model: PathBuf,
    prompt_file: PathBuf,
    out_prefix: String,
    /// Only write token ids.
    #[arg(long)]
    tokens_only: bool,
    /// Run these token ids (little-endian u32) instead of tokenizing the prompt.
    #[arg(long)]
    token_file: Option<PathBuf>,
    /// Keep activations in f32 for Q8_0 mat-vecs instead of quantizing them.
    #[arg(long)]
    f32_activations: bool,
    /// Evaluate GELU through ggml's fp16 lookup table, like llama.cpp.
    #[arg(long)]
    fp16_gelu: bool,
    /// With --f32-activations, take Q8_0 mat-vec sums in f64.
    #[arg(long)]
    f64_sums: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let gguf = GgufFile::open_path(&args.model)
        .with_context(|| format!("cannot open model `{}`", args.model.display()))?;
    let tokens = match &args.token_file {
        Some(path) => read_tokens(&fs::read(path).with_context(|| format!("cannot read `{}`", path.display()))?)?,
        None => {
            let prompt = fs::read_to_string(&args.prompt_file)
                .with_context(|| format!("cannot read `{}`", args.prompt_file.display()))?;
            Tokenizer::from_gguf(&gguf)?.encode(&prompt, true)
        }
    };
    fs::write(format!("{}.tokens", args.out_prefix), token_bytes(&tokens))?;
    if args.tokens_only {
        eprintln!("wrote {} tokens", tokens.len());
        return Ok(());
    }

    let hp = Gemma4HyperParams::from_gguf_metadata(&gguf)?;
    drop(gguf);
    let started = Instant::now();
    let weights = load_gemma4_e2b(&args.model)?;
    let loaded = started.elapsed();
    let mut model = Gemma4CpuModel::<KvCache<Vec<f32>, Vec<f32>>>::from_weights(weights, hp)?
        .with_numerics(Numerics {
            quantize_q8_activations: !args.f32_activations,
            fp16_gelu: args.fp16_gelu,
            f64_sums: args.f64_sums,
        });
    let started = Instant::now();
    let rows = sequence_logits(&mut model, &tokens)?;
    let ran = started.elapsed();
    let bytes: Vec<u8> = rows.iter().flatten().flat_map(|value| value.to_le_bytes()).collect();
    fs::write(format!("{}.logits", args.out_prefix), bytes)?;
    eprintln!(
        "wrote {} tokens x {} logits (load {:.1} s, {:.1} ms/token)",
        rows.len(),
        rows.first().map_or(0, Vec::len),
        loaded.as_secs_f64(),
        ran.as_secs_f64() * 1000.0 / rows.len().max(1) as f64
    );
    Ok(())
}

fn read_tokens(bytes: &[u8]) -> Result<Vec<TokenId>> {
    if !bytes.len().is_multiple_of(4) {
        bail!("token file length {} is not a multiple of 4", bytes.len());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| TokenId(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
        .collect())
}

fn token_bytes(tokens: &[TokenId]) -> Vec<u8> {
    tokens.iter().flat_map(|token| token.0.to_le_bytes()).collect()
}
