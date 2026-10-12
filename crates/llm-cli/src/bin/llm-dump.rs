//! Engine side of the logit comparison in `tools/compare`. Mirrors
//! `tools/compare/llama_dump.cpp`: tokenizes PROMPT_FILE (BOS added, special
//! tokens parsed) and writes OUT_PREFIX.tokens as little-endian u32 ids.

use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use llm_gemma4::impl_::tokenizer::Tokenizer;
use llm_gguf::reader::GgufFile;

#[derive(Debug, Parser)]
#[command(name = "llm-dump", about = "Write token ids for a prompt file")]
struct Args {
    model: PathBuf,
    prompt_file: PathBuf,
    out_prefix: String,
    /// Only write token ids (the only mode so far).
    #[arg(long)]
    tokens_only: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let prompt = fs::read_to_string(&args.prompt_file)
        .with_context(|| format!("cannot read `{}`", args.prompt_file.display()))?;
    let gguf = GgufFile::open_path(&args.model)
        .with_context(|| format!("cannot open model `{}`", args.model.display()))?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let tokens = tokenizer.encode(&prompt, true);
    let bytes: Vec<u8> = tokens.iter().flat_map(|token| token.0.to_le_bytes()).collect();
    fs::write(format!("{}.tokens", args.out_prefix), bytes)?;
    eprintln!("wrote {} tokens", tokens.len());
    Ok(())
}
