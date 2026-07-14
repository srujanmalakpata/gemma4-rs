use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use llm_eng::sample::SamplingConfig;
use llm_gemma4::{
    idea::Gemma4HyperParams,
    impl_::{
        chat_template::format_user_prompt,
        generate::{generate_cpu, GenerateRequest},
        model::{Gemma4CpuModel, HeadConfig},
        tokenizer::Tokenizer,
        weights::load_gemma4_e2b,
    },
};
use llm_gguf::reader::GgufFile;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(name = "llm", about = "Run a Gemma 4 GGUF on the CPU")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value_t = 64)]
    max_tokens: usize,
    #[arg(long, default_value_t = 4_096)]
    max_context: usize,
    #[arg(long)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();
    let default_filter = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter)))
        .with_writer(std::io::stderr)
        .init();

    if let Err(error) = run(args) {
        eprintln!("llm: {error:#}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<()> {
    let gguf = GgufFile::open_path(&args.model)
        .with_context(|| format!("cannot open model `{}`", args.model.display()))?;
    let tokenizer = Tokenizer::from_gguf(&gguf)
        .with_context(|| format!("cannot read tokenizer from `{}`", args.model.display()))?;
    let hp = Gemma4HyperParams::from_gguf_metadata(&gguf);
    let heads = HeadConfig::from_gguf_metadata(&gguf, hp.d_model)
        .with_context(|| format!("cannot read attention configuration from `{}`", args.model.display()))?;
    drop(gguf);

    let weights = load_gemma4_e2b(&args.model)
        .with_context(|| format!("cannot load model `{}`", args.model.display()))?;
    let mut model = Gemma4CpuModel::<llm_eng::context_memory::KvCache<Vec<f32>, Vec<f32>>>::from_weights(weights, hp, heads)?
        .with_eos_token_id(tokenizer.eos_token_id().map(|id| id.0));
    let formatted = format_user_prompt(&args.prompt);
    let prompt_tokens = tokenizer.encode(&formatted, true);
    let generated = generate_cpu(
        &mut model,
        GenerateRequest {
            prompt_tokens: &prompt_tokens,
            max_new_tokens: args.max_tokens,
            sampling: SamplingConfig::gemma4_defaults(),
            max_context: args.max_context,
        },
    )?;

    println!("{}", tokenizer.decode(&generated));
    Ok(())
}
