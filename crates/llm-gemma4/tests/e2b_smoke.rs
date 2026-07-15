use llm_eng::{context_memory::KvCache, sample::SamplingConfig};
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

#[test]
fn e2b_generates_tokens_when_model_is_configured() {
    let Ok(path) = std::env::var("LLM_E2B_GGUF") else {
        eprintln!("skipping E2B smoke: LLM_E2B_GGUF is unset");
        return;
    };

    let gguf = GgufFile::open_path(&path).unwrap();
    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();
    let hp = Gemma4HyperParams::from_gguf_metadata(&gguf);
    let heads = HeadConfig::from_gguf_metadata(&gguf, hp.d_model).unwrap();
    drop(gguf);

    let weights = load_gemma4_e2b(&path).unwrap();
    let mut model =
        Gemma4CpuModel::<KvCache<Vec<f32>, Vec<f32>>>::from_weights(weights, hp, heads)
            .unwrap()
            .with_eos_token_id(tokenizer.eos_token_id().map(|id| id.0));
    let prompt_tokens = tokenizer.encode(&format_user_prompt("Hello"), true);
    let generated = generate_cpu(
        &mut model,
        GenerateRequest {
            prompt_tokens: &prompt_tokens,
            max_new_tokens: 2,
            sampling: SamplingConfig::gemma4_defaults(),
            max_context: 128,
        },
    )
    .unwrap();

    assert!(!generated.is_empty());
}
