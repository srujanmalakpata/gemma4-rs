use llm_eng::{context_memory::KvCache, sample::SamplingConfig};
use llm_gemma4::{
    idea::{proportional_rope_freq_factors, Gemma4HyperParams},
    impl_::{
        chat_template::format_user_prompt,
        generate::{generate_cpu, GenerateRequest},
        model::Gemma4CpuModel,
        tokenizer::Tokenizer,
        weights::load_gemma4_e2b,
    },
};
use llm_gguf::reader::GgufFile;

#[test]
fn e2b_metadata_matches_the_published_config() {
    let Ok(path) = std::env::var("LLM_E2B_GGUF") else {
        eprintln!("skipping E2B metadata check: LLM_E2B_GGUF is unset");
        return;
    };
    let gguf = GgufFile::open_path(&path).unwrap();
    assert_eq!(
        Gemma4HyperParams::from_gguf_metadata(&gguf).unwrap(),
        Gemma4HyperParams::e2b()
    );
}

#[test]
fn e2b_rope_freqs_encode_proportional_rope_on_a_quarter_of_the_pairs() {
    let Ok(path) = std::env::var("LLM_E2B_GGUF") else {
        eprintln!("skipping E2B rope_freqs check: LLM_E2B_GGUF is unset");
        return;
    };
    let mut gguf = GgufFile::open_path(&path).unwrap();
    let bytes = gguf.read_tensor_bytes("rope_freqs.weight").unwrap();
    let factors: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(factors, proportional_rope_freq_factors(512, 0.25));
}

#[test]
fn e2b_generates_tokens_when_model_is_configured() {
    let Ok(path) = std::env::var("LLM_E2B_GGUF") else {
        eprintln!("skipping E2B smoke: LLM_E2B_GGUF is unset");
        return;
    };

    let gguf = GgufFile::open_path(&path).unwrap();
    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();
    let hp = Gemma4HyperParams::from_gguf_metadata(&gguf).unwrap();
    drop(gguf);

    let weights = load_gemma4_e2b(&path).unwrap();
    let mut model =
        Gemma4CpuModel::<KvCache<Vec<f32>, Vec<f32>>>::from_weights(weights, hp)
            .unwrap()
            .with_eos_token_ids(tokenizer.end_of_generation_ids().into_iter().map(|id| id.0));
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
