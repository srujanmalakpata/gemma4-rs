use llm_eng::{
    context_memory::{FullRecompute, KvCache},
    sample::SamplingConfig,
};
use llm_gemma4::{
    generate::{generate_cpu, generate_cpu_recompute, GenerateRequest},
    idea::{Gemma4HyperParams, GlobalPRope},
    model::Gemma4CpuModel,
};
use llm_idea::types::TokenId;

fn synthetic_hp() -> Gemma4HyperParams {
    Gemma4HyperParams {
        // Layers 0-3 local, 4 global, 5 local, 6 global (last). Layers 5 and
        // 6 share the caches of layers 3 and 4.
        n_layers: 7,
        vocab_size: 32,
        d_model: 32,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim_local: 8,
        head_dim_global: 16,
        ple_dim: 4,
        sliding_window: 8,
        shared_kv_layers: 2,
        local_rope_base: 10_000.0,
        global_p_rope: GlobalPRope {
            p: 0.25,
            base: 1_000_000.0,
        },
        rms_norm_eps: 1e-6,
        final_logit_softcap: 30.0,
    }
}

#[test]
fn generate_matches_between_cache_strategies() {
    let hp = synthetic_hp();
    let request = GenerateRequest {
        prompt_tokens: &[TokenId(1), TokenId(2), TokenId(3)],
        max_new_tokens: 3,
        sampling: SamplingConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
        },
        max_context: 16,
    };
    let mut cached = Gemma4CpuModel::<KvCache<Vec<f32>, Vec<f32>>>::synthetic(7, hp.clone());
    let mut recomputed = Gemma4CpuModel::<FullRecompute<Vec<f32>, Vec<f32>>>::synthetic(7, hp);

    let cached_tokens = generate_cpu(&mut cached, request).unwrap();
    let recomputed_tokens = generate_cpu_recompute(&mut recomputed, request).unwrap();

    assert_eq!(cached_tokens.len(), 3);
    assert_eq!(cached_tokens, recomputed_tokens);
    assert_relative_eq(cached.last_logits(), recomputed.last_logits(), 1e-4);
    assert!(cached.last_logits().iter().all(|logit| logit.abs() <= 30.0));
}

fn assert_relative_eq(left: &[f32], right: &[f32], tolerance: f32) {
    assert_eq!(left.len(), right.len());
    for (&left, &right) in left.iter().zip(right) {
        let scale = left.abs().max(right.abs()).max(1.0);
        assert!(
            (left - right).abs() <= tolerance * scale,
            "{left} != {right} within {tolerance} relative tolerance"
        );
    }
}
