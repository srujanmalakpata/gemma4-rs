use llm_eng::{context_memory::ContextMemory, sample::{SamplingConfig, sample_logits}, EngError};
use llm_idea::{transformer, types::TokenId};
use rand::SeedableRng;

use super::model::{CpuActivations, Gemma4CpuModel};

#[derive(Clone, Copy, Debug)]
pub struct GenerateRequest<'a> {
    pub prompt_tokens: &'a [TokenId],
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
    pub max_context: usize,
}

pub fn generate_cpu<C>(
    model: &mut Gemma4CpuModel<C>,
    request: GenerateRequest<'_>,
) -> Result<Vec<TokenId>, EngError>
where
    C: ContextMemory<K = Vec<f32>, V = Vec<f32>> + Default,
{
    if request.prompt_tokens.len() > request.max_context {
        return Err(EngError::ContextTooLong {
            got: request.prompt_tokens.len(),
            max: request.max_context,
        });
    }
    if request.max_new_tokens == 0 {
        return Ok(Vec::new());
    }
    if request.prompt_tokens.is_empty() {
        return Err(EngError::Msg("generation requires at least one prompt token".into()));
    }

    model.clear_context();
    let mut logits = Vec::new();
    for &token in request.prompt_tokens {
        // Activations are only needed within one token's forward pass.
        logits = transformer::prefill(model, &(), &[token], &mut CpuActivations::default())?;
        model.finish_prefill_token();
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let mut generated = Vec::with_capacity(request.max_new_tokens);
    for index in 0..request.max_new_tokens {
        if request.prompt_tokens.len() + generated.len() >= request.max_context {
            return Err(EngError::ContextTooLong {
                got: request.prompt_tokens.len() + generated.len() + 1,
                max: request.max_context,
            });
        }
        let token = TokenId(sample_logits(
            &logits,
            request.sampling.temperature,
            request.sampling.top_p,
            request.sampling.top_k,
            &mut rng,
        ));
        generated.push(token);
        if model.is_eos(token.0) {
            break;
        }
        if index + 1 == request.max_new_tokens {
            break;
        }
        logits = transformer::decode_step(
            model,
            &(),
            token,
            request.prompt_tokens.len() + generated.len() - 1,
            &mut CpuActivations::default(),
        )?;
    }
    Ok(generated)
}

/// Reference path: rebuilds the complete sequence for every logits request.
/// It deliberately does not reuse incremental K/V state.
pub fn generate_cpu_recompute<C>(
    model: &mut Gemma4CpuModel<C>,
    request: GenerateRequest<'_>,
) -> Result<Vec<TokenId>, EngError>
where
    C: ContextMemory<K = Vec<f32>, V = Vec<f32>> + Default,
{
    if request.prompt_tokens.is_empty() {
        return Err(EngError::Msg("generation requires at least one prompt token".into()));
    }
    if request.prompt_tokens.len() > request.max_context {
        return Err(EngError::ContextTooLong { got: request.prompt_tokens.len(), max: request.max_context });
    }
    let mut sequence = request.prompt_tokens.to_vec();
    let mut generated = Vec::with_capacity(request.max_new_tokens);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    for _ in 0..request.max_new_tokens {
        model.clear_context();
        let mut logits = Vec::new();
        for &token in &sequence {
            logits = transformer::prefill(model, &(), &[token], &mut CpuActivations::default())?;
            model.finish_prefill_token();
        }
        let token = TokenId(sample_logits(&logits, request.sampling.temperature, request.sampling.top_p, request.sampling.top_k, &mut rng));
        generated.push(token);
        if model.is_eos(token.0) {
            break;
        }
        if sequence.len() + 1 >= request.max_context {
            return Err(EngError::ContextTooLong { got: sequence.len() + 1, max: request.max_context });
        }
        sequence.push(token);
    }
    Ok(generated)
}

/// Runs `tokens` through the model from an empty context and returns the
/// logits after each one; row `i` scores the token that follows `tokens[i]`.
pub fn sequence_logits<C>(model: &mut Gemma4CpuModel<C>, tokens: &[TokenId]) -> Result<Vec<Vec<f32>>, EngError>
where
    C: ContextMemory<K = Vec<f32>, V = Vec<f32>> + Default,
{
    model.clear_context();
    tokens
        .iter()
        .map(|&token| {
            let logits = transformer::prefill(model, &(), &[token], &mut CpuActivations::default())?;
            model.finish_prefill_token();
            Ok(logits)
        })
        .collect()
}
