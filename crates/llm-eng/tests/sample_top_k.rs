use llm_eng::sample::sample_logits;
use rand::SeedableRng;

/// The sampler before the top-k selection change: weights for every logit,
/// a full sort, then top-k and top-p.
fn full_sort_reference(logits: &[f32], temperature: f32, top_p: f32, top_k: usize, rng: &mut impl rand::Rng) -> u32 {
    use rand::RngExt;
    let best = (0..logits.len()).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap();
    let mut candidates: Vec<_> = logits
        .iter()
        .enumerate()
        .map(|(index, &logit)| (index, ((logit - logits[best]) / temperature).exp()))
        .collect();
    candidates.sort_unstable_by(|(_, a), (_, b)| b.total_cmp(a));
    candidates.truncate(top_k.min(candidates.len()));
    let total: f32 = candidates.iter().map(|(_, weight)| weight).sum();
    let threshold = top_p.clamp(0.0, 1.0) * total;
    let (mut kept, mut cumulative) = (0, 0.0);
    for (_, weight) in &candidates {
        cumulative += weight;
        kept += 1;
        if cumulative >= threshold {
            break;
        }
    }
    candidates.truncate(kept);
    let total: f32 = candidates.iter().map(|(_, weight)| weight).sum();
    let sample = rng.random_range(0.0..total);
    let mut cumulative = 0.0;
    for (index, weight) in candidates {
        cumulative += weight;
        if sample < cumulative {
            return index as u32;
        }
    }
    best as u32
}

#[test]
fn top_k_selection_samples_like_a_full_sort() {
    // Distinct logits (no ties), so both versions see the same candidates.
    let logits: Vec<f32> = (0..5_000).map(|i| ((i * 7_919) % 5_003) as f32 * 0.003 - 7.0).collect();
    for seed in 0..200 {
        for (temperature, top_p, top_k) in [(1.0, 0.95, 64), (0.7, 1.0, 10), (1.3, 0.5, 5_000), (1.0, 0.9, 1)] {
            let fast = sample_logits(&logits, temperature, top_p, top_k, &mut rand::rngs::StdRng::seed_from_u64(seed));
            let reference = full_sort_reference(&logits, temperature, top_p, top_k, &mut rand::rngs::StdRng::seed_from_u64(seed));
            assert_eq!(fast, reference, "seed {seed}, T {temperature}, top-p {top_p}, top-k {top_k}");
        }
    }
}
