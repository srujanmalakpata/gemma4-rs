use llm_eng::context_memory::{ContextMemory, FullRecompute, KvCache};

/// Toy attention: score = q·k, output = score * v (scalars).
fn toy_attend(q: f32, ks: &[f32], vs: &[f32]) -> f32 {
    assert_eq!(ks.len(), vs.len());
    let mut acc = 0.0;
    for i in 0..ks.len() {
        acc += (q * ks[i]) * vs[i];
    }
    acc
}

#[test]
fn kv_cache_matches_full_recompute_on_scalar_fixture() {
    let qs = [1.0_f32, 2.0, 3.0];
    let ks = [0.5_f32, -1.0, 2.0];
    let vs = [1.0_f32, 2.0, 0.5];

    let mut recompute = FullRecompute::<f32, f32>::default();
    let mut kv = KvCache::<f32, f32>::default();

    let mut out_r = Vec::new();
    let mut out_k = Vec::new();
    for t in 0..3 {
        recompute.append(0, ks[t], vs[t]).unwrap();
        kv.append(0, ks[t], vs[t]).unwrap();
        let (k_r, v_r) = recompute.view(0, t + 1).unwrap();
        let (k_k, v_k) = kv.view(0, t + 1).unwrap();
        out_r.push(toy_attend(qs[t], k_r, v_r));
        out_k.push(toy_attend(qs[t], k_k, v_k));
    }
    for i in 0..3 {
        assert!((out_r[i] - out_k[i]).abs() < 1e-6);
    }
}

#[test]
fn zero_temperature_sampling_selects_the_highest_logit() {
    let mut rng = rand::rng();

    assert_eq!(
        llm_eng::sample::sample_logits(&[0.5, 4.0, 1.0], 0.0, 0.95, 64, &mut rng),
        1
    );
}
