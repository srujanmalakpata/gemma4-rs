use llm_gemma4::eng::{
    kv_source_layer, windowed_len, GlobalPolicy, KvShareError, LocalWindowPolicy,
};
use llm_gemma4::idea::{layer_attention_kind, AttentionKind, Gemma4HyperParams};

#[test]
fn e2b_locked_hyperparams() {
    let hp = Gemma4HyperParams::e2b();
    assert_eq!(hp.n_layers, 35);
    assert_eq!(hp.vocab_size, 262_144);
    assert_eq!(hp.d_model, 1_536);
    assert_eq!(hp.ple_dim, 256);
    assert_eq!(hp.sliding_window, 512);
    assert_eq!(hp.shared_kv_layers, 20);
    assert_eq!(hp.local_rope_base, 10_000.0);
    assert_eq!(hp.global_p_rope.p, 0.25);
    assert_eq!(hp.global_p_rope.base, 1_000_000.0);
}

#[test]
fn e2b_uses_4_local_then_1_global_and_final_global() {
    let hp = Gemma4HyperParams::e2b();
    for layer in 0..hp.n_layers {
        let expected = if layer % 5 == 4 {
            AttentionKind::Global
        } else {
            AttentionKind::Local {
                window: hp.sliding_window,
            }
        };
        assert_eq!(layer_attention_kind(&hp, layer), expected, "layer {layer}");
    }
    assert_eq!(layer_attention_kind(&hp, 34), AttentionKind::Global);
}

#[test]
fn policies_select_the_expected_context_ranges() {
    assert_eq!(windowed_len(1000, 512), 488..1000);
    assert_eq!(LocalWindowPolicy::new(512).range(1000), 488..1000);
    assert_eq!(GlobalPolicy.range(1000), 0..1000);
}

#[test]
fn e2b_last_twenty_layers_reuse_layer_13_or_14() {
    let hp = Gemma4HyperParams::e2b();
    for layer in 0..15 {
        assert_eq!(kv_source_layer(&hp, layer), Ok(layer), "layer {layer}");
    }
    for layer in 15..35 {
        let expected = match layer_attention_kind(&hp, layer) {
            AttentionKind::Local { .. } => 13,
            AttentionKind::Global => 14,
        };
        assert_eq!(kv_source_layer(&hp, layer), Ok(expected), "layer {layer}");
    }
}

#[test]
fn kv_source_layer_rejects_invalid_inputs() {
    let hp = Gemma4HyperParams::e2b();
    assert_eq!(
        kv_source_layer(&hp, 35),
        Err(KvShareError::LayerOutOfRange { layer: 35, n_layers: 35 })
    );
    let all_shared = Gemma4HyperParams { shared_kv_layers: 35, ..hp.clone() };
    assert_eq!(
        kv_source_layer(&all_shared, 0),
        Err(KvShareError::TooManySharedLayers { shared: 35, n_layers: 35 })
    );
    let no_global_owner = Gemma4HyperParams { shared_kv_layers: 31, ..hp };
    assert_eq!(
        kv_source_layer(&no_global_owner, 34),
        Err(KvShareError::NoEarlierLayerOfSameKind { layer: 34 })
    );
}
