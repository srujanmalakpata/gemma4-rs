use llm_gemma4::eng::{
    kv_slot, windowed_len, GlobalPolicy, KvSlotError, LocalWindowPolicy,
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
    assert_eq!(hp.kv_slots, 20);
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
fn e2b_kv_slots_cover_twenty_evenly_spaced_groups() {
    assert_eq!(kv_slot(0, 35, 20).unwrap(), 0);
    assert_eq!(kv_slot(34, 35, 20).unwrap(), 19);
    assert_eq!(kv_slot(4, 35, 20).unwrap(), 2);
}

#[test]
fn kv_slot_rejects_invalid_inputs() {
    assert_eq!(kv_slot(0, 0, 20), Err(KvSlotError::ZeroLayers));
    assert_eq!(kv_slot(0, 35, 0), Err(KvSlotError::ZeroSlots));
    assert_eq!(
        kv_slot(35, 35, 20),
        Err(KvSlotError::LayerOutOfRange {
            layer: 35,
            n_layers: 35
        })
    );
}
