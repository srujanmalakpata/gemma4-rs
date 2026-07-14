use llm_gemma4::eng::{kv_slot, windowed_len, GlobalPolicy, LocalWindowPolicy};
use llm_gemma4::idea::{layer_attention_kind, AttentionKind, Gemma4HyperParams};

#[test]
fn e2b_uses_4_local_then_1_global_and_final_global() {
    let hp = Gemma4HyperParams::e2b();
    assert_eq!(hp.n_layers, 35);
    assert_eq!(hp.sliding_window, 512);
    assert_eq!(
        layer_attention_kind(&hp, 0),
        AttentionKind::Local { window: 512 }
    );
    assert_eq!(
        layer_attention_kind(&hp, 3),
        AttentionKind::Local { window: 512 }
    );
    assert_eq!(layer_attention_kind(&hp, 4), AttentionKind::Global);
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
    assert_eq!(kv_slot(0, 35, 20), 0);
    assert_eq!(kv_slot(34, 35, 20), 19);
    assert_eq!(kv_slot(4, 35, 20), 2);
}
