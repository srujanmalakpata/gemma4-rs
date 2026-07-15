use llm_gemma4::impl_::chat_template::format_user_prompt;

#[test]
fn wraps_user_turn_with_gemma4_markers() {
    let s = format_user_prompt("Hello");
    assert!(s.contains("<|turn|>user"));
    assert!(s.contains("Hello"));
    assert!(s.contains("<turn|>"));
    assert!(s.contains("<|turn|>model"));
}
