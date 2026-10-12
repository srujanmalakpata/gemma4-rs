use llm_gemma4::impl_::chat_template::format_user_prompt;

#[test]
fn wraps_user_turn_with_gemma4_markers() {
    assert_eq!(
        format_user_prompt("Hello"),
        "<|turn>user\nHello<turn|>\n<|turn>model\n"
    );
}

#[test]
fn trims_the_message_like_the_official_template() {
    assert_eq!(
        format_user_prompt("  Hello \n"),
        "<|turn>user\nHello<turn|>\n<|turn>model\n"
    );
}
