/// Formats an instruction-tuned Gemma 4 user turn.
///
/// Add the beginning-of-sequence token through [`super::tokenizer::Tokenizer::encode`]
/// with `add_bos = true`; never put literal `[BOS]` text in the prompt.
pub fn format_user_prompt(prompt: &str) -> String {
    format!("<|turn|>user\n{prompt}<turn|>\n<|turn|>model\n")
}
