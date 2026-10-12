/// Formats one instruction-tuned Gemma 4 user turn and opens the model turn,
/// as the official chat template does for a single user message with
/// `add_generation_prompt`: `<|turn>user\n{text}<turn|>\n<|turn>model\n`.
/// The template trims the message text.
///
/// Add the beginning-of-sequence token through [`super::tokenizer::Tokenizer::encode`]
/// with `add_bos = true`; never put literal `<bos>` text in the prompt.
pub fn format_user_prompt(prompt: &str) -> String {
    format!("<|turn>user\n{}<turn|>\n<|turn>model\n", prompt.trim())
}
