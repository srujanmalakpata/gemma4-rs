use std::io::{Read, Seek};

use llm_gguf::reader::GgufFile;
use llm_idea::types::TokenId;
use thiserror::Error;

const TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const SCORES_KEY: &str = "tokenizer.ggml.scores";
const TYPES_KEY: &str = "tokenizer.ggml.token_type";

/// SentencePiece `tokenizer.ggml.token_type` values (GGUF / llama.cpp convention).
const SP_NORMAL: i32 = 1;
const SP_UNKNOWN: i32 = 2;
const SP_CONTROL: i32 = 3;
const SP_USER_DEFINED: i32 = 4;
const SP_UNUSED: i32 = 5;
const SP_BYTE: i32 = 6;

/// Types eligible for longest-prefix greedy matching on user text.
/// CONTROL, UNKNOWN, UNUSED, and USER_DEFINED are special/reserved; BYTE is kept
/// for the unknown-char fallback path (not greedy prefix match).
fn is_greedy_matchable(token_type: i32) -> bool {
    if matches!(
        token_type,
        SP_UNKNOWN | SP_CONTROL | SP_USER_DEFINED | SP_UNUSED | SP_BYTE
    ) {
        return false;
    }
    token_type == SP_NORMAL
}

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("GGUF metadata `{0}` is missing")]
    MissingMetadata(&'static str),
    #[error("tokenizer metadata lengths differ: tokens={tokens}, scores={scores}, token_type={token_types}")]
    MismatchedMetadata {
        tokens: usize,
        scores: usize,
        token_types: usize,
    },
}

pub struct Tokenizer {
    tokens: Vec<String>,
    scores: Vec<f32>,
    token_types: Vec<i32>,
    bos: Option<TokenId>,
    eos: Option<TokenId>,
    unknown: Option<TokenId>,
}

impl Tokenizer {
    pub fn from_gguf<R: Read + Seek>(gguf: &GgufFile<R>) -> Result<Self, TokenizerError> {
        let tokens = gguf
            .metadata_strings(TOKENS_KEY)
            .ok_or(TokenizerError::MissingMetadata(TOKENS_KEY))?;
        let scores = gguf
            .metadata_f32s(SCORES_KEY)
            .ok_or(TokenizerError::MissingMetadata(SCORES_KEY))?;
        let token_types = gguf
            .metadata_i32s(TYPES_KEY)
            .ok_or(TokenizerError::MissingMetadata(TYPES_KEY))?;

        Self::from_parts(
            tokens.to_vec(),
            scores.to_vec(),
            token_types.to_vec(),
            gguf.metadata_u32("tokenizer.ggml.bos_token_id"),
            gguf.metadata_u32("tokenizer.ggml.eos_token_id"),
        )
    }

    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<TokenId> {
        // ponytail: greedy SP encode — upgrade to full SentencePiece proto if mismatch vs golden; O(chars×vocab) scan per step
        let mut ids = Vec::new();
        if add_bos {
            ids.extend(self.bos);
        }

        let normalized = text.replace(' ', "▁");
        let mut remaining = normalized.as_str();
        while !remaining.is_empty() {
            let best = self
                .tokens
                .iter()
                .enumerate()
                .filter(|(id, token)| {
                    is_greedy_matchable(self.token_types[*id])
                        && !is_special(token)
                        && remaining.starts_with(token.as_str())
                })
                .max_by(|(left_id, left), (right_id, right)| {
                    left.len()
                        .cmp(&right.len())
                        .then_with(|| self.scores[*left_id].total_cmp(&self.scores[*right_id]))
                });

            if let Some((id, token)) = best {
                ids.push(TokenId(id as u32));
                remaining = &remaining[token.len()..];
            } else {
                let next = remaining.chars().next().expect("remaining is non-empty");
                let byte_id = self.tokens.iter().enumerate().find_map(|(id, token)| {
                    (self.token_types[id] == SP_BYTE && byte_token(token) == Some(next as u8))
                        .then_some(TokenId(id as u32))
                });
                ids.extend(byte_id.or(self.unknown));
                remaining = &remaining[next.len_utf8()..];
            }
        }
        ids
    }

    pub fn decode(&self, ids: &[TokenId]) -> String {
        let mut output = String::new();
        let mut byte_fallback = Vec::new();

        for TokenId(id) in ids {
            let Some(token) = self.tokens.get(*id as usize) else {
                continue;
            };
            if let Some(byte) = byte_token(token) {
                byte_fallback.push(byte);
                continue;
            }
            flush_bytes(&mut output, &mut byte_fallback);
            if !is_special(token) {
                output.push_str(token);
            }
        }
        flush_bytes(&mut output, &mut byte_fallback);
        output.replace('▁', " ").trim_start().to_owned()
    }

    pub fn bos_token_id(&self) -> Option<TokenId> {
        self.bos
    }

    pub fn eos_token_id(&self) -> Option<TokenId> {
        self.eos
    }

    fn from_parts(
        tokens: Vec<String>,
        scores: Vec<f32>,
        token_types: Vec<i32>,
        bos: Option<u32>,
        eos: Option<u32>,
    ) -> Result<Self, TokenizerError> {
        if tokens.len() != scores.len() || tokens.len() != token_types.len() {
            return Err(TokenizerError::MismatchedMetadata {
                tokens: tokens.len(),
                scores: scores.len(),
                token_types: token_types.len(),
            });
        }
        let find = |names: &[&str]| {
            tokens
                .iter()
                .position(|token| names.contains(&token.as_str()))
                .map(|id| TokenId(id as u32))
        };
        Ok(Self {
            unknown: find(&["<unk>"]),
            bos: bos.map(TokenId).or_else(|| find(&["<bos>", "<s>"])),
            eos: eos.map(TokenId).or_else(|| find(&["<eos>", "</s>"])),
            tokens,
            scores,
            token_types,
        })
    }
}

fn is_special(token: &str) -> bool {
    matches!(token, "<unk>" | "<bos>" | "<eos>" | "<s>" | "</s>")
}

fn byte_token(token: &str) -> Option<u8> {
    token
        .strip_prefix("<0x")
        .and_then(|hex| hex.strip_suffix('>'))
        .filter(|hex| hex.len() == 2)
        .and_then(|hex| u8::from_str_radix(hex, 16).ok())
}

fn flush_bytes(output: &mut String, bytes: &mut Vec<u8>) {
    if !bytes.is_empty() {
        output.push_str(&String::from_utf8_lossy(bytes));
        bytes.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedily_encodes_and_decodes_sentencepiece_tokens() {
        let tokenizer = Tokenizer::from_parts(
            vec![
                "<unk>".into(),
                "<bos>".into(),
                "<eos>".into(),
                "▁hello".into(),
                "▁".into(),
                "world".into(),
            ],
            vec![0.0; 6],
            vec![
                SP_UNKNOWN,
                SP_CONTROL,
                SP_CONTROL,
                SP_NORMAL,
                SP_NORMAL,
                SP_NORMAL,
            ],
            None,
            None,
        )
        .unwrap();

        assert_eq!(
            tokenizer.encode(" hello world", true),
            vec![TokenId(1), TokenId(3), TokenId(4), TokenId(5)]
        );
        assert_eq!(
            tokenizer.decode(&[TokenId(3), TokenId(4), TokenId(5)]),
            "hello world"
        );
    }
}
