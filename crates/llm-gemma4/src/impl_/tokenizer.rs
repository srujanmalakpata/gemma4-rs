//! Gemma 4 tokenizer, following llama.cpp's `gemma4` BPE tokenizer:
//!
//! 1. Split special tokens (control, user-defined and unknown types) out of
//!    the text, longest token first.
//! 2. In each remaining piece, replace spaces with `▁` and split it into runs
//!    of newlines and runs of other characters.
//! 3. BPE-merge each run, starting from single characters and always
//!    applying the lowest-ranked merge from `tokenizer.ggml.merges` first.
//! 4. Emit pieces that are not in the vocabulary as `<0xXX>` byte tokens.

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    io::{Read, Seek},
};

use llm_gguf::reader::GgufFile;
use llm_idea::types::TokenId;
use thiserror::Error;

const TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const TYPES_KEY: &str = "tokenizer.ggml.token_type";
const MERGES_KEY: &str = "tokenizer.ggml.merges";

/// `tokenizer.ggml.token_type` values (GGUF / llama.cpp convention).
const TYPE_UNKNOWN: i32 = 2;
const TYPE_CONTROL: i32 = 3;
const TYPE_USER_DEFINED: i32 = 4;
const TYPE_BYTE: i32 = 6;

const SPACE: char = '▁';

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("GGUF metadata `{0}` is missing")]
    MissingMetadata(&'static str),
    #[error("tokenizer metadata lengths differ: tokens={tokens}, token_type={token_types}")]
    MismatchedMetadata { tokens: usize, token_types: usize },
}

pub struct Tokenizer {
    tokens: Vec<String>,
    token_types: Vec<i32>,
    ids: HashMap<String, u32>,
    /// Merge rank keyed by `"left right"`, as stored in the GGUF.
    merge_ranks: HashMap<String, u32>,
    /// Special tokens, longest text first.
    specials: Vec<u32>,
    bos: Option<TokenId>,
    eos: Option<TokenId>,
}

impl Tokenizer {
    pub fn from_gguf<R: Read + Seek>(gguf: &GgufFile<R>) -> Result<Self, TokenizerError> {
        let tokens = gguf
            .metadata_strings(TOKENS_KEY)
            .ok_or(TokenizerError::MissingMetadata(TOKENS_KEY))?;
        let token_types = gguf
            .metadata_i32s(TYPES_KEY)
            .ok_or(TokenizerError::MissingMetadata(TYPES_KEY))?;
        let merges = gguf
            .metadata_strings(MERGES_KEY)
            .ok_or(TokenizerError::MissingMetadata(MERGES_KEY))?;

        Self::from_parts(
            tokens.to_vec(),
            token_types.to_vec(),
            merges,
            gguf.metadata_u32("tokenizer.ggml.bos_token_id"),
            gguf.metadata_u32("tokenizer.ggml.eos_token_id"),
        )
    }

    /// Encodes `text`, recognising special tokens such as `<|turn>` written
    /// in the text (llama.cpp's `parse_special = true`).
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<TokenId> {
        let mut ids = Vec::new();
        if add_bos {
            ids.extend(self.bos);
        }
        for fragment in split_specials(text, &self.specials, &self.tokens) {
            match fragment {
                Fragment::Special(id) => ids.push(TokenId(id)),
                Fragment::Text(text) => {
                    let escaped = text.replace(' ', &SPACE.to_string());
                    for word in split_newline_runs(&escaped) {
                        ids.extend(self.encode_word(word).into_iter().map(TokenId));
                    }
                }
            }
        }
        ids
    }

    pub fn decode(&self, ids: &[TokenId]) -> String {
        let mut output = String::new();
        let mut bytes = Vec::new();

        for TokenId(id) in ids {
            let id = *id as usize;
            let (Some(token), Some(&token_type)) = (self.tokens.get(id), self.token_types.get(id)) else {
                continue;
            };
            if token_type == TYPE_BYTE {
                if let Some(byte) = byte_token(token) {
                    bytes.push(byte);
                    continue;
                }
            }
            flush_bytes(&mut output, &mut bytes);
            if !matches!(token_type, TYPE_CONTROL | TYPE_UNKNOWN) {
                output.push_str(token);
            }
        }
        flush_bytes(&mut output, &mut bytes);
        output.replace(SPACE, " ")
    }

    pub fn bos_token_id(&self) -> Option<TokenId> {
        self.bos
    }

    pub fn eos_token_id(&self) -> Option<TokenId> {
        self.eos
    }

    /// Tokens that end a model turn: `<eos>` and, for chat, `<turn|>`.
    pub fn end_of_generation_ids(&self) -> Vec<TokenId> {
        let turn_end = self.ids.get("<turn|>").map(|&id| TokenId(id));
        self.eos.into_iter().chain(turn_end).collect()
    }

    fn encode_word(&self, word: &str) -> Vec<u32> {
        let pieces = if word.starts_with('\n') && self.ids.contains_key(word) {
            vec![word]
        } else {
            bpe_merge(word, |left, right| self.merge_rank(left, right))
        };
        pieces
            .into_iter()
            .flat_map(|piece| match self.ids.get(piece) {
                Some(&id) => vec![id],
                None => piece
                    .bytes()
                    .filter_map(|byte| self.ids.get(&format!("<0x{byte:02X}>")).copied())
                    .collect(),
            })
            .collect()
    }

    fn merge_rank(&self, left: &str, right: &str) -> Option<u32> {
        self.merge_ranks.get(&format!("{left} {right}")).copied()
    }

    fn from_parts(
        tokens: Vec<String>,
        token_types: Vec<i32>,
        merges: &[String],
        bos: Option<u32>,
        eos: Option<u32>,
    ) -> Result<Self, TokenizerError> {
        if tokens.len() != token_types.len() {
            return Err(TokenizerError::MismatchedMetadata {
                tokens: tokens.len(),
                token_types: token_types.len(),
            });
        }
        let ids: HashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as u32))
            .collect();
        let merge_ranks = merges
            .iter()
            .enumerate()
            .map(|(rank, merge)| (merge.clone(), rank as u32))
            .collect();
        let mut specials: Vec<u32> = (0..tokens.len() as u32)
            .filter(|&id| {
                matches!(
                    token_types[id as usize],
                    TYPE_CONTROL | TYPE_USER_DEFINED | TYPE_UNKNOWN
                )
            })
            .collect();
        specials.sort_by_key(|&id| Reverse(tokens[id as usize].len()));
        let find = |name: &str| ids.get(name).map(|&id| TokenId(id));
        Ok(Self {
            bos: bos.map(TokenId).or_else(|| find("<bos>")),
            eos: eos.map(TokenId).or_else(|| find("<eos>")),
            tokens,
            token_types,
            ids,
            merge_ranks,
            specials,
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Fragment<'a> {
    Text(&'a str),
    Special(u32),
}

/// Splits every occurrence of each special token out of the text, taking
/// the special tokens in the order given (llama.cpp uses longest first).
fn split_specials<'a>(text: &'a str, specials: &[u32], tokens: &[String]) -> Vec<Fragment<'a>> {
    let mut fragments = if text.is_empty() { Vec::new() } else { vec![Fragment::Text(text)] };
    for &id in specials {
        let special = tokens[id as usize].as_str();
        if special.is_empty() {
            continue;
        }
        fragments = fragments
            .into_iter()
            .flat_map(|fragment| match fragment {
                Fragment::Text(text) => split_on(text, special, id),
                other => vec![other],
            })
            .collect();
    }
    fragments
}

fn split_on<'a>(text: &'a str, special: &str, id: u32) -> Vec<Fragment<'a>> {
    let mut fragments = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(special) {
        if start > 0 {
            fragments.push(Fragment::Text(&rest[..start]));
        }
        fragments.push(Fragment::Special(id));
        rest = &rest[start + special.len()..];
    }
    if !rest.is_empty() {
        fragments.push(Fragment::Text(rest));
    }
    fragments
}

/// Splits text into maximal runs of `\n` and maximal runs of anything else
/// (llama.cpp's `[^\n]+|[\n]+` pre-tokenizer for Gemma 4).
fn split_newline_runs(text: &str) -> Vec<&str> {
    let mut words = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    for index in 1..=bytes.len() {
        if index == bytes.len() || (bytes[index] == b'\n') != (bytes[start] == b'\n') {
            words.push(&text[start..index]);
            start = index;
        }
    }
    words
}

/// Byte-pair merging over the characters of `word`. Repeatedly merges the
/// adjacent pair with the lowest rank (leftmost on ties), like llama.cpp's
/// `llm_tokenizer_bpe_session`.
fn bpe_merge(word: &str, rank: impl Fn(&str, &str) -> Option<u32>) -> Vec<&str> {
    // Each symbol is a byte range of `word`; merged-away symbols have len 0.
    struct Symbol {
        start: usize,
        len: usize,
        prev: Option<usize>,
        next: Option<usize>,
    }
    type Queue = BinaryHeap<Reverse<(u32, usize, usize, usize)>>;

    let mut symbols: Vec<Symbol> = word
        .char_indices()
        .map(|(start, ch)| Symbol { start, len: ch.len_utf8(), prev: None, next: None })
        .collect();
    let count = symbols.len();
    for (index, symbol) in symbols.iter_mut().enumerate() {
        symbol.prev = index.checked_sub(1);
        symbol.next = (index + 1 < count).then_some(index + 1);
    }
    let text = |symbol: &Symbol| &word[symbol.start..symbol.start + symbol.len];

    // Min-heap of (rank, left, left_len, right_len).
    let push = |queue: &mut Queue, symbols: &[Symbol], left: usize| {
        if let Some(right) = symbols[left].next {
            if let Some(rank) = rank(text(&symbols[left]), text(&symbols[right])) {
                queue.push(Reverse((rank, left, symbols[left].len, symbols[right].len)));
            }
        }
    };
    let mut queue = Queue::new();
    for left in 0..count {
        push(&mut queue, &symbols, left);
    }

    while let Some(Reverse((_, left, left_len, right_len))) = queue.pop() {
        let Some(right) = symbols[left].next else { continue };
        // Skip pairs made stale by an earlier merge.
        if left_len == 0 || symbols[left].len != left_len || symbols[right].len != right_len {
            continue;
        }
        symbols[left].len += right_len;
        symbols[right].len = 0;
        symbols[left].next = symbols[right].next;
        if let Some(next) = symbols[right].next {
            symbols[next].prev = Some(left);
        }
        if let Some(prev) = symbols[left].prev {
            push(&mut queue, &symbols, prev);
        }
        push(&mut queue, &symbols, left);
    }

    symbols.iter().filter(|symbol| symbol.len > 0).map(text).collect()
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

    const NORMAL: i32 = 1;

    fn tokenizer() -> Tokenizer {
        let vocab = [
            ("<pad>", TYPE_CONTROL),
            ("<eos>", TYPE_CONTROL),
            ("<bos>", TYPE_CONTROL),
            ("<|turn>", TYPE_CONTROL),
            ("<turn|>", TYPE_CONTROL),
            ("▁", NORMAL),
            ("h", NORMAL),
            ("i", NORMAL),
            ("hi", NORMAL),
            ("▁hi", NORMAL),
            ("\n", NORMAL),
            ("\n\n", NORMAL),
            ("<0xC3>", TYPE_BYTE),
            ("<0xA9>", TYPE_BYTE),
        ];
        let merges = ["h i", "▁ hi"].map(String::from);
        Tokenizer::from_parts(
            vocab.iter().map(|(token, _)| token.to_string()).collect(),
            vocab.iter().map(|&(_, kind)| kind).collect(),
            &merges,
            Some(2),
            Some(1),
        )
        .unwrap()
    }

    fn ids(values: &[u32]) -> Vec<TokenId> {
        values.iter().copied().map(TokenId).collect()
    }

    #[test]
    fn merges_by_rank_and_maps_spaces() {
        assert_eq!(tokenizer().encode("hi hi", true), ids(&[2, 8, 9]));
    }

    #[test]
    fn splits_special_tokens_out_of_text() {
        assert_eq!(
            tokenizer().encode("<|turn>hi<turn|>\n", false),
            ids(&[3, 8, 4, 10])
        );
    }

    #[test]
    fn keeps_newline_runs_whole_when_the_vocab_has_them() {
        assert_eq!(tokenizer().encode("hi\n\nhi", false), ids(&[8, 11, 8]));
    }

    #[test]
    fn falls_back_to_byte_tokens() {
        assert_eq!(tokenizer().encode("é", false), ids(&[12, 13]));
    }

    #[test]
    fn decode_skips_control_tokens_and_joins_bytes() {
        let tokenizer = tokenizer();
        assert_eq!(tokenizer.decode(&ids(&[2, 8, 9, 12, 13, 4])), "hi hié");
    }

    #[test]
    fn end_of_generation_includes_turn_end() {
        assert_eq!(tokenizer().end_of_generation_ids(), ids(&[1, 4]));
    }

    #[test]
    fn bpe_prefers_the_leftmost_pair_on_equal_rank() {
        let rank = |left: &str, right: &str| (left == "a" && right == "a").then_some(0);
        assert_eq!(bpe_merge("aaa", rank), vec!["aa", "a"]);
    }
}
