//! Splitting a document into the spans that get embedded.
//!
//! Markdown splits at headings first, then packs sections up to the target
//! size; other text packs whole lines. Sizes match the old engine (1,024-token
//! target, 200-token overlap) so the vectors stay comparable with what the
//! benchmarks measured. Token counts are the old engine's estimate
//! (`ck-embed/src/tokenizer.rs`), not a real tokenizer: the model's own
//! context (2,048) leaves room for the error.

use crate::corpus::RelPath;

pub const TARGET_TOKENS: usize = 1024;
pub const OVERLAP_TOKENS: usize = 200;

/// A span of a document. Bytes are a half-open range into the file's UTF-8
/// text, on char boundaries; lines are 1-based and inclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chunk {
    pub byte_start: usize,
    pub byte_end: usize,
    pub line_start: usize,
    pub line_end: usize,
}

/// Chunks of `text`, in order, covering every non-blank line at least once.
/// An empty or blank document gives no chunks.
pub fn chunk(rel: &RelPath, text: &str) -> Vec<Chunk> {
    todo!("markdown when language(rel) == \"markdown\", else line packing")
}

/// Estimated token count, as the old engine counted.
pub fn estimate_tokens(text: &str) -> usize {
    todo!("port ck-embed/src/tokenizer.rs estimate")
}

/// The language label printed in results: "markdown", "rust", "python",
/// "typescript", "javascript", "go", "c", "cpp", "csharp", "ruby", "haskell",
/// "zig", "json", "yaml", "toml", "text". Lowercase, from the extension.
pub fn language(rel: &RelPath) -> &'static str {
    todo!()
}
