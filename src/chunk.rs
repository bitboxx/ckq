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

/// A run of whole lines, borrowed from the document: the unit the packer
/// assembles chunks from.
#[derive(Debug, Clone, Copy)]
struct Unit<'a> {
    text: &'a str,
    byte_start: usize,
    byte_end: usize,
    line_start: usize,
    line_end: usize,
    tokens: usize,
}

impl<'a> Unit<'a> {
    fn merge_group(group: &[Unit<'a>]) -> Unit<'a> {
        Unit {
            text: "",
            byte_start: group[0].byte_start,
            byte_end: group[group.len() - 1].byte_end,
            line_start: group[0].line_start,
            line_end: group[group.len() - 1].line_end,
            tokens: group.iter().map(|u| u.tokens).sum(),
        }
    }
}

/// Chunks of `text`, in order, covering every non-blank line at least once.
/// An empty or blank document gives no chunks.
pub fn chunk(rel: &RelPath, text: &str) -> Vec<Chunk> {
    let lines = split_lines(text);
    if lines.is_empty() {
        return Vec::new();
    }
    let units = if language(rel) == "markdown" {
        split_sections(&lines)
    } else {
        lines.clone()
    };
    pack(units)
        .into_iter()
        .map(|u| Chunk {
            byte_start: u.byte_start,
            byte_end: u.byte_end,
            line_start: u.line_start,
            line_end: u.line_end,
        })
        .collect()
}

/// One unit per non-blank line: byte end excludes the line break. Blank lines
/// are dropped: chunks need not cover them, and dropping them keeps an
/// all-blank document chunk-free.
fn split_lines(text: &str) -> Vec<Unit<'_>> {
    let mut out = Vec::new();
    let mut offset = 0usize;
    for (i, raw) in text.split_inclusive('\n').enumerate() {
        let trimmed = raw.trim_end_matches(['\n', '\r']);
        let start = offset;
        let end = start + trimmed.len();
        offset += raw.len();
        if trimmed.trim().is_empty() {
            continue;
        }
        out.push(Unit {
            text: trimmed,
            byte_start: start,
            byte_end: end,
            line_start: i + 1,
            line_end: i + 1,
            tokens: estimate_tokens(trimmed),
        });
    }
    out
}

/// Markdown sections: a section is a heading line plus everything up to the
/// next heading of any level, ignoring headings inside fenced code blocks. A
/// section too big for one chunk is hard-split back into lines.
fn split_sections<'a>(lines: &[Unit<'a>]) -> Vec<Unit<'a>> {
    let mut sections: Vec<Vec<Unit<'_>>> = Vec::new();
    let mut fence = false;
    for line in lines {
        let t = line.text.trim_start();
        if t.starts_with("```") || t.starts_with("~~~") {
            fence = !fence;
        }
        let is_heading = !fence && t.starts_with('#') && t[1..].starts_with([' ', '#']);
        if is_heading || sections.is_empty() {
            sections.push(vec![*line]);
        } else {
            sections.last_mut().expect("non-empty").push(*line);
        }
    }
    let mut units = Vec::new();
    for group in sections {
        let tokens: usize = group.iter().map(|u| u.tokens).sum();
        if tokens > TARGET_TOKENS {
            units.extend(group);
        } else {
            units.push(Unit::merge_group(&group));
        }
    }
    units
}

/// Accumulate units into chunks of at most `TARGET_TOKENS`, starting each new
/// chunk with the tail of the previous one, up to `OVERLAP_TOKENS`.
fn pack(mut units: Vec<Unit<'_>>) -> Vec<Unit<'_>> {
    let mut chunks: Vec<Unit<'_>> = Vec::new();
    let mut cur: Vec<Unit<'_>> = Vec::new();
    let mut cur_tokens = 0usize;
    for unit in units.drain(..) {
        if !cur.is_empty() && cur_tokens + unit.tokens > TARGET_TOKENS {
            chunks.push(Unit::merge_group(&cur));
            let mut overlap: Vec<Unit<'_>> = Vec::new();
            let mut overlap_tokens = 0usize;
            for u in cur.iter().rev() {
                if overlap_tokens + u.tokens > OVERLAP_TOKENS {
                    break;
                }
                overlap.insert(0, *u);
                overlap_tokens += u.tokens;
            }
            cur = overlap;
            cur_tokens = overlap_tokens;
        }
        cur_tokens += unit.tokens;
        cur.push(unit);
    }
    if !cur.is_empty() {
        chunks.push(Unit::merge_group(&cur));
    }
    chunks
}

/// Estimated token count, as the old engine counted.
pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    let char_count = text.chars().count();
    let code_indicators = count_code_indicators(text);
    let total_lines = text.lines().count().max(1);
    let code_density = code_indicators as f32 / total_lines as f32;
    let chars_per_token = if code_density > 0.3 {
        4.2 // likely code
    } else if code_density > 0.1 {
        4.4 // mixed
    } else {
        4.8 // natural language
    };
    let wide = count_wide_script_chars(text);
    let narrow = char_count - wide;
    let tokens = wide as f32 / 1.2 + narrow as f32 / chars_per_token;
    tokens.ceil() as usize
}

/// Characters from scripts whose tokenizers spend roughly one token per
/// character: Han, kana, Hangul, Thai, Lao, Khmer and Myanmar.
fn count_wide_script_chars(text: &str) -> usize {
    fn is_wide(code: u32) -> bool {
        matches!(code,
            0x3400..=0x4DBF     // CJK Extension A
            | 0x4E00..=0x9FFF   // CJK Unified
            | 0xF900..=0xFAFF   // CJK Compatibility Ideographs
            | 0x20000..=0x2A6DF // CJK Extension B
            | 0x3040..=0x30FF   // Hiragana, Katakana
            | 0x1100..=0x11FF   // Hangul Jamo
            | 0x3130..=0x318F   // Hangul Compatibility Jamo
            | 0xAC00..=0xD7AF   // Hangul Syllables
            | 0x0E00..=0x0E7F   // Thai
            | 0x0E80..=0x0EFF   // Lao
            | 0x1780..=0x17FF   // Khmer
            | 0x1000..=0x109F   // Myanmar
        )
    }
    text.chars().filter(|c| is_wide(*c as u32)).count()
}

fn count_code_indicators(text: &str) -> usize {
    let mut count = 0;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
            continue;
        }
        if trimmed.contains('{') || trimmed.contains('}') {
            count += 1;
        }
        if trimmed.contains(';') && !trimmed.ends_with('.') {
            count += 1;
        }
        if trimmed.contains("fn ")
            || trimmed.contains("def ")
            || trimmed.contains("function ")
            || trimmed.contains("func ")
        {
            count += 1;
        }
        if trimmed.contains("->") || trimmed.contains("=>") || trimmed.contains("::") {
            count += 1;
        }
        if trimmed.starts_with("pub ")
            || trimmed.starts_with("private ")
            || trimmed.starts_with("public ")
        {
            count += 1;
        }
    }
    count
}

/// The language label printed in results: "markdown", "rust", "python",
/// "typescript", "javascript", "go", "c", "cpp", "csharp", "ruby", "haskell",
/// "zig", "json", "yaml", "toml", "text". Lowercase, from the extension.
pub fn language(rel: &RelPath) -> &'static str {
    match rel.extension().as_deref() {
        Some("md") | Some("markdown") => "markdown",
        Some("rs") => "rust",
        Some("py") => "python",
        Some("ts") | Some("tsx") | Some("mts") | Some("cts") => "typescript",
        Some("js") | Some("jsx") | Some("mjs") | Some("cjs") => "javascript",
        Some("go") => "go",
        Some("c") | Some("h") => "c",
        Some("cpp") | Some("cc") | Some("cxx") | Some("hpp") | Some("hxx") => "cpp",
        Some("cs") => "csharp",
        Some("rb") => "ruby",
        Some("hs") => "haskell",
        Some("zig") => "zig",
        Some("json") => "json",
        Some("yaml") | Some("yml") => "yaml",
        Some("toml") => "toml",
        _ => "text",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(p: &str) -> RelPath {
        RelPath::new(std::path::Path::new(p)).unwrap()
    }

    #[test]
    fn empty_and_blank_documents_chunk_to_nothing() {
        assert!(chunk(&rel("a.md"), "").is_empty());
        assert!(chunk(&rel("a.md"), "\n\n  \n").is_empty());
    }

    #[test]
    fn a_short_document_is_one_chunk_covering_its_lines() {
        let text = "# Title\n\nOne paragraph.\n";
        let chunks = chunk(&rel("a.md"), text);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].line_start, 1);
        assert_eq!(chunks[0].line_end, 3);
        assert_eq!(
            &text[chunks[0].byte_start..chunks[0].byte_end],
            "# Title\n\nOne paragraph."
        );
    }

    #[test]
    fn markdown_sections_split_at_headings() {
        let mut text = String::new();
        for i in 0..8 {
            text.push_str(&format!("# Section {i}\n"));
            text.push_str(&"words of body text here. ".repeat(60));
            text.push('\n');
        }
        let chunks = chunk(&rel("a.md"), &text);
        assert!(
            chunks.len() >= 2,
            "eight sections must split: {}",
            chunks.len()
        );
        // Each chunk after the first starts on a heading line or in the
        // overlap tail of the previous chunk, never mid-heading.
        for c in &chunks[1..] {
            let first = text[c.byte_start..].lines().next().unwrap();
            assert!(
                first.starts_with('#') || !first.contains("Section"),
                "chunk starts mid-content: {first:?}"
            );
        }
    }

    #[test]
    fn every_line_of_a_big_file_is_covered() {
        let mut text = String::new();
        for i in 1..=4200 {
            text.push_str(&format!(
                "Line {i}: the archivist shelves another bound volume of chronicles.\n"
            ));
        }
        let chunks = chunk(&rel("big.md"), &text);
        assert!(chunks.len() > 5);
        // Index i-1 is line i: every line of the file must fall in some chunk.
        let mut covered = vec![false; 4200];
        for c in &chunks {
            for seen in covered.iter_mut().take(c.line_end).skip(c.line_start - 1) {
                *seen = true;
            }
            assert!(c.byte_end <= text.len());
        }
        assert!(covered.iter().all(|c| *c), "some line uncovered");
    }

    #[test]
    fn token_estimate_counts_wide_scripts_at_full_rate() {
        let latin = "hello world of text"; // 19 chars
        let cjk = "山中的寺庙藏于云雾之间"; // 11 wide chars
        assert!(estimate_tokens(cjk) >= 9, "wide script undercounted");
        assert!(estimate_tokens(latin) >= 4);
        assert_eq!(estimate_tokens(""), 0);
    }
}
