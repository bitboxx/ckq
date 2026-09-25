//use anyhow::Result;

/// Simple token estimation for code and text
/// This is a rough approximation since we don't have access to the actual model tokenizer
pub struct TokenEstimator;

impl TokenEstimator {
    /// Estimate token count for text
    /// Based on empirical analysis of code and text tokenization:
    /// - Code: ~4.2 characters per token
    /// - Text: ~4.8 characters per token
    /// - Average: ~4.5 characters per token
    ///
    /// Those ratios hold for Latin scripts. CJK and several Southeast-Asian
    /// scripts run 1-1.5 characters per token, so they are counted separately:
    /// one blended ratio underestimated a CJK chunk by ~3x, the chunk sailed
    /// past every size check and the embedder then silently truncated it.
    pub fn estimate_tokens(text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }

        let char_count = text.chars().count();

        // Detect if text is primarily code vs natural language
        let code_indicators = Self::count_code_indicators(text);
        let total_lines = text.lines().count().max(1);
        let code_density = code_indicators as f32 / total_lines as f32;

        // Adjust ratio based on code density
        let chars_per_token = if code_density > 0.3 {
            // Likely code - more tokens due to symbols, identifiers
            4.2
        } else if code_density > 0.1 {
            // Mixed content
            4.4
        } else {
            // Primarily natural language
            4.8
        };

        let wide = Self::count_wide_script_chars(text);
        let narrow = char_count - wide;
        let tokens = wide as f32 / 1.2 + narrow as f32 / chars_per_token;
        tokens.ceil() as usize
    }

    /// Characters from scripts whose tokenizers spend roughly one token per
    /// character: Han (incl. extensions and compat), kana, Hangul, Thai, Lao,
    /// Khmer and Myanmar.
    fn count_wide_script_chars(text: &str) -> usize {
        fn is_wide(code: u32) -> bool {
            matches!(code,
                0x3400..=0x4DBF   // CJK Extension A
                | 0x4E00..=0x9FFF // CJK Unified
                | 0xF900..=0xFAFF // CJK Compatibility Ideographs
                | 0x20000..=0x2A6DF // CJK Extension B
                | 0x3040..=0x30FF // Hiragana, Katakana
                | 0x1100..=0x11FF // Hangul Jamo
                | 0x3130..=0x318F // Hangul Compatibility Jamo
                | 0xAC00..=0xD7AF // Hangul Syllables
                | 0x0E00..=0x0E7F // Thai
                | 0x0E80..=0x0EFF // Lao
                | 0x1780..=0x17FF // Khmer
                | 0x1000..=0x109F // Myanmar
            )
        }
        text.chars().filter(|c| is_wide(*c as u32)).count()
    }

    /// Check if text exceeds token limit for a given model
    pub fn exceeds_limit(text: &str, max_tokens: usize) -> bool {
        Self::estimate_tokens(text) > max_tokens
    }

    /// Get token limit for different models
    pub fn get_model_limit(model_name: &str) -> usize {
        match model_name {
            "BAAI/bge-small-en-v1.5" => 512,
            "sentence-transformers/all-MiniLM-L6-v2" => 512,
            "nomic-embed-text-v1" => 8192,
            "nomic-embed-text-v1.5" => 8192,
            "jina-embeddings-v2-base-code" => 8192,
            "BAAI/bge-base-en-v1.5" => 512,
            "BAAI/bge-large-en-v1.5" => 512,
            "BAAI/bge-m3" => 8192,
            "onnx-community/Qwen3-Embedding-0.6B-ONNX" => 8192,
            "Qwen/Qwen3-Embedding-0.6B-GGUF" => 8192,
            "bartowski/granite-embedding-278m-multilingual-GGUF" => 512,
            "ggml-org/embeddinggemma-300M-GGUF" | "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF" => {
                2048
            }
            "gpustack/bge-m3-GGUF" => 8192,
            "Xenova/paraphrase-multilingual-MiniLM-L12-v2" => 512,
            "Xenova/paraphrase-multilingual-mpnet-base-v2" => 512,
            _ => 8192, // Default to Nomic limit
        }
    }

    /// Count code-specific indicators to help classify content
    fn count_code_indicators(text: &str) -> usize {
        let mut count = 0;

        for line in text.lines() {
            let trimmed = line.trim();

            // Skip empty lines and comments
            if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
                continue;
            }

            // Look for code patterns
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens_empty() {
        assert_eq!(TokenEstimator::estimate_tokens(""), 0);
    }

    #[test]
    fn test_estimate_tokens_simple() {
        let text = "Hello, world!";
        let tokens = TokenEstimator::estimate_tokens(text);
        // Should be around 3 tokens, estimation might vary
        assert!((2..=4).contains(&tokens), "Got {tokens} tokens");
    }

    #[test]
    fn test_estimate_tokens_cjk_not_underestimated() {
        // CJK tokenizes at ~1-1.5 chars/token; a 1000-char chunk is really
        // ~800+ tokens. The old ~4.5 chars/token ratio called it ~222, the
        // chunk sailed past every size check and got truncated at embed time.
        let text = "語".repeat(1000);
        let tokens = TokenEstimator::estimate_tokens(&text);
        assert!((750..=1100).contains(&tokens), "Got {tokens} tokens");
    }

    #[test]
    fn test_estimate_tokens_mixed_script() {
        // 450 latin chars (~94 tokens at 4.8) plus 100 CJK chars (~83 tokens).
        let text = format!("{} {}", "x".repeat(450), "漢字".repeat(50));
        let tokens = TokenEstimator::estimate_tokens(&text);
        assert!((150..=210).contains(&tokens), "Got {tokens} tokens");
    }

    #[test]
    fn test_estimate_tokens_thai_counts_as_wide() {
        let thai = "ก".repeat(60);
        let tokens = TokenEstimator::estimate_tokens(&thai);
        assert!((45..=60).contains(&tokens), "Got {tokens} tokens");
    }

    #[test]
    fn test_estimate_tokens_code() {
        let code = r#"
fn main() {
    println!("Hello, world!");
    let x = 42;
    return x;
}
"#;
        let tokens = TokenEstimator::estimate_tokens(code);
        // Code typically has more tokens due to symbols
        assert!((15..=25).contains(&tokens), "Got {tokens} tokens");
    }

    #[test]
    fn test_exceeds_limit() {
        assert!(!TokenEstimator::exceeds_limit("short text", 100));

        let long_text = "word ".repeat(200); // ~1000 characters
        assert!(TokenEstimator::exceeds_limit(&long_text, 100));
    }

    #[test]
    fn test_model_limits() {
        assert_eq!(
            TokenEstimator::get_model_limit("BAAI/bge-small-en-v1.5"),
            512
        );
        assert_eq!(
            TokenEstimator::get_model_limit("nomic-embed-text-v1.5"),
            8192
        );
        assert_eq!(TokenEstimator::get_model_limit("unknown-model"), 8192);
    }

    #[test]
    fn test_code_detection() {
        let code = r"
pub fn calculate(x: i32) -> i32 {
    let result = x * 2;
    return result;
}
";
        let tokens = TokenEstimator::estimate_tokens(code);

        let text = r"
This is a paragraph about programming.
It contains some discussion of functions and variables.
But it's written in natural language.
";
        let text_tokens = TokenEstimator::estimate_tokens(text);

        // Code should generally have slightly more tokens per character
        // due to more symbols and shorter identifiers
        let code_ratio = tokens as f32 / code.chars().count() as f32;
        let text_ratio = text_tokens as f32 / text.chars().count() as f32;

        assert!(
            code_ratio >= text_ratio * 0.8,
            "Code ratio {code_ratio} should be similar to or higher than text ratio {text_ratio}"
        );
    }
}
