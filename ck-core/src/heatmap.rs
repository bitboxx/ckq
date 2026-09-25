use std::collections::HashSet;

/// Represents the gradient band for semantic heatmap scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeatmapBucket {
    None,
    Step1,
    Step2,
    Step3,
    Step4,
    Step5,
    Step6,
    Step7,
    Step8,
}

impl HeatmapBucket {
    /// Map a similarity score (0.0..=1.0) to a colour bucket.
    pub fn from_score(score: f32) -> Self {
        if score >= 0.875 {
            HeatmapBucket::Step8
        } else if score >= 0.75 {
            HeatmapBucket::Step7
        } else if score >= 0.625 {
            HeatmapBucket::Step6
        } else if score >= 0.5 {
            HeatmapBucket::Step5
        } else if score >= 0.375 {
            HeatmapBucket::Step4
        } else if score >= 0.25 {
            HeatmapBucket::Step3
        } else if score >= 0.125 {
            HeatmapBucket::Step2
        } else if score > 0.0 {
            HeatmapBucket::Step1
        } else {
            HeatmapBucket::None
        }
    }

    /// RGB colour that should be used for the bucket, if any.
    pub fn rgb(self) -> Option<(u8, u8, u8)> {
        match self {
            HeatmapBucket::None => None,
            HeatmapBucket::Step1 => Some((180, 180, 180)),
            HeatmapBucket::Step2 => Some((140, 140, 140)),
            HeatmapBucket::Step3 => Some((100, 130, 100)),
            HeatmapBucket::Step4 => Some((50, 120, 80)),
            HeatmapBucket::Step5 => Some((0, 140, 60)),
            HeatmapBucket::Step6 => Some((0, 160, 70)),
            HeatmapBucket::Step7 => Some((0, 180, 80)),
            HeatmapBucket::Step8 => Some((0, 255, 100)),
        }
    }

    /// Whether the bucket should receive a bold style for additional emphasis.
    pub fn is_bold(self) -> bool {
        matches!(self, HeatmapBucket::Step8)
    }
}

/// Split text into meaningful tokens for heatmap highlighting, preserving spacing
/// and punctuation as discrete tokens so coloured output lines up with the original input.
pub fn split_into_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current_token = String::new();

    for ch in text.chars() {
        match ch {
            ' ' | '\t' | '\n' => {
                if !current_token.is_empty() {
                    tokens.push(current_token.clone());
                    current_token.clear();
                }
                tokens.push(ch.to_string());
            }
            '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' | ':' | '.' | '!' | '?' => {
                if !current_token.is_empty() {
                    tokens.push(current_token.clone());
                    current_token.clear();
                }
                tokens.push(ch.to_string());
            }
            _ => current_token.push(ch),
        }
    }

    if !current_token.is_empty() {
        tokens.push(current_token);
    }

    tokens
}

/// Calculate a similarity score between an individual token and the raw query text.
/// Whitespace and punctuation tokens are ignored and score 0.0.
pub fn calculate_token_similarity(token: &str, pattern: &str) -> f32 {
    if token.trim().is_empty() || token.chars().all(|c| !c.is_alphanumeric()) {
        return 0.0;
    }

    let token_lower = token.to_lowercase();
    let pattern_lower = pattern.to_lowercase();

    if token_lower == pattern_lower {
        return 1.0;
    }

    let pattern_words: Vec<&str> = pattern_lower.split_whitespace().collect();
    let mut max_score: f32 = 0.0;

    for pattern_word in &pattern_words {
        if pattern_word.len() < 3 {
            continue;
        }

        if token_lower == *pattern_word {
            max_score = max_score.max(0.9);
        } else if token_lower.contains(pattern_word) {
            let ratio = pattern_word.len() as f32 / token_lower.len() as f32;
            max_score = max_score.max(0.6 * ratio);
        } else if pattern_word.contains(&token_lower) && token_lower.len() >= 3 {
            let ratio = token_lower.len() as f32 / pattern_word.len() as f32;
            max_score = max_score.max(0.5 * ratio);
        } else {
            let similarity = calculate_fuzzy_similarity(&token_lower, pattern_word);
            max_score = max_score.max(similarity * 0.4);
        }
    }

    max_score
}

fn calculate_fuzzy_similarity(s1: &str, s2: &str) -> f32 {
    if s1.is_empty() || s2.is_empty() || s1.len() < 3 || s2.len() < 3 {
        return 0.0;
    }

    let len1 = s1.len();
    let len2 = s2.len();
    let max_len = len1.max(len2);

    let s1_chars: HashSet<char> = s1.chars().collect();
    let s2_chars: HashSet<char> = s2.chars().collect();
    let common_chars = s1_chars.intersection(&s2_chars).count();

    common_chars as f32 / max_len as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenisation_preserves_spacing_and_punctuation() {
        let tokens = split_into_tokens("fn main() {\n    println!(\"hello\");\n}");
        // Each space is its own token to enable independent highlighting
        assert_eq!(
            tokens,
            vec![
                "fn".to_string(),
                " ".to_string(),
                "main".to_string(),
                "(".to_string(),
                ")".to_string(),
                " ".to_string(),
                "{".to_string(),
                "\n".to_string(),
                " ".to_string(),
                " ".to_string(),
                " ".to_string(),
                " ".to_string(),
                "println".to_string(),
                "!".to_string(),
                "(".to_string(),
                "\"hello\"".to_string(),
                ")".to_string(),
                ";".to_string(),
                "\n".to_string(),
                "}".to_string(),
            ]
        );
    }

    #[test]
    fn similarity_scores_expected_patterns() {
        assert_eq!(calculate_token_similarity("hello", "hello"), 1.0);
        assert!(calculate_token_similarity("hello", "hell") > 0.0);
        assert_eq!(calculate_token_similarity("{", "hello"), 0.0);
    }

    #[test]
    fn heatmap_bucket_mapping_matches_thresholds() {
        assert_eq!(HeatmapBucket::from_score(0.0), HeatmapBucket::None);
        assert_eq!(HeatmapBucket::from_score(0.01), HeatmapBucket::Step1);
        assert_eq!(HeatmapBucket::from_score(0.2), HeatmapBucket::Step2);
        assert_eq!(HeatmapBucket::from_score(0.3), HeatmapBucket::Step3);
        assert_eq!(HeatmapBucket::from_score(0.4), HeatmapBucket::Step4);
        assert_eq!(HeatmapBucket::from_score(0.5), HeatmapBucket::Step5);
        assert_eq!(HeatmapBucket::from_score(0.7), HeatmapBucket::Step6);
        assert_eq!(HeatmapBucket::from_score(0.8), HeatmapBucket::Step7);
        assert_eq!(HeatmapBucket::from_score(0.9), HeatmapBucket::Step8);
    }

    #[test]
    fn bucket_rgb_matches_expected_values() {
        assert_eq!(HeatmapBucket::Step1.rgb(), Some((180, 180, 180)));
        assert_eq!(HeatmapBucket::Step8.rgb(), Some((0, 255, 100)));
        assert!(HeatmapBucket::None.rgb().is_none());
    }
}
