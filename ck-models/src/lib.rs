use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub provider: String,
    pub dimensions: usize,
    pub max_tokens: usize,
    pub description: String,
    /// Which file to pull from the repo. Only meaningful for the llamacpp
    /// provider, where one GGUF repo holds several quantizations.
    #[serde(default)]
    pub gguf_file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistry {
    pub models: HashMap<String, ModelConfig>,
    pub default_model: String,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        let mut models = HashMap::new();

        // Add enhanced models
        // Multilingual models. These three need no instruction prefix, which
        // matters because fastembed does not prepend one (see bge-m3 vs e5 below).
        models.insert(
            "bge-m3-gguf".to_string(),
            ModelConfig {
                name: "gpustack/bge-m3-GGUF".to_string(),
                provider: "llamacpp".to_string(),
                dimensions: 1024,
                max_tokens: 8192,
                description: "BGE-M3 multilingual via llama.cpp, 8k context".to_string(),
                gguf_file: "bge-m3-Q8_0.gguf".to_string(),
            },
        );

        models.insert(
            "gemma-q4".to_string(),
            ModelConfig {
                name: "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF".to_string(),
                provider: "llamacpp".to_string(),
                dimensions: 768,
                max_tokens: 2048,
                description: "EmbeddingGemma 300M, quantization-aware trained Q4_0".to_string(),
                gguf_file: "embeddinggemma-300M-qat-Q4_0.gguf".to_string(),
            },
        );

        models.insert(
            "granite-gguf".to_string(),
            ModelConfig {
                name: "bartowski/granite-embedding-278m-multilingual-GGUF".to_string(),
                provider: "llamacpp".to_string(),
                dimensions: 768,
                max_tokens: 512,
                description:
                    "IBM Granite 278M multilingual, Apache-2.0, no instruction prefix needed"
                        .to_string(),
                gguf_file: "granite-embedding-278m-multilingual-Q8_0.gguf".to_string(),
            },
        );

        models.insert(
            "gemma-gguf".to_string(),
            ModelConfig {
                name: "ggml-org/embeddinggemma-300M-GGUF".to_string(),
                provider: "llamacpp".to_string(),
                dimensions: 768,
                max_tokens: 2048,
                description: "EmbeddingGemma 300M, 100+ languages (wants a task prefix)"
                    .to_string(),
                gguf_file: "embeddinggemma-300M-Q8_0.gguf".to_string(),
            },
        );

        models.insert(
            "qwen3-gguf".to_string(),
            ModelConfig {
                name: "Qwen/Qwen3-Embedding-0.6B-GGUF".to_string(),
                provider: "llamacpp".to_string(),
                dimensions: 1024,
                max_tokens: 8192,
                description: "Qwen3-Embedding 0.6B via llama.cpp, GPU-accelerated".to_string(),
                gguf_file: "Qwen3-Embedding-0.6B-Q8_0.gguf".to_string(),
            },
        );

        Self {
            models,
            // ckq defaults to a multilingual model. bge-small is English-only and
            // scored 2/4 on the trilingual fixture where this scores 4/4, at
            // roughly stock ck's search latency.
            default_model: "granite-gguf".to_string(),
        }
    }
}

impl ModelRegistry {
    fn format_available_models(&self) -> String {
        self.models.keys().cloned().collect::<Vec<_>>().join(", ")
    }

    fn resolve_alias_or_name(&self, key: &str) -> Option<(String, &ModelConfig)> {
        if let Some(config) = self.models.get(key) {
            return Some((key.to_string(), config));
        }

        self.models
            .iter()
            .find(|(_, config)| config.name == key)
            .map(|(alias, config)| (alias.clone(), config))
    }

    pub fn resolve(&self, requested: Option<&str>) -> Result<(String, ModelConfig)> {
        match requested {
            Some(name) => {
                let (alias, config) = self.resolve_alias_or_name(name).ok_or_else(|| {
                    anyhow!(
                        "Unknown model '{}'. Available models: {}",
                        name,
                        self.format_available_models()
                    )
                })?;
                Ok((alias, config.clone()))
            }
            None => {
                let alias = self.default_model.clone();
                let config = self
                    .get_default_model()
                    .cloned()
                    .ok_or_else(|| anyhow!("No default model configured in registry"))?;
                Ok((alias, config))
            }
        }
    }

    pub fn aliases(&self) -> Vec<String> {
        let mut keys = self.models.keys().cloned().collect::<Vec<_>>();
        keys.sort();
        keys
    }

    pub fn load(path: &Path) -> Result<Self> {
        if path.exists() {
            let data = std::fs::read_to_string(path)?;
            Ok(serde_json::from_str(&data)?)
        } else {
            Ok(Self::default())
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    pub fn get_model(&self, name: &str) -> Option<&ModelConfig> {
        self.models.get(name)
    }

    pub fn get_default_model(&self) -> Option<&ModelConfig> {
        self.models.get(&self.default_model)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankModelConfig {
    pub name: String,
    pub provider: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankModelRegistry {
    pub models: HashMap<String, RerankModelConfig>,
    pub default_model: String,
}

impl Default for RerankModelRegistry {
    fn default() -> Self {
        let mut models = HashMap::new();

        models.insert(
            "jina".to_string(),
            RerankModelConfig {
                name: "jina-reranker-v1-turbo-en".to_string(),
                provider: "fastembed".to_string(),
                description:
                    "Jina Turbo reranker (default) tuned for English code + text relevance"
                        .to_string(),
            },
        );

        models.insert(
            "bge".to_string(),
            RerankModelConfig {
                name: "BAAI/bge-reranker-base".to_string(),
                provider: "fastembed".to_string(),
                description: "BGE reranker base model for multilingual use cases".to_string(),
            },
        );

        models.insert(
            "mxbai".to_string(),
            RerankModelConfig {
                name: "mixedbread-ai/mxbai-rerank-xsmall-v1".to_string(),
                provider: "mixedbread".to_string(),
                description: "Mixedbread xsmall reranker (quantized) optimized for local inference"
                    .to_string(),
            },
        );

        Self {
            models,
            default_model: "jina".to_string(),
        }
    }
}

impl RerankModelRegistry {
    fn format_available_models(&self) -> String {
        self.models.keys().cloned().collect::<Vec<_>>().join(", ")
    }

    fn resolve_alias_or_name(&self, key: &str) -> Option<(String, &RerankModelConfig)> {
        if let Some(config) = self.models.get(key) {
            return Some((key.to_string(), config));
        }

        self.models
            .iter()
            .find(|(_, config)| config.name == key)
            .map(|(alias, config)| (alias.clone(), config))
    }

    pub fn resolve(&self, requested: Option<&str>) -> Result<(String, RerankModelConfig)> {
        match requested {
            Some(name) => {
                let (alias, config) = self.resolve_alias_or_name(name).ok_or_else(|| {
                    anyhow!(
                        "Unknown rerank model '{}'. Available models: {}",
                        name,
                        self.format_available_models()
                    )
                })?;
                Ok((alias, config.clone()))
            }
            None => {
                let alias = self.default_model.clone();
                let config = self
                    .models
                    .get(&self.default_model)
                    .cloned()
                    .ok_or_else(|| anyhow!("No default reranking model configured"))?;
                Ok((alias, config))
            }
        }
    }

    pub fn aliases(&self) -> Vec<String> {
        let mut keys = self.models.keys().cloned().collect::<Vec<_>>();
        keys.sort();
        keys
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    pub model: String,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub index_backend: String,
}

impl Default for ProjectConfig {
    fn default() -> Self {
        Self {
            model: "bge-small".to_string(),
            chunk_size: 512,
            chunk_overlap: 128,
            index_backend: "hnsw".to_string(),
        }
    }
}

impl ProjectConfig {
    pub fn load(path: &Path) -> Result<Self> {
        if path.exists() {
            let data = std::fs::read_to_string(path)?;
            Ok(serde_json::from_str(&data)?)
        } else {
            Ok(Self::default())
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MULTILINGUAL: [(&str, &str, usize, usize); 3] = [
        (
            "gemma-q4",
            "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF",
            768,
            2048,
        ),
        ("gemma-gguf", "ggml-org/embeddinggemma-300M-GGUF", 768, 2048),
        (
            "granite-gguf",
            "bartowski/granite-embedding-278m-multilingual-GGUF",
            768,
            512,
        ),
    ];

    #[test]
    fn multilingual_aliases_resolve_to_expected_config() {
        let registry = ModelRegistry::default();

        for (alias, name, dimensions, max_tokens) in MULTILINGUAL {
            let (resolved_alias, config) = registry
                .resolve(Some(alias))
                .unwrap_or_else(|e| panic!("alias '{alias}' should resolve: {e}"));

            assert_eq!(resolved_alias, alias);
            assert_eq!(config.name, name, "alias '{alias}' maps to the wrong model");
            assert_eq!(config.dimensions, dimensions, "wrong dims for '{alias}'");
            assert_eq!(
                config.max_tokens, max_tokens,
                "wrong max_tokens for '{alias}'"
            );
            assert_eq!(config.provider, "llamacpp", "wrong provider for '{alias}'");
        }
    }

    #[test]
    fn multilingual_models_resolve_by_full_name_too() {
        let registry = ModelRegistry::default();

        for (_, name, _, _) in MULTILINGUAL {
            let (_, config) = registry
                .resolve(Some(name))
                .unwrap_or_else(|e| panic!("model '{name}' should resolve by name: {e}"));
            assert_eq!(config.name, name);
        }
    }

    /// ckq departs from upstream ck here on purpose: the default is multilingual.
    /// Upstream keeps `bge-small` for backwards compatibility, and PR #199 does
    /// not touch it.
    #[test]
    fn the_default_is_multilingual() {
        let registry = ModelRegistry::default();

        let (alias, config) = registry.resolve(None).expect("default should resolve");
        assert_eq!(alias, "granite-gguf");
        assert_eq!(
            config.name,
            "bartowski/granite-embedding-278m-multilingual-GGUF"
        );
        assert_eq!(config.provider, "llamacpp");
    }

    #[test]
    fn every_registered_model_declares_nonzero_limits() {
        let registry = ModelRegistry::default();

        for alias in registry.aliases() {
            let (_, config) = registry
                .resolve(Some(&alias))
                .expect("a listed alias resolves");
            assert!(config.dimensions > 0, "'{alias}' declares zero dimensions");
            assert!(config.max_tokens > 0, "'{alias}' declares zero max_tokens");
        }
    }

    #[test]
    fn unknown_alias_is_an_error() {
        let registry = ModelRegistry::default();
        assert!(registry.resolve(Some("no-such-model")).is_err());
    }
}
