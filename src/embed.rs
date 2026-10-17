//! Text to vectors, with llama.cpp.
//!
//! The model's prompts are applied here and nowhere else: `embed_query` adds
//! the query prefix, `embed_documents` the document prefix. The old engine
//! applied them in its client and again in its daemon, so every query reached
//! the model with both.
//!
//! No daemon. A search loads the model in its own process: the GGUF is
//! memory-mapped, so a second process shares the pages instead of a copy.
//! Queries run on the device that is fastest for one short text from a cold
//! start, which may well be the CPU (no Metal shader compilation); indexing
//! runs on the GPU where there is one. Port the context and batching logic
//! from `ck-embed/src/llamacpp.rs`: one context sized for indexing batches, a
//! small one for a single query, mean or last-token pooling, L2 normalisation.
//! llama.cpp's own log lines go to stderr or nowhere, never to stdout.

use crate::models::ModelSpec;
use anyhow::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// Every layer on the GPU (Metal on macOS, CPU elsewhere unless built with a GPU backend).
    Gpu,
    Cpu,
}

/// For a search: `CKQ_QUERY_DEVICE=cpu|gpu` when set, else whichever measured
/// faster for a cold single query on this platform.
pub fn query_device() -> Device {
    todo!("decide by measurement; record the numbers in the commit message")
}

pub trait Embedder {
    fn dims(&self) -> usize;
    /// Unit vectors, one per text, the document prefix applied.
    fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    /// A unit vector for one query, the query prefix applied.
    fn embed_query(&mut self, text: &str) -> Result<Vec<f32>>;
}

pub struct LlamaEmbedder {
    spec: &'static ModelSpec,
    // the llama.cpp backend, model and contexts
}

impl LlamaEmbedder {
    pub fn load(spec: &'static ModelSpec, device: Device) -> Result<Self> {
        todo!()
    }
}

impl Embedder for LlamaEmbedder {
    fn dims(&self) -> usize {
        self.spec.dims
    }

    fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        todo!()
    }

    fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        todo!()
    }
}
