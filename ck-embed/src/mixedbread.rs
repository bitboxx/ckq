use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, anyhow};
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use ndarray::{Array2, ArrayView, ArrayViewD, Axis, Ix1, Ix2, Ix3};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value;
use tokenizers::{EncodeInput, Tokenizer};

use crate::{
    Embedder, ModelDownloadCallback, model_cache_root,
    reranker::{RerankModelDownloadCallback, RerankResult, Reranker},
};
use ck_models::{ModelConfig, RerankModelConfig};

const EMBED_TOKENIZER_PATH: &str = "tokenizer.json";
const EMBED_MODEL_PATH: &str = "onnx/model_quantized.onnx";

/// Which ONNX file to pull. CoreML handles INT8 QDQ graphs poorly and splits them
/// node by node, so fp16 is worth trying: CKQ_ONNX_FILE=onnx/model_fp16.onnx
fn embed_model_path() -> String {
    std::env::var("CKQ_ONNX_FILE").unwrap_or_else(|_| EMBED_MODEL_PATH.to_string())
}
const RERANK_TOKENIZER_PATH: &str = "tokenizer.json";
const RERANK_MODEL_PATH: &str = "onnx/model_quantized.onnx";

/// Which position of the ONNX output holds the sentence vector.
///
/// BERT-style encoders put it at the first token. Causal embedders such as
/// Qwen3-Embedding put it at the last real token, so the padding has to be
/// skipped using the attention mask.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pooling {
    FirstToken,
    LastToken,
}

pub struct MixedbreadEmbedder {
    session: Session,
    tokenizer: Tokenizer,
    dim: usize,
    max_length: usize,
    model_name: String,
    requires_token_type_ids: bool,
    requires_position_ids: bool,
    /// (layers, kv_heads, head_dim) when the graph is a decoder export that
    /// declares past_key_values inputs; empty tensors satisfy a prefill.
    kv_cache_shape: Option<(usize, usize, usize)>,
    pooling: Pooling,
}

impl MixedbreadEmbedder {
    pub fn new(
        config: &ModelConfig,
        progress_callback: Option<ModelDownloadCallback>,
    ) -> Result<Self> {
        Self::new_with_pooling(config, progress_callback, Pooling::FirstToken)
    }

    pub fn new_with_pooling(
        config: &ModelConfig,
        progress_callback: Option<ModelDownloadCallback>,
        pooling: Pooling,
    ) -> Result<Self> {
        if let Some(cb) = progress_callback.as_ref() {
            cb(&format!(
                "Downloading Mixedbread embedding model ({}) if needed...",
                config.name
            ));
        }

        let (model_path, tokenizer_path) =
            download_assets(&config.name, &embed_model_path(), EMBED_TOKENIZER_PATH)?;

        if let Some(cb) = progress_callback.as_ref() {
            cb("Loading Mixedbread embedder session...");
        }

        // `mut` only on macOS, where the CoreML block below reassigns it.
        // Without the attribute this is a hard error under `-D warnings` on
        // every other platform, which is how CI first caught it.
        #[cfg_attr(not(target_os = "macos"), allow(unused_mut))]
        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_cpus::get().max(1))?;

        // Upstream ck runs every ONNX embedder on the CPU. CoreML hands the graph
        // to the GPU/ANE and is worth a lot on encoder models.
        //
        // It cannot be used for the causal exports. Those declare past_key_values
        // with a dynamic shape, and a prefill passes them with zero elements,
        // which CoreML refuses outright rather than falling back:
        //   "has a dynamic shape ({-1,8,-1,128}) but the runtime shape
        //    ({1,8,0,128}) has zero elements. This is not supported by the
        //    CoreML EP."
        // Pooling is the available proxy: LastToken means a causal export.
        // Off by default: measured slower than plain CPU in every configuration
        // on this hardware. CKQ_COREML=1 to try it on another machine.
        #[cfg(target_os = "macos")]
        if pooling == Pooling::FirstToken && std::env::var("CKQ_COREML").is_ok() {
            use ort::execution_providers::CoreMLExecutionProvider;
            use ort::execution_providers::coreml::{
                ComputeUnits, ModelFormat, SpecializationStrategy,
            };
            // The defaults are close to useless here. ModelFormat::NeuralNetwork is
            // the default and supports fewer operators than MLProgram, so most of
            // the graph falls back to CPU node by node; that is why the untuned EP
            // was worth 4%. MLComputeUnits is unset by default, so the GPU and ANE
            // are never asked for.
            builder = builder.with_execution_providers([CoreMLExecutionProvider::default()
                .with_model_format(ModelFormat::MLProgram)
                .with_compute_units(ComputeUnits::All)
                .with_specialization_strategy(SpecializationStrategy::FastPrediction)
                .with_static_input_shapes(true)
                .build()])?;
        }

        let session = builder.commit_from_file(&model_path)?;

        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!("Tokenizer error: {e}"))?;

        let requires_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        // Causal embedders such as Qwen3-Embedding take explicit position_ids;
        // BERT-style encoders derive them internally and do not declare the input.
        let requires_position_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "position_ids");

        // Some "embedding" ONNX exports are the full causal LM graph and declare a
        // past_key_values input per layer. A prefill only needs them present and
        // empty, so count the layers and read the head shape from the key
        // declaration itself — Qwen3's (8, 128) used to be hardcoded, and any
        // other causal export would have built a wrongly-shaped cache and died
        // inside the runtime with no hint of why.
        let kv_key_input = session
            .inputs()
            .iter()
            .find(|input| {
                input.name().starts_with("past_key_values.") && input.name().ends_with(".key")
            })
            .map(|input| input.name().to_string());
        let kv_cache_shape = match kv_key_input {
            Some(key_input) => {
                let kv_layers = session
                    .inputs()
                    .iter()
                    .filter(|input| {
                        input.name().starts_with("past_key_values.")
                            && input.name().ends_with(".key")
                    })
                    .count();
                let (kv_heads, head_dim) = kv_head_shape(&session, &key_input, &config.name)?;
                Some((kv_layers, kv_heads, head_dim))
            }
            None => None,
        };

        Ok(Self {
            session,
            tokenizer,
            dim: config.dimensions,
            max_length: config.max_tokens,
            model_name: config.name.clone(),
            requires_token_type_ids,
            requires_position_ids,
            kv_cache_shape,
            pooling,
        })
    }

    #[allow(clippy::type_complexity)]
    fn build_inputs(
        &self,
        texts: &[String],
    ) -> Result<(Array2<i64>, Array2<i64>, Option<Array2<i64>>)> {
        let mut encodings = Vec::with_capacity(texts.len());
        let mut truncated = 0usize;
        for text in texts {
            let encoding = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| anyhow!("Tokenizer encode failed: {e}"))?;
            if encoding.len() > self.max_length {
                truncated += 1;
            }
            encodings.push(encoding);
        }
        if truncated > 0 {
            warn_truncation(truncated, texts.len(), self.max_length, self.pooling);
        }

        let seq_len = encodings
            .iter()
            .map(tokenizers::Encoding::len)
            .max()
            .unwrap_or(1)
            .min(self.max_length)
            .max(1);

        let batch = encodings.len();
        let mut input_ids = vec![0i64; batch * seq_len];
        let mut attention_mask = vec![0i64; batch * seq_len];
        let mut token_types = if self.requires_token_type_ids {
            Some(vec![0i64; batch * seq_len])
        } else {
            None
        };

        for (row, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let type_ids = encoding.get_type_ids();
            let len = ids.len().min(seq_len);

            let row_offset = row * seq_len;
            for idx in 0..len {
                input_ids[row_offset + idx] = ids[idx] as i64;
                attention_mask[row_offset + idx] = mask[idx] as i64;
            }

            if let Some(ref mut token_types_buf) = token_types
                && !type_ids.is_empty()
            {
                for idx in 0..len {
                    token_types_buf[row_offset + idx] = type_ids[idx] as i64;
                }
            }
        }

        let token_type_array =
            token_types.map(|buf| Array2::from_shape_vec((batch, seq_len), buf).unwrap());

        Ok((
            Array2::from_shape_vec((batch, seq_len), input_ids)
                .expect("validated dimensions for input ids"),
            Array2::from_shape_vec((batch, seq_len), attention_mask)
                .expect("validated dimensions for attention mask"),
            token_type_array,
        ))
    }

    fn normalize(
        rows: ArrayViewD<'_, f32>,
        model: &str,
        dim: usize,
        pooling: Pooling,
        lengths: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        let ndim = rows.ndim();
        match ndim {
            // Already pooled by the graph: one vector per input, nothing to select.
            2 => {
                let view = rows.into_dimensionality::<Ix2>()?;
                view.rows()
                    .into_iter()
                    .map(|row| normalize_row(row, model, dim))
                    .collect()
            }
            // (batch, sequence, hidden): pick the position that carries the vector.
            3 => {
                let view = rows.into_dimensionality::<Ix3>()?;
                let seq_len = view.shape()[1];
                view.outer_iter()
                    .enumerate()
                    .map(|(row, matrix)| {
                        let index = match pooling {
                            Pooling::FirstToken => 0,
                            // Right-padded, so the last real token is len - 1.
                            // Clamp: a zero length would underflow, and a length
                            // past the tensor would panic on index_axis.
                            Pooling::LastToken => lengths
                                .get(row)
                                .copied()
                                .unwrap_or(seq_len)
                                .saturating_sub(1)
                                .min(seq_len.saturating_sub(1)),
                        };
                        normalize_row(matrix.index_axis(Axis(0), index), model, dim)
                    })
                    .collect()
            }
            other => Err(anyhow!("Unexpected embedding tensor rank: {other}")),
        }
    }
}
/// (kv_heads, head_dim) from the declared rank-4 shape
/// (batch, kv_heads, past_len, head_dim) of a `past_key_values.*.key` input.
/// Batch and past_len are dynamic in these exports and do not matter for an
/// empty prefill cache; a dynamic head dimension means the export does not pin
/// its attention geometry, and erroring clearly beats guessing a shape that
/// belongs to a different model.
fn kv_head_shape(session: &Session, key_input: &str, model: &str) -> Result<(usize, usize)> {
    use ort::value::ValueType;

    let dims = match session
        .inputs()
        .iter()
        .find(|input| input.name() == key_input)
        .map(|input| input.dtype())
    {
        Some(ValueType::Tensor { shape, .. }) => shape,
        Some(other) => {
            return Err(anyhow!(
                "{model}: KV input '{key_input}' is not a tensor ({other:?})"
            ));
        }
        None => unreachable!("the name was taken from this same session"),
    };
    let fixed = |idx: usize, what: &str| -> Result<usize> {
        dims.get(idx)
            .copied()
            .filter(|dim| *dim > 0)
            .map(|dim| dim as usize)
            .ok_or_else(|| {
                anyhow!(
                    "{model}: KV input '{key_input}' declares a dynamic {what} dimension \
                     ({dims:?}); cannot build its empty prefill cache"
                )
            })
    };
    Ok((fixed(1, "key-value head")?, fixed(3, "head")?))
}

impl Embedder for MixedbreadEmbedder {
    fn id(&self) -> &'static str {
        "mixedbread"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let (input_ids, attention_mask, token_types) = self.build_inputs(texts)?;

        // Real token count per row, so last-token pooling skips the padding.
        let lengths: Vec<usize> = attention_mask
            .rows()
            .into_iter()
            .map(|row| row.iter().filter(|&&m| m != 0).count())
            .collect();

        let outputs = if self.requires_token_type_ids {
            let token_types = token_types.expect("token type ids required but missing");
            self.session.run(ort::inputs![
                Value::from_array(input_ids)?,
                Value::from_array(attention_mask)?,
                Value::from_array(token_types)?
            ])?
        } else if let Some((layers, kv_heads, head_dim)) = self.kv_cache_shape {
            // Decoder export: feed position_ids plus an empty KV cache per layer.
            // past_len is 0, so every cache tensor has zero elements and the
            // prefill computes the whole sequence in one pass.
            let (batch, seq_len) = input_ids.dim();
            let positions: Vec<i64> = (0..batch)
                .flat_map(|_| (0..seq_len).map(|i| i as i64))
                .collect();
            let position_ids = Array2::from_shape_vec((batch, seq_len), positions)
                .expect("validated dimensions for position ids");

            let mut inputs = ort::inputs![
                "input_ids" => Value::from_array(input_ids)?,
                "attention_mask" => Value::from_array(attention_mask)?,
                "position_ids" => Value::from_array(position_ids)?,
            ];
            for layer in 0..layers {
                let empty = ndarray::Array4::<f32>::zeros((batch, kv_heads, 0, head_dim));
                inputs.push((
                    format!("past_key_values.{layer}.key").into(),
                    Value::from_array(empty.clone())?.into(),
                ));
                inputs.push((
                    format!("past_key_values.{layer}.value").into(),
                    Value::from_array(empty)?.into(),
                ));
            }
            self.session.run(inputs)?
        } else if self.requires_position_ids {
            // Right-padded, so positions simply count up from zero on every row.
            let (batch, seq_len) = input_ids.dim();
            let positions: Vec<i64> = (0..batch)
                .flat_map(|_| (0..seq_len).map(|i| i as i64))
                .collect();
            let position_ids = Array2::from_shape_vec((batch, seq_len), positions)
                .expect("validated dimensions for position ids");
            self.session.run(ort::inputs![
                Value::from_array(input_ids)?,
                Value::from_array(attention_mask)?,
                Value::from_array(position_ids)?
            ])?
        } else {
            self.session.run(ort::inputs![
                Value::from_array(input_ids)?,
                Value::from_array(attention_mask)?
            ])?
        };

        let embedding_tensor = outputs[0]
            .try_extract_array::<f32>()
            .context("Failed to extract embedding tensor")?;

        Self::normalize(
            embedding_tensor,
            self.model_name.as_str(),
            self.dim,
            self.pooling,
            &lengths,
        )
    }
}

pub struct MixedbreadReranker {
    session: Session,
    tokenizer: Tokenizer,
    max_length: usize,
    requires_token_type_ids: bool,
}

impl MixedbreadReranker {
    pub fn new(
        config: &RerankModelConfig,
        progress_callback: Option<RerankModelDownloadCallback>,
    ) -> Result<Self> {
        if let Some(cb) = progress_callback.as_ref() {
            cb(&format!(
                "Downloading Mixedbread reranker model ({}) if needed...",
                config.name
            ));
        }

        let (model_path, tokenizer_path) =
            download_assets(&config.name, RERANK_MODEL_PATH, RERANK_TOKENIZER_PATH)?;

        if let Some(cb) = progress_callback.as_ref() {
            cb("Loading Mixedbread reranker session...");
        }

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_cpus::get().max(1))?
            .commit_from_file(&model_path)?;

        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!("Tokenizer error: {e}"))?;

        let requires_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        Ok(Self {
            session,
            tokenizer,
            max_length: 512,
            requires_token_type_ids,
        })
    }

    #[allow(clippy::type_complexity)]
    fn build_inputs(
        &self,
        query: &str,
        documents: &[String],
    ) -> Result<(Array2<i64>, Array2<i64>, Option<Array2<i64>>)> {
        let mut encodings = Vec::with_capacity(documents.len());
        for doc in documents {
            let encoding = self
                .tokenizer
                .encode(EncodeInput::Dual(query.into(), doc.as_str().into()), true)
                .map_err(|e| anyhow!("Tokenizer encode failed: {e}"))?;
            encodings.push(encoding);
        }

        let seq_len = encodings
            .iter()
            .map(tokenizers::Encoding::len)
            .max()
            .unwrap_or(1)
            .min(self.max_length)
            .max(1);

        let batch = encodings.len();
        let mut input_ids = vec![0i64; batch * seq_len];
        let mut attention_mask = vec![0i64; batch * seq_len];
        let mut token_types = if self.requires_token_type_ids {
            Some(vec![0i64; batch * seq_len])
        } else {
            None
        };

        for (row, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let type_ids = encoding.get_type_ids();
            let len = ids.len().min(seq_len);
            let offset = row * seq_len;

            for idx in 0..len {
                input_ids[offset + idx] = ids[idx] as i64;
                attention_mask[offset + idx] = mask[idx] as i64;
            }

            if let Some(ref mut token_types_buf) = token_types
                && !type_ids.is_empty()
            {
                for idx in 0..len {
                    token_types_buf[offset + idx] = type_ids[idx] as i64;
                }
            }
        }

        let token_type_array =
            token_types.map(|buf| Array2::from_shape_vec((batch, seq_len), buf).unwrap());

        Ok((
            Array2::from_shape_vec((batch, seq_len), input_ids)
                .expect("validated dimensions for input ids"),
            Array2::from_shape_vec((batch, seq_len), attention_mask)
                .expect("validated dimensions for attention mask"),
            token_type_array,
        ))
    }
}

impl Reranker for MixedbreadReranker {
    fn id(&self) -> &'static str {
        "mixedbread_reranker"
    }

    fn rerank(&mut self, query: &str, documents: &[String]) -> Result<Vec<RerankResult>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let (input_ids, attention_mask, token_types) = self.build_inputs(query, documents)?;

        let outputs = if self.requires_token_type_ids {
            let token_types = token_types.expect("token type ids required but missing");
            self.session.run(ort::inputs![
                Value::from_array(input_ids)?,
                Value::from_array(attention_mask)?,
                Value::from_array(token_types)?
            ])?
        } else {
            self.session.run(ort::inputs![
                Value::from_array(input_ids)?,
                Value::from_array(attention_mask)?
            ])?
        };

        let logits = outputs[0]
            .try_extract_array::<f32>()
            .context("Failed to extract reranker logits")?
            .into_dimensionality::<Ix2>()?;

        let mut results = Vec::with_capacity(documents.len());
        for (i, row) in logits.rows().into_iter().enumerate() {
            let logit = row
                .get(0)
                .copied()
                .unwrap_or_else(|| row.iter().copied().next().unwrap_or(0.0));
            let score = 1.0 / (1.0 + (-logit).exp());
            results.push(RerankResult {
                query: query.to_string(),
                document: documents[i].clone(),
                score,
            });
        }

        Ok(results)
    }
}

fn normalize_row(row: ArrayView<'_, f32, Ix1>, model: &str, dim: usize) -> Result<Vec<f32>> {
    // A registry/model dimension mismatch must fail loudly like the llama.cpp
    // path does: zero-padding used to turn it into a plausible half-empty
    // vector that silently poisoned every similarity score that touched it.
    if row.len() != dim {
        return Err(anyhow!(
            "{model} produced {}-d vectors but the registry declares {dim}",
            row.len()
        ));
    }
    let mut values = row.to_vec();
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = norm.recip();
        for value in &mut values {
            *value *= inv;
        }
    }
    Ok(values)
}

/// Truncation must not be silent: for a last-token pooling model the dropped
/// tail is exactly the token that carries the sentence vector, so the result
/// is wrong rather than merely lossy. Once per run is enough — an index job
/// truncates in the thousands and one visible line says it all.
static TRUNCATION_WARNED: AtomicBool = AtomicBool::new(false);

fn warn_truncation(count: usize, total: usize, max_length: usize, pooling: Pooling) {
    if TRUNCATION_WARNED.swap(true, Ordering::Relaxed) {
        return;
    }
    let gravity = if pooling == Pooling::LastToken {
        "this model pools its LAST token, so truncation removes the token that carries \
         the sentence vector: the embedding is wrong, not just lossy"
    } else {
        "the truncated tokens are absent from the pooled vector"
    };
    eprintln!(
        "ckq: warning: truncated {count} of {total} inputs to {max_length} tokens before \
         embedding ({gravity}; further truncations are not logged)"
    );
}

pub(crate) fn download_assets(
    model_id: &str,
    model_path: &str,
    tokenizer_path: &str,
) -> Result<(PathBuf, PathBuf)> {
    let cache_dir = model_cache_root()?;
    std::fs::create_dir_all(&cache_dir)?;

    let api = ApiBuilder::new()
        .with_cache_dir(cache_dir)
        .build()
        .context("Failed to initialize Hugging Face Hub client")?;

    let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, "main".to_string());
    let tokenizer = api
        .repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            "main".to_string(),
        ))
        .get(tokenizer_path)
        .with_context(|| format!("Failed to download tokenizer for {model_id}"))?;
    let model = api
        .repo(repo)
        .get(model_path)
        .with_context(|| format!("Failed to download ONNX model for {model_id}"))?;

    Ok((model, tokenizer))
}
