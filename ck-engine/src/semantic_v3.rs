use anyhow::Result;
use ck_core::{CkError, SearchOptions, SearchResult};
use std::path::Path;
use walkdir::WalkDir;

use super::{
    SearchProgressCallback, extract_content_from_span, find_nearest_index_root,
    resolve_model_from_root,
};

/// New semantic search implementation using span-based storage
pub async fn semantic_search_v3(options: &SearchOptions) -> Result<ck_core::SearchResults> {
    semantic_search_v3_with_progress(options, None).await
}

pub async fn semantic_search_v3_with_progress(
    options: &SearchOptions,
    progress_callback: Option<SearchProgressCallback>,
) -> Result<ck_core::SearchResults> {
    // Find the index root
    let index_root = find_nearest_index_root(&options.path).unwrap_or_else(|| {
        if options.path.is_file() {
            options.path.parent().unwrap_or(&options.path).to_path_buf()
        } else {
            options.path.clone()
        }
    });

    let index_dir = ck_core::index_dir(&index_root);
    if !index_dir.exists() {
        return Err(CkError::Index(
            "Index creation failed. Please try running 'ckq --index' explicitly.".to_string(),
        )
        .into());
    }
    // Refuse to serve results from an index dir that a different root claimed
    // via a CK_INDEX_DIR basename-hash collision. No-op in-tree.
    ck_core::check_index_root_marker(&index_root)?;

    if let Some(ref callback) = progress_callback {
        callback("Loading embeddings...");
    }

    // Build the path scope filter once, up front. Previously this was
    // applied AFTER top_k inside the iteration loop, so a whole-codebase
    // index plus a narrow `path=` query could return zero matches when
    // the global top_k results all lived outside the requested scope.
    let scope = PathScope::new(&options.path, &index_root);
    let timing = std::env::var_os("CKQ_TIMING").is_some();
    let t0 = std::time::Instant::now();

    let (vectors, cache_hit) = load_all_chunks(&index_dir, &index_root)?;
    // The include and scope checks run once per file, not once per chunk. An
    // include pattern that covers the whole index matches every file, so skip
    // the check: it canonicalizes each path, a system call per indexed file.
    let root = index_root
        .canonicalize()
        .unwrap_or_else(|_| index_root.clone());
    let includes_everything = options
        .include_patterns
        .iter()
        .any(|pattern| pattern.is_dir && root.starts_with(&pattern.path));
    let keep: Vec<bool> = vectors
        .files
        .iter()
        .map(|file| {
            (includes_everything || super::path_matches_include(file, &options.include_patterns))
                && scope.contains(file)
        })
        .collect();
    let file_chunks: Vec<usize> = (0..vectors.len())
        .filter(|&i| keep[vectors.chunks[i].0 as usize])
        .collect();

    if timing {
        eprintln!(
            "timing: vector load {:?} ({} chunks, {})",
            t0.elapsed(),
            file_chunks.len(),
            if cache_hit { "cache" } else { "sidecars" }
        );
    }
    let t1 = std::time::Instant::now();
    if file_chunks.is_empty() {
        return Err(CkError::Index(
            "No embeddings found. Run 'ckq --index' first with embeddings.".to_string(),
        )
        .into());
    }

    if let Some(ref callback) = progress_callback {
        callback(&format!(
            "Found {} chunks with embeddings",
            file_chunks.len()
        ));
    }

    // Create embedder and embed the query
    if let Some(ref callback) = progress_callback {
        callback("Loading embedding model...");
    }

    let resolved_model = resolve_model_from_root(&index_root, options.embedding_model.as_deref())?;
    if let Some(ref callback) = progress_callback {
        if resolved_model.alias == resolved_model.canonical_name() {
            callback(&format!(
                "Using embedding model {} ({} dims)",
                resolved_model.canonical_name(),
                resolved_model.dimensions()
            ));
        } else {
            callback(&format!(
                "Using embedding model {} (alias '{}', {} dims)",
                resolved_model.canonical_name(),
                resolved_model.alias,
                resolved_model.dimensions()
            ));
        }
    }

    let mut embedder = ck_embed::create_embedder_for_config(
        &resolved_model.config,
        &ck_embed::EmbedderOptions::process(),
        None,
    )?;
    // Role::Query, not the default: an asymmetric model puts a different
    // instruction on the question than on the answer, and only then do the two
    // land near each other.
    let query_embeddings =
        embedder.embed_with(std::slice::from_ref(&options.query), ck_embed::Role::Query)?;

    if query_embeddings.is_empty() {
        return Ok(ck_core::SearchResults {
            matches: Vec::new(),
            closest_below_threshold: None,
        });
    }

    let query_embedding = &query_embeddings[0];
    if timing {
        eprintln!("timing: embed query {:?}", t1.elapsed());
    }

    if let Some(ref callback) = progress_callback {
        callback("Computing similarity scores...");
    }

    // Compute similarities
    let mut similarities: Vec<(f32, usize)> = file_chunks
        .iter()
        .map(|&i| (cosine_similarity(query_embedding, vectors.embedding(i)), i))
        .collect();

    // Sort by similarity (highest first)
    similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Apply threshold and top_k filtering
    let mut results = Vec::new();
    let mut closest_below_threshold: Option<SearchResult> = None;
    let limit = options.top_k.unwrap_or(similarities.len());

    for (similarity, i) in similarities.into_iter().take(limit) {
        let (file, span) = &vectors.chunks[i];
        let file_path = &vectors.files[*file as usize];
        let is_below_threshold = options
            .threshold
            .is_some_and(|threshold| similarity < threshold);

        // Extract content from the file using the span, skip if file doesn't exist
        let content = if options.full_section {
            match extract_content_from_span(file_path, span).await {
                Ok(content) => content,
                Err(_) => {
                    // Skip files that no longer exist (stale index entries)
                    continue;
                }
            }
        } else {
            match extract_content_from_span(file_path, span).await {
                Ok(full_content) => {
                    // Take first 3 lines for preview
                    full_content.lines().take(3).collect::<Vec<_>>().join("\n")
                }
                Err(_) => {
                    // Skip files that no longer exist (stale index entries)
                    continue;
                }
            }
        };

        let search_result = SearchResult {
            signals: None,
            file: file_path.clone(),
            span: span.clone(),
            score: similarity,
            preview: content,
            lang: ck_core::Language::from_path(file_path),
            symbol: None,
            chunk_hash: None,
            index_epoch: None,
        };

        if is_below_threshold {
            // Track the closest below-threshold result (first one since sorted by highest first)
            if closest_below_threshold.is_none() {
                closest_below_threshold = Some(search_result);
            }
        } else {
            // Add to main results if above threshold
            results.push(search_result);
        }
    }

    // Apply reranking if enabled
    if options.rerank && !results.is_empty() {
        if let Some(ref callback) = progress_callback {
            callback("Reranking results for improved relevance...");
        }

        let rerank_registry = ck_models::RerankModelRegistry::default();
        let (rerank_alias, rerank_config) = rerank_registry
            .resolve(options.rerank_model.as_deref())
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        match ck_embed::create_reranker_for_config(&rerank_config, None) {
            Ok(mut reranker) => {
                if let Some(ref callback) = progress_callback {
                    callback(&format!("Reranking results with model {rerank_alias}"));
                }

                let documents: Vec<String> = results.iter().map(|r| r.preview.clone()).collect();

                match reranker.rerank(&options.query, &documents) {
                    Ok(rerank_results) => {
                        // Create a map from document text to indices for handling duplicates
                        let mut doc_to_indices: std::collections::HashMap<String, Vec<usize>> =
                            std::collections::HashMap::new();
                        for (i, result) in results.iter().enumerate() {
                            doc_to_indices
                                .entry(result.preview.clone())
                                .or_default()
                                .push(i);
                        }

                        // Update results with reranked scores
                        // The reranker returns results in reranked order, so we match by document text
                        for rerank_result in rerank_results.iter() {
                            if let Some(indices) = doc_to_indices.get_mut(&rerank_result.document)
                                && let Some(idx) = indices.pop()
                            {
                                results[idx].score = rerank_result.score;
                            }
                        }

                        // Re-sort by reranked scores
                        results.sort_by(|a, b| {
                            b.score
                                .partial_cmp(&a.score)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });

                        // Apply top_k limit again after reranking
                        if let Some(limit) = options.top_k {
                            results.truncate(limit);
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Reranking failed, using original scores: {}", e);
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Failed to create reranker, using original scores: {}", e);
            }
        }
    }

    Ok(ck_core::SearchResults {
        matches: results,
        closest_below_threshold,
    })
}

/// Every chunk with an embedding in the index, from the packed cache when it is
/// current, otherwise from the sidecars, which then refresh the cache. The bool
/// says whether the cache was used.
fn load_all_chunks(
    index_dir: &Path,
    index_root: &Path,
) -> Result<(ck_index::vector_cache::VectorSet, bool)> {
    use ck_index::vector_cache;
    if let Some(chunks) = vector_cache::load(index_dir) {
        return Ok((chunks, true));
    }
    // Taken before the walk: if the index changes while the sidecars are being
    // read, the save below sees a different manifest and writes nothing.
    let stamp = vector_cache::manifest_stamp(index_dir);
    let mut chunks = vector_cache::VectorSet::default();
    for entry in WalkDir::new(index_dir) {
        let entry = entry?;
        let path = entry.path();
        if !entry.file_type().is_file() || path.extension().and_then(|s| s.to_str()) != Some("ck") {
            continue;
        }
        let Ok(index_entry) = ck_index::load_index_entry(path) else {
            continue;
        };
        let Some(original_file) = reconstruct_original_path(path, index_dir, index_root) else {
            continue;
        };
        for chunk in index_entry.chunks {
            if let Some(embedding) = chunk.embedding {
                chunks.push(&original_file, chunk.span, &embedding);
            }
        }
    }
    if let Some(stamp) = stamp
        && let Err(e) = vector_cache::save(index_dir, stamp, &chunks)
    {
        tracing::debug!("could not write the vector cache: {e}");
    }
    Ok((chunks, false))
}

/// Rebuild the packed vector cache for the index at or above `path`, if it is
/// stale. Run after indexing, so the next search finds it current.
pub fn refresh_vector_cache(path: &Path) -> Result<()> {
    let Some(index_root) = find_nearest_index_root(path) else {
        return Ok(());
    };
    let index_dir = ck_core::index_dir(&index_root);
    load_all_chunks(&index_dir, &index_root).map(|_| ())
}

/// Scope a semantic query to a file, a directory, or the whole index.
///
/// Cached canonical form of `options.path` so per-chunk membership
/// checks don't re-canonicalize on every iteration.
enum PathScope {
    All,
    File(std::path::PathBuf),
    Dir(std::path::PathBuf),
}

impl PathScope {
    fn new(path: &Path, index_root: &Path) -> Self {
        if path == Path::new(".") {
            return Self::All;
        }
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        // Searching the whole index is the common case; it needs no per-file
        // canonicalize, which is a system call per indexed file.
        if index_root.canonicalize().ok().as_deref() == Some(canonical.as_path()) {
            return Self::All;
        }
        if path.is_file() {
            Self::File(canonical)
        } else {
            Self::Dir(canonical)
        }
    }

    fn contains(&self, file: &Path) -> bool {
        match self {
            Self::All => true,
            Self::File(target) => {
                let canonical = file.canonicalize().unwrap_or_else(|_| file.to_path_buf());
                canonical == *target
            }
            Self::Dir(target) => {
                let canonical = file.canonicalize().unwrap_or_else(|_| file.to_path_buf());
                canonical.starts_with(target)
            }
        }
    }
}

fn reconstruct_original_path(
    sidecar_path: &Path,
    index_dir: &Path,
    repo_root: &Path,
) -> Option<std::path::PathBuf> {
    // Remove the index directory prefix and .ck extension
    let relative_path = sidecar_path.strip_prefix(index_dir).ok()?;
    let mut original_path = relative_path.with_extension("");

    // Handle the .ck extension removal
    if let Some(name) = original_path.file_name() {
        let name_str = name.to_string_lossy();
        if let Some(original_name) = name_str.strip_suffix(".ck") {
            let mut new_path = original_path.clone();
            new_path.set_file_name(original_name);
            original_path = new_path;
        }
    }

    Some(repo_root.join(original_path))
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod path_scope_tests {
    use super::PathScope;
    use std::fs;
    use std::path::Path;
    use tempfile::TempDir;

    #[test]
    fn all_matches_anything() {
        let scope = PathScope::new(Path::new("."), Path::new("/nonexistent-root"));
        assert!(scope.contains(Path::new("/tmp/whatever")));
        assert!(scope.contains(Path::new("./relative")));
    }

    #[test]
    fn dir_matches_descendants_only() {
        let tmp = TempDir::new().unwrap();
        let scoped = tmp.path().join("inside");
        let outside = tmp.path().join("outside");
        fs::create_dir(&scoped).unwrap();
        fs::create_dir(&outside).unwrap();
        let inside_file = scoped.join("a.txt");
        let outside_file = outside.join("b.txt");
        fs::write(&inside_file, "x").unwrap();
        fs::write(&outside_file, "y").unwrap();

        let scope = PathScope::new(&scoped, tmp.path());
        assert!(scope.contains(&inside_file));
        assert!(!scope.contains(&outside_file));
    }

    #[test]
    fn the_index_root_itself_is_all() {
        let tmp = TempDir::new().unwrap();
        let scope = PathScope::new(tmp.path(), tmp.path());
        assert!(matches!(scope, PathScope::All));
    }

    #[test]
    fn file_matches_exactly_that_file() {
        let tmp = TempDir::new().unwrap();
        let target = tmp.path().join("target.txt");
        let other = tmp.path().join("other.txt");
        fs::write(&target, "x").unwrap();
        fs::write(&other, "y").unwrap();

        let scope = PathScope::new(&target, &tmp.path().join("elsewhere"));
        assert!(scope.contains(&target));
        assert!(!scope.contains(&other));
    }
}
