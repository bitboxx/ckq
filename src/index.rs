//! Building and updating the index. The only code that writes the store.
//!
//! A file is re-read only when its size or mtime changed, and re-embedded
//! only when its blake3 hash changed. Deleted files are removed in the same
//! run. The model comes from `--model`, else from the existing index, else
//! the default; a different model than the index holds means a full rebuild.

use crate::corpus::{Corpus, RelPath};
use crate::embed::{Device, Embedder, LlamaEmbedder};
use crate::models::{self, ModelSpec};
use crate::store::{FileRecord, IndexMeta, Store};
use crate::walk::WalkOptions;
use anyhow::{Context, Result, anyhow};
use std::time::Instant;

#[derive(Debug, Clone, Default)]
pub struct IndexOptions {
    /// Model alias from `--model`; `None` keeps the index's own.
    pub model: Option<String>,
    pub walk: WalkOptions,
    /// No progress output on stderr.
    pub quiet: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndexReport {
    pub added: usize,
    pub updated: usize,
    pub removed: usize,
    pub unchanged: usize,
    /// Files skipped because they could not be read (permissions) or are not
    /// UTF-8 text. They do not fail the run.
    pub skipped: usize,
    pub chunks_embedded: usize,
}

impl IndexReport {
    /// The one-line summary `--index` prints on stderr.
    pub fn summary(&self) -> String {
        let skipped = if self.skipped > 0 {
            format!(", {} skipped", self.skipped)
        } else {
            String::new()
        };
        format!(
            "indexed: {} added, {} updated, {} removed, {} unchanged, {} chunks embedded{skipped}",
            self.added, self.updated, self.removed, self.unchanged, self.chunks_embedded
        )
    }
}

/// Which model this run indexes with, and the meta it implies. `None` keeps
/// the index's own when it has one.
fn resolve_model(corpus: &Corpus, opts: &IndexOptions) -> Result<(&'static ModelSpec, IndexMeta)> {
    if let Some(name) = &opts.model {
        let spec = models::resolve(name)?;
        return Ok((spec, IndexMeta::for_model(spec)));
    }
    if let Some(store) = Store::open_read(corpus)? {
        let meta = store.meta()?;
        let spec = models::resolve(&meta.model_repo)?;
        return Ok((spec, IndexMeta::for_model(spec)));
    }
    let spec = models::resolve(models::DEFAULT_ALIAS)?;
    Ok((spec, IndexMeta::for_model(spec)))
}

/// Bring the index in line with the files on disk.
///
/// With `CKQ_TIMING=1`, print one line to stderr with the time spent walking,
/// reading, chunking, embedding and writing.
pub fn update(corpus: &Corpus, opts: &IndexOptions) -> Result<IndexReport> {
    let timing = std::env::var_os("CKQ_TIMING").is_some();
    let t0 = Instant::now();
    crate::walk::ensure_ckignore(corpus)?;
    let walked = crate::walk::walk(corpus, &opts.walk)?;
    let t_walk = t0.elapsed();

    let (spec, meta) = resolve_model(corpus, opts)?;
    // A different model than the index holds means a full rebuild: its
    // vectors live in a different space and must not be mixed.
    if let Some(store) = Store::open_read(corpus)? {
        let existing = store.meta()?;
        if existing.signature != meta.signature {
            if !opts.quiet {
                eprintln!(
                    "index model changed from {} to {}; rebuilding",
                    existing.model_alias, meta.model_alias
                );
            }
            clean(corpus)?;
        }
    }

    let mut store = Store::open_write(corpus, &meta).context("open_write")?;
    let indexed = store.files().context("files")?;
    let mut report = IndexReport::default();

    let mut to_embed: Vec<(FileRecord, String, Vec<crate::chunk::Chunk>)> = Vec::new();
    let t_read = Instant::now();
    for file in &walked {
        match std::fs::read(corpus.abs(&file.rel)) {
            Ok(bytes) => {
                let hash = blake3::hash(&bytes).to_hex().to_string();
                let rec = FileRecord {
                    rel: file.rel.clone(),
                    hash: hash.clone(),
                    size: file.size,
                    mtime_ns: file.mtime_ns,
                };
                let prior = indexed.get(&file.rel);
                if prior.is_some_and(|p| p.hash == hash) {
                    if prior.is_some_and(|p| p.size == file.size && p.mtime_ns == file.mtime_ns) {
                        report.unchanged += 1;
                    } else {
                        // Same bytes, new stat: remember the stat, keep the vectors.
                        store.note_stat(&rec).context("note_stat")?;
                        report.unchanged += 1;
                    }
                    continue;
                }
                let text = match String::from_utf8(bytes) {
                    Ok(t) => t,
                    Err(_) => {
                        report.skipped += 1;
                        continue;
                    }
                };
                let chunks = crate::chunk::chunk(&file.rel, &text);
                if prior.is_some() {
                    report.updated += 1;
                } else {
                    report.added += 1;
                }
                to_embed.push((rec, text, chunks));
            }
            Err(_) => {
                report.skipped += 1;
            }
        }
    }
    let t_readwrite = t_read.elapsed();

    let t_embed = Instant::now();
    let mut embedded = 0usize;
    if !to_embed.is_empty() {
        if !opts.quiet {
            eprintln!(
                "embedding {} files with {}...",
                to_embed.len(),
                meta.model_alias
            );
        }
        let mut embedder = LlamaEmbedder::load(spec, Device::Gpu)?;
        for (rec, text, chunks) in &to_embed {
            let texts: Vec<&str> = chunks
                .iter()
                .map(|c| &text[c.byte_start..c.byte_end])
                .collect();
            let vectors = embedder.embed_documents(&texts)?;
            embedded += vectors.len();
            store
                .put_file(rec, text, chunks, &vectors)
                .context("put_file")?;
        }
    }
    let t_embedding = t_embed.elapsed();

    // What the index holds that the walk no longer sees is gone.
    let live: std::collections::HashSet<&RelPath> = walked.iter().map(|f| &f.rel).collect();
    for rel in indexed.keys() {
        if !live.contains(rel) {
            store.remove_file(rel).context("remove_file")?;
            report.removed += 1;
        }
    }

    if timing {
        eprintln!("timing: walk {t_walk:?}, read+chunk {t_readwrite:?}, embed {t_embedding:?}");
    }
    report.chunks_embedded = embedded;
    Ok(report)
}

/// Index or re-index one file (`--add`), leaving the rest as it is.
pub fn add_file(corpus: &Corpus, rel: &RelPath, opts: &IndexOptions) -> Result<IndexReport> {
    let (spec, meta) = resolve_model(corpus, opts)?;
    if let Some(store) = Store::open_read(corpus)? {
        let existing = store.meta()?;
        if existing.signature != meta.signature {
            return Err(anyhow!(
                "the index at {} was built with a different embedding signature; \
                 run `ckq --clean` first",
                corpus.index_dir().display()
            ));
        }
    }
    let mut store = Store::open_write(corpus, &meta).context("open_write")?;
    let path = corpus.abs(rel);
    let bytes = std::fs::read(&path).map_err(|e| anyhow!("reading {}: {e}", path.display()))?;
    let text =
        String::from_utf8(bytes).map_err(|_| anyhow!("{} is not UTF-8 text", path.display()))?;
    let hash = blake3::hash(text.as_bytes()).to_hex().to_string();
    let file_meta = std::fs::metadata(&path)?;
    let rec = FileRecord {
        rel: rel.clone(),
        hash,
        size: file_meta.len(),
        mtime_ns: mtime_ns(&file_meta),
    };
    let chunks = crate::chunk::chunk(rel, &text);
    let mut report = IndexReport::default();
    let prior = store.files()?.contains_key(rel);
    let vectors = if chunks.is_empty() {
        Vec::new()
    } else {
        let mut embedder = LlamaEmbedder::load(spec, Device::Gpu)?;
        let texts: Vec<&str> = chunks
            .iter()
            .map(|c| &text[c.byte_start..c.byte_end])
            .collect();
        let vectors = embedder.embed_documents(&texts)?;
        report.chunks_embedded = vectors.len();
        vectors
    };
    store.put_file(&rec, &text, &chunks, &vectors)?;
    if prior {
        report.updated += 1;
    } else {
        report.added += 1;
    }
    Ok(report)
}

fn mtime_ns(meta: &std::fs::Metadata) -> i64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0)
}

/// Files whose size or mtime differ from the index, plus files on disk the
/// index lacks and files it holds that are gone. Cheap: no reading, no
/// hashing. A search prints a one-line stderr warning when this is non-zero;
/// it never updates the index itself.
pub fn stale_count(corpus: &Corpus, opts: &WalkOptions) -> Result<usize> {
    let Some(store) = Store::open_read(corpus)? else {
        return Ok(0);
    };
    let indexed = store.files().context("files")?;
    let walked = crate::walk::walk(corpus, opts)?;
    let mut stale = 0usize;
    for file in &walked {
        match indexed.get(&file.rel) {
            None => stale += 1,
            Some(rec) => {
                if rec.size != file.size || rec.mtime_ns != file.mtime_ns {
                    stale += 1;
                }
            }
        }
    }
    let live: std::collections::HashSet<&RelPath> = walked.iter().map(|f| &f.rel).collect();
    stale += indexed.keys().filter(|rel| !live.contains(*rel)).count();
    Ok(stale)
}

/// Remove the index (`--clean`): the whole `.ck` directory, including any
/// files the old engine left there (manifest.json, sidecars, vectors.cache,
/// tantivy_index).
pub fn clean(corpus: &Corpus) -> Result<()> {
    let dir = corpus.index_dir();
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    Ok(())
}
