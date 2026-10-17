//! Building and updating the index. The only code that writes the store.
//!
//! A file is re-read only when its size or mtime changed, and re-embedded
//! only when its blake3 hash changed. Deleted files are removed in the same
//! run. The model comes from `--model`, else from the existing index, else
//! the default; a different model than the index holds means a full rebuild.

use crate::corpus::{Corpus, RelPath};
use crate::walk::WalkOptions;
use anyhow::Result;

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

/// Bring the index in line with the files on disk.
///
/// With `CKQ_TIMING=1`, print one line to stderr with the time spent walking,
/// reading, chunking, embedding and writing.
pub fn update(corpus: &Corpus, opts: &IndexOptions) -> Result<IndexReport> {
    todo!("walk; diff against store.files(); embed new/changed in batches on embed::Device::Gpu; put_file / remove_file")
}

/// Index or re-index one file (`--add`), leaving the rest as it is.
pub fn add_file(corpus: &Corpus, rel: &RelPath, opts: &IndexOptions) -> Result<IndexReport> {
    todo!()
}

/// Files whose size or mtime differ from the index, plus files on disk the
/// index lacks and files it holds that are gone. Cheap: no reading, no
/// hashing. A search prints a one-line stderr warning when this is non-zero;
/// it never updates the index itself.
pub fn stale_count(corpus: &Corpus, opts: &WalkOptions) -> Result<usize> {
    todo!()
}

/// Remove the index (`--clean`): the whole `.ck` directory, including any
/// files the old engine left there (manifest.json, sidecars, vectors.cache,
/// tantivy_index).
pub fn clean(corpus: &Corpus) -> Result<()> {
    todo!()
}
