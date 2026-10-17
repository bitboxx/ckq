//! Which files are in the corpus.
//!
//! The `ignore` walker, so `.gitignore`, `.ignore` and `.ckignore` all apply.
//! Symlinked directories are never followed (a loop must not hang anything)
//! and the root is never glob-expanded. `.ck` and `.git` are always skipped,
//! and so is any file that looks binary.

use crate::corpus::{Corpus, RelPath};
use anyhow::Result;

/// Patterns a fresh `.ckignore` starts with, written by `--index` when the
/// corpus has none. Ported from upstream's default so existing corpora behave
/// the same.
pub const DEFAULT_CKIGNORE: &str = "";

#[derive(Debug, Clone, Default)]
pub struct WalkOptions {
    /// Include dot-files and dot-directories.
    pub hidden: bool,
    /// Ignore `.ckignore` files (`--no-ckignore`).
    pub no_ckignore: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileInfo {
    pub rel: RelPath,
    pub size: u64,
    /// Modification time in nanoseconds since the epoch.
    pub mtime_ns: i64,
}

/// Every indexable text file under the root, sorted by path.
pub fn walk(corpus: &Corpus, opts: &WalkOptions) -> Result<Vec<FileInfo>> {
    todo!("ignore::WalkBuilder: follow_links(false), hidden(!opts.hidden), add_custom_ignore_filename(\".ckignore\") unless no_ckignore, skip .ck/.git, skip binaries via looks_binary")
}

/// A file is binary when its first 8 KiB hold a NUL byte or are not UTF-8
/// after allowing for a cut multi-byte sequence at the end.
pub fn looks_binary(head: &[u8]) -> bool {
    todo!()
}

/// Write `DEFAULT_CKIGNORE` to the root when it has no `.ckignore` yet.
pub fn ensure_ckignore(corpus: &Corpus) -> Result<()> {
    todo!()
}
