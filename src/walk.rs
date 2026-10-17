//! Which files are in the corpus.
//!
//! The `ignore` walker, so `.gitignore`, `.ignore` and `.ckqignore` all apply.
//! Symlinked directories are never followed (a loop must not hang anything)
//! and the root is never glob-expanded. `.ckq` and `.git` are always skipped,
//! and so is a legacy `.ck` left by the old engine, and any file that looks
//! binary.

use crate::corpus::{Corpus, RelPath};
use anyhow::Result;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Patterns a fresh `.ckqignore` starts with, written by `--index` when the
/// corpus has none. Ported from upstream's default so existing corpora behave
/// the same.
pub const DEFAULT_CKQIGNORE: &str = r"# .ckqignore - Default patterns for ckq semantic search
# Created automatically during first index
# Syntax: same as .gitignore (glob patterns, ! for negation)

# Images
*.png
*.jpg
*.jpeg
*.gif
*.bmp
*.svg
*.ico
*.webp
*.tiff

# Video
*.mp4
*.avi
*.mov
*.mkv
*.wmv
*.flv
*.webm

# Audio
*.mp3
*.wav
*.flac
*.aac
*.ogg
*.m4a

# Binary/Compiled
*.exe
*.dll
*.so
*.dylib
*.a
*.lib
*.obj
*.o

# Archives
*.zip
*.tar
*.tar.gz
*.tgz
*.rar
*.7z
*.bz2
*.gz

# Data files
*.db
*.sqlite
*.sqlite3
*.parquet
*.arrow

# Config formats (issue #27)
*.json
*.yaml
*.yml

# Add your custom patterns below this line
";

#[derive(Debug, Clone, Default)]
pub struct WalkOptions {
    /// Include dot-files and dot-directories.
    pub hidden: bool,
    /// Ignore `.ckqignore` files (`--no-ckqignore`).
    pub no_ckqignore: bool,
    /// Do not read `.gitignore` files (`--no-ignore`).
    pub no_ignore: bool,
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
    let mut builder = ignore::WalkBuilder::new(&corpus.root);
    builder
        .follow_links(false)
        .hidden(!opts.hidden)
        .git_ignore(!opts.no_ignore)
        .git_global(!opts.no_ignore)
        .git_exclude(!opts.no_ignore)
        .ignore(!opts.no_ignore)
        .parents(true);
    if !opts.no_ckqignore {
        builder.add_custom_ignore_filename(".ckqignore");
    }
    builder.filter_entry(|entry| {
        if entry.depth() == 0 {
            return true;
        }
        let name = entry.file_name().to_string_lossy();
        name != crate::corpus::INDEX_DIR
            && name != crate::corpus::LEGACY_INDEX_DIR
            && name != ".git"
    });

    let mut out = Vec::new();
    for entry in builder.build() {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue, // an unreadable directory is not a failure
        };
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let Some(rel) = corpus.rel(entry.path()) else {
            continue;
        };
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if !is_text(entry.path()) {
            continue;
        }
        let mtime_ns = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);
        out.push(FileInfo {
            rel,
            size: meta.len(),
            mtime_ns,
        });
    }
    out.sort_by(|a, b| a.rel.cmp(&b.rel));
    Ok(out)
}

/// Read the first 8 KiB and hand them to [`looks_binary`].
fn is_text(path: &Path) -> bool {
    let mut head = [0u8; 8192];
    let n = match File::open(path).and_then(|mut f| f.read(&mut head)) {
        Ok(n) => n,
        Err(_) => return false,
    };
    !looks_binary(&head[..n])
}

/// A file is binary when its first 8 KiB hold a NUL byte or are not UTF-8
/// after allowing for a cut multi-byte sequence at the end.
pub fn looks_binary(head: &[u8]) -> bool {
    if head.contains(&0) {
        return true;
    }
    match std::str::from_utf8(head) {
        Ok(_) => false,
        Err(e) => {
            // A cut multi-byte sequence at the end of the head is fine; any
            // other invalid UTF-8 is binary.
            !(e.error_len().is_none() && head.len() - e.valid_up_to() <= 4)
        }
    }
}

/// Write `DEFAULT_CKQIGNORE` to the root when it has no `.ckqignore` yet.
pub fn ensure_ckqignore(corpus: &Corpus) -> Result<()> {
    let path = corpus.root.join(".ckqignore");
    if !path.exists() {
        std::fs::write(&path, DEFAULT_CKQIGNORE)?;
    }
    Ok(())
}
