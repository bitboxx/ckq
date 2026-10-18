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
    /// The head could not be read (permissions, a failing disk): not the
    /// same as binary. The file is still walked, so an index keeps its rows
    /// and reports it skipped instead of silently treating it as deleted.
    pub unreadable: bool,
}

/// Every indexable text file under the root, sorted by path — plus files
/// whose head could not be read, marked [`FileInfo::unreadable`]: dropping
/// them here would make `--index` delete a previously indexed file over a
/// transient permission error. Every file is probed.
pub fn walk(corpus: &Corpus, opts: &WalkOptions) -> Result<Vec<FileInfo>> {
    walk_known(corpus, opts, &std::collections::HashMap::new())
}

/// [`walk`] for a caller that already knows the index: a file whose size
/// and mtime match `indexed` was probed when it was indexed, so it is
/// assumed text and never opened — the stat comparison is the whole point
/// of the index, and re-reading 8 KiB of every file on every search made
/// the "cheap" staleness walk the most expensive part of one.
pub fn walk_known(
    corpus: &Corpus,
    opts: &WalkOptions,
    indexed: &std::collections::HashMap<RelPath, (u64, i64)>,
) -> Result<Vec<FileInfo>> {
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
        let mtime_ns = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);
        // A stat the index already vouches for: no probe, no open.
        if indexed
            .get(&rel)
            .is_some_and(|(size, mtime)| *size == meta.len() && *mtime == mtime_ns)
        {
            out.push(FileInfo {
                rel,
                size: meta.len(),
                mtime_ns,
                unreadable: false,
            });
            continue;
        }
        let unreadable = match probe(entry.path()) {
            Probe::Text => false,
            Probe::Binary => continue,
            Probe::Unreadable => true,
        };
        out.push(FileInfo {
            rel,
            size: meta.len(),
            mtime_ns,
            unreadable,
        });
    }
    out.sort_by(|a, b| a.rel.cmp(&b.rel));
    Ok(out)
}

/// What reading the first 8 KiB concluded. An open or read error is its own
/// verdict: treating it as "not text" made an unreadable file vanish from
/// the walk, and from the index with it.
enum Probe {
    Text,
    Binary,
    Unreadable,
}

/// Read the first 8 KiB and hand them to [`looks_binary`].
fn probe(path: &Path) -> Probe {
    let mut head = [0u8; 8192];
    let n = match File::open(path).and_then(|mut f| f.read(&mut head)) {
        Ok(n) => n,
        Err(_) => return Probe::Unreadable,
    };
    if looks_binary(&head[..n]) {
        Probe::Binary
    } else {
        Probe::Text
    }
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
