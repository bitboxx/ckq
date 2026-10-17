//! Document identity.
//!
//! A document is its path relative to the corpus root, with `/` separators and
//! no leading `./`. That string is what the store keys on, what the lexical and
//! semantic methods return, and what fusion compares. Nothing else is ever
//! stored or compared. The bugs this replaces came from the old engine keeping
//! four path forms (manifest, sidecars, vector cache, tantivy) that disagreed,
//! so `--hybrid` never fused and `--sem` returned nothing from another cwd.
//!
//! Two functions leave this module with a filesystem path: [`Corpus::abs`], for
//! reading a file, and [`Corpus::display`], for printing one. Neither depends on
//! the process working directory.

use anyhow::{Result, bail};
use std::fmt;
use std::path::{Path, PathBuf};

/// Directory, under the corpus root, that holds the index.
pub const INDEX_DIR: &str = ".ck";

/// A path relative to the corpus root: `/`-separated, no leading `./`, never
/// empty, never containing `..`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RelPath(String);

impl RelPath {
    /// Build from a root-relative path. Rejects absolute paths, `..` and empties.
    pub fn new(rel: &Path) -> Result<Self> {
        todo!("normalise separators to '/', strip a leading './', reject '..' and absolute")
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// True when `self` is `prefix` or lies under it.
    pub fn starts_with(&self, prefix: &RelPath) -> bool {
        self.0 == prefix.0 || self.0.starts_with(&format!("{}/", prefix.0))
    }

    /// The file name's extension, lowercased, if any.
    pub fn extension(&self) -> Option<String> {
        Path::new(&self.0)
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
    }
}

impl fmt::Display for RelPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Which part of the corpus a search covers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Scope {
    All,
    /// A subdirectory, or a single file, of the corpus.
    Under(RelPath),
}

impl Scope {
    pub fn contains(&self, rel: &RelPath) -> bool {
        match self {
            Scope::All => true,
            Scope::Under(prefix) => rel.starts_with(prefix),
        }
    }
}

/// A corpus: a directory whose `.ck` holds the index.
#[derive(Debug, Clone)]
pub struct Corpus {
    /// Absolute root, from `std::path::absolute`: no symlink resolution, so the
    /// same argument always gives the same root, and `/var` stays `/var`.
    pub root: PathBuf,
    /// The root as output paths start: the caller's argument when it was
    /// absolute (a wrapper strips that prefix from every result), otherwise the
    /// caller's relative argument, `.` when none was given.
    display_root: PathBuf,
}

impl Corpus {
    /// The corpus a search of `arg` reads from: the nearest directory at or
    /// above `arg` that holds `.ck`, and the part of it `arg` names. An `arg`
    /// with no index above it is an error that says to run `ckq --index`.
    pub fn for_search(arg: &Path) -> Result<(Corpus, Scope)> {
        todo!("walk up from absolute(arg) to the first dir with .ck; scope = rel(arg) or All")
    }

    /// A corpus with no index, for regex mode: `arg` itself when it is a
    /// directory, its parent scoped to the file when it is a file.
    pub fn plain(arg: &Path) -> Result<(Corpus, Scope)> {
        if !arg.exists() {
            bail!("Path does not exist: {}", arg.display());
        }
        todo!()
    }

    /// The corpus `--index` writes: `arg` itself, created on first use.
    pub fn for_index(arg: &Path) -> Result<Corpus> {
        if !arg.is_dir() {
            bail!("Path does not exist or is not a directory: {}", arg.display());
        }
        todo!("root = absolute(arg); display_root from arg")
    }

    pub fn index_dir(&self) -> PathBuf {
        self.root.join(INDEX_DIR)
    }

    /// Root-relative identity of a path under the root (absolute, or relative
    /// to the root). `None` when it lies outside.
    pub fn rel(&self, path: &Path) -> Option<RelPath> {
        todo!()
    }

    /// Where to read the file. Absolute, independent of the working directory.
    pub fn abs(&self, rel: &RelPath) -> PathBuf {
        self.root.join(rel.as_str())
    }

    /// How the file is printed: `display_root` joined with `rel`. Every output
    /// path goes through here, whichever method found the file.
    pub fn display(&self, rel: &RelPath) -> PathBuf {
        self.display_root.join(rel.as_str())
    }
}
