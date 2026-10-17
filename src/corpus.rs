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
        let mut parts: Vec<&std::ffi::OsStr> = Vec::new();
        for comp in rel.components() {
            use std::path::Component::*;
            match comp {
                Normal(p) => parts.push(p),
                CurDir => {}
                RootDir | Prefix(_) => bail!("path is absolute: {}", rel.display()),
                ParentDir => bail!("path leaves the corpus: {}", rel.display()),
            }
        }
        if parts.is_empty() {
            bail!("empty path");
        }
        let joined = parts
            .iter()
            .map(|p| p.to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        Ok(Self(joined))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// True when `self` is `prefix` or lies under it.
    #[allow(dead_code)] // part of the module's surface; the store scopes in SQL
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
    /// Kept with `RelPath::starts_with`: the pair is the module's surface.
    #[allow(dead_code)]
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
        let abs = absolute(arg);
        let start = if abs.is_dir() { &abs } else { parent(&abs)? };
        let Some(root) = nearest_index_root(start) else {
            bail!(
                "No ckq index found for {}; run `ckq --index {}` first",
                arg.display(),
                arg.display()
            );
        };
        let scope = if root == start {
            Scope::All
        } else {
            Scope::Under(rel_of(&root, start)?)
        };
        Ok((Corpus::at(&root, &display_prefix(arg, &abs)), scope))
    }

    /// A corpus with no index, for regex mode: `arg` itself when it is a
    /// directory, its parent scoped to the file when it is a file.
    pub fn plain(arg: &Path) -> Result<(Corpus, Scope)> {
        if !arg.exists() {
            bail!("Path does not exist: {}", arg.display());
        }
        let abs = absolute(arg);
        if abs.is_dir() {
            return Ok((
                Corpus::at(&abs, &display_prefix(arg, &abs)),
                Scope::All,
            ));
        }
        let parent = parent(&abs)?;
        Ok((
            Corpus::at(&parent, &display_prefix(arg, &abs)),
            Scope::Under(rel_of(&parent, &abs)?),
        ))
    }

    /// The corpus `--index` writes: `arg` itself, created on first use.
    pub fn for_index(arg: &Path) -> Result<Corpus> {
        if !arg.is_dir() {
            bail!("Path does not exist or is not a directory: {}", arg.display());
        }
        let abs = absolute(arg);
        Ok(Corpus::at(&abs, &display_prefix(arg, &abs)))
    }

    /// The corpus a `--clean`/`--status`/`--add` argument names: the nearest
    /// directory at or above `arg` that holds `.ck`, else `arg` itself when it
    /// is a directory.
    pub fn for_command(arg: &Path) -> Result<Corpus> {
        let abs = absolute(arg);
        let start = if abs.is_dir() { &abs } else { parent(&abs)? };
        let root = nearest_index_root(start).unwrap_or_else(|| start.to_path_buf());
        Ok(Corpus::at(&root, &display_prefix(arg, &abs)))
    }

    fn at(root: &Path, display_root: &Path) -> Corpus {
        Corpus {
            root: root.to_path_buf(),
            display_root: display_root.to_path_buf(),
        }
    }

    pub fn index_dir(&self) -> PathBuf {
        self.root.join(INDEX_DIR)
    }

    /// Root-relative identity of a path under the root (absolute, or relative
    /// to the root). `None` when it lies outside.
    pub fn rel(&self, path: &Path) -> Option<RelPath> {
        let rel = if path.is_absolute() {
            path.strip_prefix(&self.root).ok()?
        } else {
            path
        };
        RelPath::new(rel).ok()
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

/// `std::path::absolute` against the process working directory.
fn absolute(p: &Path) -> PathBuf {
    std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf())
}

fn parent(p: &Path) -> Result<&Path> {
    p.parent().ok_or_else(|| anyhow::anyhow!("{} has no parent", p.display()))
}

/// The nearest directory at or above `from` that holds a `.ck` directory.
fn nearest_index_root(from: &Path) -> Option<PathBuf> {
    let mut cur = Some(from);
    while let Some(dir) = cur {
        if dir.join(INDEX_DIR).is_dir() {
            return Some(dir.to_path_buf());
        }
        cur = dir.parent();
    }
    None
}

fn rel_of(root: &Path, below: &Path) -> Result<RelPath> {
    RelPath::new(below.strip_prefix(root).unwrap_or(below))
}

/// The prefix output paths start with: the argument as given when it was
/// relative (`.` when none was given — the caller passes `.` itself), the
/// argument's parent when it named a file, so `display(rel)` reproduces what
/// the caller typed.
fn display_prefix(arg: &Path, abs: &Path) -> PathBuf {
    if abs.is_dir() {
        arg.to_path_buf()
    } else {
        arg.parent().map(Path::to_path_buf).unwrap_or_default()
    }
}
