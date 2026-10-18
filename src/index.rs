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
use anyhow::{Context, Result, anyhow, bail};
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
    // --no-ckqignore means the run ignores .ckqignore files; writing a
    // default one into a corpus that has none would mutate the very tree
    // the flag asked to leave alone, and later runs would read a file the
    // user never authored.
    if !opts.walk.no_ckqignore {
        crate::walk::ensure_ckqignore(corpus)?;
    }
    let (spec, meta) = resolve_model(corpus, opts)?;
    // A different model than the index holds means a full rebuild: its
    // vectors live in a different space and must not be mixed. A rebuild
    // discards the old stats too, so every file is probed afresh.
    let mut known = std::collections::HashMap::new();
    // The read connection must be gone before `clean` removes the store:
    // on Windows a file an open handle holds cannot be deleted, and the
    // sharing violation is the whole difference from unix's unlink.
    let rebuild = match Store::open_read(corpus)? {
        Some(store) => {
            let existing = store.meta()?;
            if existing.signature != meta.signature {
                if !opts.quiet {
                    eprintln!(
                        "index model changed from {} to {}; rebuilding",
                        existing.model_alias, meta.model_alias
                    );
                }
                true
            } else {
                known = store
                    .files()
                    .context("files")?
                    .into_iter()
                    .map(|(rel, rec)| (rel, (rec.size, rec.mtime_ns)))
                    .collect();
                false
            }
        }
        None => false,
    };
    if rebuild {
        clean(corpus)?;
    }

    // Files whose stat the index vouches for are not opened (see
    // `walk_known`); only new or changed files pay the 8 KiB probe.
    let walked = crate::walk::walk_known(corpus, &opts.walk, &known)?;
    let t_walk = t0.elapsed();

    let mut store = Store::open_write(corpus, &meta).context("open_write")?;
    let indexed = store.files().context("files")?;
    let mut report = IndexReport::default();

    let mut to_embed: Vec<(FileRecord, String, Vec<crate::chunk::Chunk>)> = Vec::new();
    let t_read = Instant::now();
    for file in &walked {
        // Same size and mtime as indexed: trust it without reading. The hook
        // that keeps a corpus current runs this after nearly every write, so
        // reading and hashing every file each time would be the whole cost.
        if indexed
            .get(&file.rel)
            .is_some_and(|p| p.size == file.size && p.mtime_ns == file.mtime_ns)
        {
            report.unchanged += 1;
            continue;
        }
        // A file whose head the walk could not open is skipped, not removed:
        // its rows stay (it is in the live set below), one warning names it,
        // and the next run that can read it embeds it again.
        if file.unreadable {
            report.skipped += 1;
            eprintln!(
                "ckq: warning: cannot read {}; keeping its indexed rows",
                corpus.abs(&file.rel).display()
            );
            continue;
        }
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
            Err(e) => {
                report.skipped += 1;
                eprintln!(
                    "ckq: warning: cannot read {}: {e}; keeping its indexed rows",
                    corpus.abs(&file.rel).display()
                );
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
    // A file the walk filters out would be indexed here and then counted
    // "gone" by every later search -- a permanent staleness warning -- until
    // the next `--index` silently removed it again. Refuse it up front, by
    // the same walk rules an index run applies.
    let walked = crate::walk::walk(corpus, &opts.walk)?;
    if !walked.iter().any(|f| f.rel == *rel) {
        bail!(
            "{} is excluded from the index (ignored, hidden or binary); pass the \
             walk flags (--hidden, --no-ignore, --no-ckqignore) to index it",
            corpus.abs(rel).display()
        );
    }
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
/// index lacks and files it holds that are gone. Cheap by construction: it
/// stats every file but opens only the new and changed ones — a file whose
/// stat matches the index is never read. A search prints a one-line stderr
/// warning when this is non-zero; it never updates the index itself.
pub fn stale_count(corpus: &Corpus, opts: &WalkOptions) -> Result<usize> {
    let Some(store) = Store::open_read(corpus)? else {
        return Ok(0);
    };
    let indexed = store.files().context("files")?;
    let known: std::collections::HashMap<RelPath, (u64, i64)> = indexed
        .iter()
        .map(|(rel, rec)| (rel.clone(), (rec.size, rec.mtime_ns)))
        .collect();
    let walked = crate::walk::walk_known(corpus, opts, &known)?;
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

/// Remove the index (`--clean`): the whole `.ckq` directory. A legacy `.ck`
/// left by the old engine is deliberately not touched — there is no
/// migration — so the caller can say it stayed.
pub fn clean(corpus: &Corpus) -> Result<bool> {
    let dir = corpus.index_dir();
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    Ok(corpus.root.join(crate::corpus::LEGACY_INDEX_DIR).exists())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::{Corpus, RelPath};
    use crate::store::{FileRecord, IndexMeta, Store};

    fn spec() -> &'static ModelSpec {
        models::resolve(models::DEFAULT_ALIAS).unwrap()
    }

    /// A file that turns unreadable between runs is skipped, not deleted:
    /// its rows survive so a transient permission error does not cost a
    /// re-embed of the whole file. Unix only: the error is modelled with
    /// chmod, and root ignores permissions.
    #[cfg(unix)]
    #[test]
    fn an_unreadable_file_keeps_its_rows_and_counts_skipped() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::TempDir::new().unwrap();
        let corpus = Corpus::for_index(dir.path()).unwrap();
        std::fs::write(dir.path().join("gone.md"), "needle\n").unwrap();
        std::fs::write(dir.path().join("keep.md"), "safe\n").unwrap();
        let rec = FileRecord {
            rel: RelPath::new(std::path::Path::new("gone.md")).unwrap(),
            hash: "h".to_string(),
            size: 7,
            mtime_ns: 1,
        };
        let chunk = crate::chunk::Chunk {
            byte_start: 0,
            byte_end: 6,
            line_start: 1,
            line_end: 1,
        };
        let mut store = Store::open_write(&corpus, &IndexMeta::for_model(spec())).unwrap();
        store.put_file(&rec, "needle", &[chunk], &[]).unwrap();
        drop(store);

        let locked = dir.path().join("gone.md");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        let report = update(&corpus, &IndexOptions::default()).unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o644)).unwrap();

        assert_eq!(report.removed, 0, "{report:?}");
        assert_eq!(report.skipped, 1, "{report:?}");
        let store = Store::open_read(&corpus).unwrap().unwrap();
        assert!(
            store
                .files()
                .unwrap()
                .contains_key(&RelPath::new(std::path::Path::new("gone.md")).unwrap())
        );
    }

    /// A file whose size and mtime match the index is never opened: its
    /// content here is binary, which a probe would see and drop, yet the
    /// stat is what the index vouches for — so staleness stays 0 and
    /// `--index` counts the file unchanged without reading it.
    #[test]
    fn stale_count_and_update_never_open_a_stat_matching_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let corpus = Corpus::for_index(dir.path()).unwrap();
        let mut bytes = vec![b'x'; 100];
        bytes[50] = 0; // a NUL: any 8 KiB probe would call this binary
        let path = dir.path().join("fake.md");
        std::fs::write(&path, &bytes).unwrap();
        let when = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
        // write-capable handle: a read-only one cannot set timestamps on
        // Windows (Access denied), and unix does not care either way.
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(when)
            .unwrap();
        let mtime_ns = when
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i64;
        let rec = FileRecord {
            rel: RelPath::new(std::path::Path::new("fake.md")).unwrap(),
            hash: "h".to_string(),
            size: bytes.len() as u64,
            mtime_ns,
        };
        let mut store = Store::open_write(&corpus, &IndexMeta::for_model(spec())).unwrap();
        store.put_file(&rec, "indexed text", &[], &[]).unwrap();
        drop(store);

        assert_eq!(
            stale_count(&corpus, &WalkOptions::default()).unwrap(),
            0,
            "a matching stat must not be probed"
        );
        let report = update(&corpus, &IndexOptions::default()).unwrap();
        assert_eq!(report.unchanged, 1, "{report:?}");
        assert_eq!(report.removed, 0, "{report:?}");

        // Touch the file and the mismatch shows up again.
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(std::time::SystemTime::now())
            .unwrap();
        assert_eq!(
            stale_count(&corpus, &WalkOptions::default()).unwrap(),
            1,
            "a changed stat must be seen"
        );
    }
}
