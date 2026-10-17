//! The index: one SQLite file, `.ck/index.db`, in WAL mode.
//!
//! It replaces the old manifest, per-file sidecars, `vectors.cache` and tantivy
//! directory. Each file is written in one transaction, so a crash or a second
//! process can never see half a file, there are no orphans to clean and no
//! cache to invalidate. Readers never block the writer and the writer never
//! blocks readers; two writers queue on SQLite's own lock (`busy_timeout`).
//!
//! Search opens the store read-only. Only `--index`, `--add` and `--clean`
//! write it.

use crate::chunk::Chunk;
use crate::corpus::{Corpus, RelPath, Scope};
use anyhow::{Context, Result, anyhow, bail};
use rusqlite::{Connection, OpenFlags, TransactionBehavior, params};
use std::collections::HashMap;

pub const DB_FILE: &str = "index.db";
pub const SCHEMA_VERSION: i64 = 1;

/// The schema. `fts` is contentless: it holds the BM25 index but not the text,
/// which stays in the files. Its rowid is `files.id`. Columns are `path` then
/// `content`, and ranking is `bm25(fts, 0.5, 1.0)`: the path counts half, as in
/// the old engine. The tokenizer keeps diacritics, as tantivy's default did, so
// `cafe` does not match `café` and whole words still do not match compounds.
pub const SCHEMA: &str = "
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    hash TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    byte_start INTEGER NOT NULL,
    byte_end INTEGER NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    vec BLOB
);
CREATE INDEX IF NOT EXISTS chunks_file ON chunks(file_id);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
    path, content, content = '', contentless_delete = 1,
    tokenize = 'unicode61 remove_diacritics 0'
);
";

/// What the index was built with. Stored in `meta`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexMeta {
    pub schema_version: i64,
    /// Model alias and repo, from `models::ModelSpec`.
    pub model_alias: String,
    pub model_repo: String,
    pub dims: usize,
    /// `models::signature`; vectors with another signature are not comparable.
    pub signature: String,
}

impl IndexMeta {
    pub fn for_model(spec: &crate::models::ModelSpec) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            model_alias: spec.alias.to_string(),
            model_repo: spec.repo.to_string(),
            dims: spec.dims,
            signature: crate::models::signature(spec),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileRecord {
    pub rel: RelPath,
    /// blake3 of the bytes, hex.
    pub hash: String,
    pub size: u64,
    pub mtime_ns: i64,
}

/// A chunk located in the corpus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkRef {
    pub rel: RelPath,
    pub chunk: Chunk,
}

/// Every chunk vector in a scope, as one contiguous row-major matrix.
#[derive(Debug, Clone)]
pub struct VectorSet {
    pub dims: usize,
    /// `chunks.len() * dims` values, unit rows.
    pub data: Vec<f32>,
    pub chunks: Vec<ChunkRef>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndexStats {
    pub files: usize,
    pub chunks: usize,
    pub embedded_chunks: usize,
    /// Sum of the indexed files' sizes.
    pub source_bytes: u64,
    pub db_bytes: u64,
}

#[derive(Debug)]
pub struct Store {
    conn: Connection,
}

impl Store {
    /// Open an existing index read-only. `None` when the corpus has none —
    /// including when a `index.db` exists but carries no schema yet, which is
    /// what another process's first, still-running `--index` looks like.
    pub fn open_read(corpus: &Corpus) -> Result<Option<Store>> {
        let path = corpus.index_dir().join(DB_FILE);
        if !path.is_file() {
            return Ok(None);
        }
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        // Before the first index switches the file to WAL, a reader can meet
        // the writer's exclusive journal lock; wait it out instead of failing.
        conn.busy_timeout(std::time::Duration::from_secs(30))?;
        // Schema without meta rows is another process's first `--index`
        // still in flight: to every reader that is no index yet. The table
        // check runs first because querying an absent table is an error.
        let table: i64 = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = 'meta'",
                [],
                |r| r.get(0),
            )
            .context("ready check")?;
        if table == 0 {
            return Ok(None);
        }
        let rows: i64 = conn
            .query_row("SELECT count(*) FROM meta", [], |r| r.get(0))
            .context("ready check")?;
        if rows == 0 {
            return Ok(None);
        }
        Ok(Some(Store { conn }))
    }

    /// Open for writing, creating the file and schema when absent. A new index
    /// takes `meta`; an existing one must already carry the same signature,
    /// otherwise this fails with a message that names `--clean`.
    pub fn open_write(corpus: &Corpus, meta: &IndexMeta) -> Result<Store> {
        std::fs::create_dir_all(corpus.index_dir())?;
        let path = corpus.index_dir().join(DB_FILE);
        let conn = Connection::open(&path)?;
        conn.busy_timeout(std::time::Duration::from_secs(30))?;
        conn.pragma_update(None, "foreign_keys", true)?;
        // A journal-mode change wants the file to itself and does not honour
        // busy_timeout, so a second `--index` racing the first one's schema
        // creation sees BUSY from the batch. The batch is idempotent; retry
        // it until the other writer is done. Whether an index was there
        // before is read afterwards, for the same reason.
        schema_with_retry(&conn)?;
        let stored: Option<String> = conn
            .query_row("SELECT value FROM meta WHERE key = 'signature'", [], |r| {
                r.get(0)
            })
            .ok();
        if let Some(old) = stored.as_ref()
            && *old != meta.signature
        {
            bail!(
                "the index at {} was built with a different embedding signature \
                 ({old} != {}); run `ckq --clean` and index again",
                corpus.index_dir().display(),
                meta.signature
            );
        }
        let store = Store { conn };
        if stored.is_none() {
            let IndexMeta {
                schema_version,
                model_alias,
                model_repo,
                dims,
                signature,
            } = meta;
            // One statement, so a concurrent reader never meets a half-written
            // meta and mistakes it for an index built by another model.
            store.conn.execute(
                "INSERT INTO meta (key, value) VALUES \
                 ('schema_version', ?1), ('model_alias', ?2), ('model_repo', ?3), \
                 ('dims', ?4), ('signature', ?5)",
                params![
                    schema_version.to_string(),
                    model_alias.clone(),
                    model_repo.clone(),
                    dims.to_string(),
                    signature.clone()
                ],
            )?;
        }
        Ok(store)
    }

    pub fn meta(&self) -> Result<IndexMeta> {
        let get = |key: &str| -> Result<String> {
            Ok(self
                .conn
                .query_row("SELECT value FROM meta WHERE key = ?1", params![key], |r| {
                    r.get(0)
                })?)
        };
        Ok(IndexMeta {
            schema_version: get("schema_version")?.parse()?,
            model_alias: get("model_alias")?,
            model_repo: get("model_repo")?,
            dims: get("dims")?.parse()?,
            signature: get("signature")?,
        })
    }

    pub fn files(&self) -> Result<HashMap<RelPath, FileRecord>> {
        let mut stmt = self
            .conn
            .prepare("SELECT path, hash, size, mtime_ns FROM files")?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, i64>(3)?,
            ))
        })?;
        let mut out = HashMap::new();
        for row in rows {
            let (path, hash, size, mtime_ns) = row?;
            let rel = RelPath::new(std::path::Path::new(&path))?;
            out.insert(
                rel.clone(),
                FileRecord {
                    rel,
                    hash,
                    size: size as u64,
                    mtime_ns,
                },
            );
        }
        Ok(out)
    }

    /// Replace everything stored for one file, in one transaction: its row, its
    /// chunks with their vectors (stored as little-endian f16), and its BM25
    /// entry over (path, text). `vectors` is empty or one per chunk.
    pub fn put_file(
        &mut self,
        file: &FileRecord,
        text: &str,
        chunks: &[Chunk],
        vectors: &[Vec<f32>],
    ) -> Result<()> {
        // IMMEDIATE takes the write lock at BEGIN, so a second writer waits
        // on busy_timeout instead of hitting the upgrade deadlock that
        // SQLITE_BUSY reports immediately, without calling the busy handler.
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute(
            "DELETE FROM fts WHERE rowid = (SELECT id FROM files WHERE path = ?1)",
            params![file.rel.as_str()],
        )?;
        tx.execute(
            "DELETE FROM files WHERE path = ?1",
            params![file.rel.as_str()],
        )?;
        tx.execute(
            "INSERT INTO files (path, hash, size, mtime_ns) VALUES (?1, ?2, ?3, ?4)",
            params![
                file.rel.as_str(),
                file.hash,
                file.size as i64,
                file.mtime_ns
            ],
        )?;
        let id: i64 = tx.query_row(
            "SELECT id FROM files WHERE path = ?1",
            params![file.rel.as_str()],
            |r| r.get(0),
        )?;
        for (ord, chunk) in chunks.iter().enumerate() {
            let vec: Option<Vec<u8>> = vectors.get(ord).map(|v| encode_vector(v));
            tx.execute(
                "INSERT INTO chunks (file_id, ord, byte_start, byte_end, line_start, line_end, vec) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    id,
                    ord as i64,
                    chunk.byte_start as i64,
                    chunk.byte_end as i64,
                    chunk.line_start as i64,
                    chunk.line_end as i64,
                    vec
                ],
            )?;
        }
        tx.execute(
            "INSERT INTO fts (rowid, path, content) VALUES (?1, ?2, ?3)",
            params![id, file.rel.as_str(), text],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Refresh the stat columns of a file whose bytes did not change.
    pub fn note_stat(&mut self, file: &FileRecord) -> Result<()> {
        self.conn.execute(
            "UPDATE files SET hash = ?2, size = ?3, mtime_ns = ?4 WHERE path = ?1",
            params![
                file.rel.as_str(),
                file.hash,
                file.size as i64,
                file.mtime_ns
            ],
        )?;
        Ok(())
    }

    /// Drop a file and everything stored for it, in one transaction.
    pub fn remove_file(&mut self, rel: &RelPath) -> Result<()> {
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute(
            "DELETE FROM fts WHERE rowid = (SELECT id FROM files WHERE path = ?1)",
            params![rel.as_str()],
        )?;
        tx.execute("DELETE FROM files WHERE path = ?1", params![rel.as_str()])?;
        tx.commit()?;
        Ok(())
    }

    /// All chunk vectors whose file is in `scope`.
    pub fn vectors(&self, scope: &Scope) -> Result<VectorSet> {
        const COLS: &str = "f.path, c.byte_start, c.byte_end, c.line_start, c.line_end, c.vec, \
                            (SELECT value FROM meta WHERE key = 'dims')";
        let mut stmt = match scope {
            Scope::All => self.conn.prepare(&format!(
                "SELECT {COLS} FROM chunks c JOIN files f ON f.id = c.file_id \
                 WHERE c.vec IS NOT NULL ORDER BY f.path, c.ord"
            ))?,
            Scope::Under(_) => self.conn.prepare(&format!(
                "SELECT {COLS} FROM chunks c JOIN files f ON f.id = c.file_id \
                 WHERE c.vec IS NOT NULL AND (f.path = ?1 OR f.path LIKE ?2 ESCAPE '\\') \
                 ORDER BY f.path, c.ord"
            ))?,
        };
        let mut dims = 0usize;
        let mut chunks = Vec::new();
        let mut data = Vec::new();
        let mut fill = |row: &rusqlite::Row<'_>| -> Result<()> {
            let path: String = row.get(0)?;
            let byte_start: i64 = row.get(1)?;
            let byte_end: i64 = row.get(2)?;
            let line_start: i64 = row.get(3)?;
            let line_end: i64 = row.get(4)?;
            let vec: Vec<u8> = row.get(5)?;
            let d: String = row.get(6)?;
            dims = d.parse()?;
            chunks.push(ChunkRef {
                rel: RelPath::new(std::path::Path::new(&path))?,
                chunk: Chunk {
                    byte_start: byte_start as usize,
                    byte_end: byte_end as usize,
                    line_start: line_start as usize,
                    line_end: line_end as usize,
                },
            });
            decode_vector(&vec, &mut data);
            Ok(())
        };
        let mut rows = match scope {
            Scope::All => stmt.query([])?,
            Scope::Under(prefix) => {
                let like = format!("{}%", escape_like(prefix.as_str()));
                stmt.query(params![prefix.as_str(), like])?
            }
        };
        while let Some(row) = rows.next()? {
            fill(row)?;
        }
        Ok(VectorSet { dims, data, chunks })
    }

    /// The chunks of one file, in order.
    #[allow(dead_code)] // part of the store's surface; spans live in chunks
    pub fn chunks_of(&self, rel: &RelPath) -> Result<Vec<Chunk>> {
        let mut stmt = self.conn.prepare(
            "SELECT byte_start, byte_end, line_start, line_end FROM chunks \
             WHERE file_id = (SELECT id FROM files WHERE path = ?1) ORDER BY ord",
        )?;
        let rows = stmt.query_map(params![rel.as_str()], |r| {
            Ok(Chunk {
                byte_start: r.get::<_, i64>(0)? as usize,
                byte_end: r.get::<_, i64>(1)? as usize,
                line_start: r.get::<_, i64>(2)? as usize,
                line_end: r.get::<_, i64>(3)? as usize,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// BM25 over files. `match_expr` is an FTS5 MATCH expression built by
    /// `search::fts_query`. Best first, at most `limit`, files in `scope` only.
    /// Scores are positive, higher is better (the negation of FTS5's `bm25()`).
    pub fn bm25(
        &self,
        match_expr: &str,
        scope: &Scope,
        limit: usize,
    ) -> Result<Vec<(RelPath, f32)>> {
        let base = "SELECT f.path, bm25(fts, 0.5, 1.0) FROM fts JOIN files f ON f.id = fts.rowid";
        let sql = match scope {
            Scope::All => format!("{base} WHERE fts MATCH ?1 ORDER BY 2 LIMIT ?2"),
            Scope::Under(_) => format!(
                "{base} WHERE fts MATCH ?1 AND (f.path = ?3 OR f.path LIKE ?4 ESCAPE '\\') \
                 ORDER BY 2 LIMIT ?2"
            ),
        };
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = match scope {
            Scope::All => stmt.query(params![match_expr, limit as i64])?,
            Scope::Under(prefix) => {
                let like = format!("{}%", escape_like(prefix.as_str()));
                stmt.query(params![match_expr, limit as i64, prefix.as_str(), like])?
            }
        };
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            let path: String = row.get(0)?;
            let score: f32 = row.get(1)?;
            out.push((RelPath::new(std::path::Path::new(&path))?, -score));
        }
        Ok(out)
    }

    pub fn stats(&self) -> Result<IndexStats> {
        let files: i64 = self
            .conn
            .query_row("SELECT count(*) FROM files", [], |r| r.get(0))?;
        let chunks: i64 = self
            .conn
            .query_row("SELECT count(*) FROM chunks", [], |r| r.get(0))?;
        let embedded: i64 = self.conn.query_row(
            "SELECT count(*) FROM chunks WHERE vec IS NOT NULL",
            [],
            |r| r.get(0),
        )?;
        let source: i64 =
            self.conn
                .query_row("SELECT coalesce(sum(size), 0) FROM files", [], |r| r.get(0))?;
        let db = self
            .conn
            .query_row(
                "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()",
                [],
                |r| r.get::<_, i64>(0),
            )
            .unwrap_or(0) as u64;
        Ok(IndexStats {
            files: files as usize,
            chunks: chunks as usize,
            embedded_chunks: embedded as usize,
            source_bytes: source as u64,
            db_bytes: db,
        })
    }
}

/// `%` and `_` escaped for a `LIKE ... ESCAPE '\'` pattern.
fn escape_like(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

fn schema_with_retry(conn: &Connection) -> Result<()> {
    let mut last: Option<rusqlite::Error> = None;
    for _ in 0..30 {
        match conn.execute_batch(SCHEMA) {
            Ok(()) => return Ok(()),
            Err(e) if is_busy(&e) => {
                last = Some(e);
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
            Err(e) => return Err(e.into()),
        }
    }
    Err(anyhow!(
        "the index stayed locked by another ckq for 30 s: {}",
        last.expect("a busy error")
    ))
}

fn is_busy(e: &rusqlite::Error) -> bool {
    matches!(
        e.sqlite_error_code(),
        Some(rusqlite::ErrorCode::DatabaseBusy)
    ) || e.to_string().contains("database is locked")
}

/// f32 to little-endian f16 bytes, and back.
pub fn encode_vector(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        let bits = half::f16::from_f32(*x).to_le_bytes();
        out.extend_from_slice(&bits);
    }
    out
}

pub fn decode_vector(bytes: &[u8], out: &mut Vec<f32>) {
    let (pairs, _) = bytes.as_chunks::<2>();
    out.extend(pairs.iter().map(|p| half::f16::from_le_bytes(*p).to_f32()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::Corpus;
    use tempfile::TempDir;

    fn corpus() -> (TempDir, Corpus) {
        let dir = TempDir::new().unwrap();
        let c = Corpus::for_index(dir.path()).unwrap();
        (dir, c)
    }

    fn meta() -> IndexMeta {
        IndexMeta::for_model(crate::models::resolve(crate::models::DEFAULT_ALIAS).unwrap())
    }

    fn rec(rel: &str, hash: &str) -> FileRecord {
        FileRecord {
            rel: RelPath::new(std::path::Path::new(rel)).unwrap(),
            hash: hash.to_string(),
            size: 10,
            mtime_ns: 1,
        }
    }

    #[test]
    fn open_read_is_none_without_an_index() {
        let (_dir, c) = corpus();
        assert!(Store::open_read(&c).unwrap().is_none());
    }

    #[test]
    fn put_file_round_trips_text_chunks_and_vectors() {
        let (_dir, c) = corpus();
        let mut store = Store::open_write(&c, &meta()).unwrap();
        let chunk = Chunk {
            byte_start: 0,
            byte_end: 5,
            line_start: 1,
            line_end: 1,
        };
        store
            .put_file(
                &rec("a.md", "h1"),
                "hello quokka",
                &[chunk],
                &[vec![0.5; 768]],
            )
            .unwrap();
        let files = store.files().unwrap();
        assert!(files.contains_key(&RelPath::new(std::path::Path::new("a.md")).unwrap()));
        let vs = store.vectors(&Scope::All).unwrap();
        assert_eq!(vs.dims, 768);
        assert_eq!(vs.chunks.len(), 1);
        assert_eq!(vs.data.len(), 768);
        let stats = store.stats().unwrap();
        assert_eq!(stats.files, 1);
        assert_eq!(stats.embedded_chunks, 1);
        let hits = store.bm25("quokka", &Scope::All, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0.as_str(), "a.md");
        assert!(hits[0].1 > 0.0);
    }

    #[test]
    fn put_file_replaces_what_was_there() {
        let (_dir, c) = corpus();
        let mut store = Store::open_write(&c, &meta()).unwrap();
        let one = Chunk {
            byte_start: 0,
            byte_end: 5,
            line_start: 1,
            line_end: 1,
        };
        let two = [Chunk {
            byte_start: 0,
            byte_end: 4,
            line_start: 1,
            line_end: 1,
        }; 2];
        store
            .put_file(
                &rec("a.md", "h1"),
                "old text about otters",
                &[one],
                &[vec![0.1; 768]],
            )
            .unwrap();
        store
            .put_file(&rec("a.md", "h2"), "new text about newts", &two, &[])
            .unwrap();
        assert!(store.bm25("otters", &Scope::All, 10).unwrap().is_empty());
        assert_eq!(store.bm25("newts", &Scope::All, 10).unwrap().len(), 1);
        assert_eq!(store.vectors(&Scope::All).unwrap().chunks.len(), 0);
        assert_eq!(store.chunks_of(&rec("a.md", "h2").rel).unwrap().len(), 2);
    }

    #[test]
    fn remove_file_takes_everything_with_it() {
        let (_dir, c) = corpus();
        let mut store = Store::open_write(&c, &meta()).unwrap();
        let chunk = Chunk {
            byte_start: 0,
            byte_end: 5,
            line_start: 1,
            line_end: 1,
        };
        store
            .put_file(&rec("a.md", "h1"), "needle", &[chunk], &[vec![0.5; 768]])
            .unwrap();
        store.remove_file(&rec("a.md", "h1").rel).unwrap();
        assert!(store.files().unwrap().is_empty());
        assert!(store.bm25("needle", &Scope::All, 10).unwrap().is_empty());
        assert_eq!(store.stats().unwrap().chunks, 0);
    }

    #[test]
    fn a_different_signature_is_refused_by_name() {
        let (_dir, c) = corpus();
        let m = meta();
        drop(Store::open_write(&c, &m).unwrap());
        let mut other = m.clone();
        other.signature = "tampered-signature".to_string();
        let err = Store::open_write(&c, &other).unwrap_err().to_string();
        assert!(err.contains("different embedding signature"), "{err}");
        assert!(err.contains("--clean"), "{err}");
    }

    #[test]
    fn scope_limits_vectors_and_bm25() {
        let (_dir, c) = corpus();
        let mut store = Store::open_write(&c, &meta()).unwrap();
        let chunk = Chunk {
            byte_start: 0,
            byte_end: 5,
            line_start: 1,
            line_end: 1,
        };
        store
            .put_file(
                &rec("sub/needle.md", "h1"),
                "needle",
                &[chunk],
                &[vec![0.9; 768]],
            )
            .unwrap();
        store
            .put_file(&rec("top.md", "h2"), "needle", &[chunk], &[vec![0.1; 768]])
            .unwrap();
        let sub = Scope::Under(RelPath::new(std::path::Path::new("sub")).unwrap());
        let vs = store.vectors(&sub).unwrap();
        assert_eq!(vs.chunks.len(), 1);
        assert_eq!(vs.chunks[0].rel.as_str(), "sub/needle.md");
        let hits = store.bm25("needle", &sub, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0.as_str(), "sub/needle.md");
    }

    #[test]
    fn f16_round_trip() {
        let v = vec![0.0, 1.0, -0.5, 0.6875];
        let bytes = encode_vector(&v);
        let mut out = Vec::new();
        decode_vector(&bytes, &mut out);
        assert_eq!(out.len(), 4);
        assert!((out[3] - 0.6875).abs() < 1e-3);
    }
}
