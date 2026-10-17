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
use anyhow::Result;
use std::collections::HashMap;

pub const DB_FILE: &str = "index.db";
pub const SCHEMA_VERSION: i64 = 1;

/// The schema. `fts` is contentless: it holds the BM25 index but not the text,
/// which stays in the files. Its rowid is `files.id`. Columns are `path` then
/// `content`, and ranking is `bm25(fts, 0.5, 1.0)`: the path counts half, as in
/// the old engine. The tokenizer keeps diacritics, as tantivy's default did, so
/// `cafe` does not match `café` and whole words still do not match compounds.
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
    pub db_bytes: u64,
}

pub struct Store {
    conn: rusqlite::Connection,
}

impl Store {
    /// Open an existing index read-only. `None` when the corpus has none.
    pub fn open_read(corpus: &Corpus) -> Result<Option<Store>> {
        todo!()
    }

    /// Open for writing, creating the file and schema when absent. A new index
    /// takes `meta`; an existing one must already carry the same signature,
    /// otherwise this fails with a message that names `--clean`.
    pub fn open_write(corpus: &Corpus, meta: &IndexMeta) -> Result<Store> {
        todo!("busy_timeout 30 s; foreign_keys on")
    }

    pub fn meta(&self) -> Result<IndexMeta> {
        todo!()
    }

    pub fn files(&self) -> Result<HashMap<RelPath, FileRecord>> {
        todo!()
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
        todo!()
    }

    /// Drop a file and everything stored for it, in one transaction.
    pub fn remove_file(&mut self, rel: &RelPath) -> Result<()> {
        todo!()
    }

    /// All chunk vectors whose file is in `scope`.
    pub fn vectors(&self, scope: &Scope) -> Result<VectorSet> {
        todo!()
    }

    /// The chunks of one file, in order.
    pub fn chunks_of(&self, rel: &RelPath) -> Result<Vec<Chunk>> {
        todo!()
    }

    /// BM25 over files. `match_expr` is an FTS5 MATCH expression built by
    /// `search::fts_query`. Best first, at most `limit`, files in `scope` only.
    /// Scores are positive, higher is better (the negation of FTS5's `bm25()`).
    pub fn bm25(&self, match_expr: &str, scope: &Scope, limit: usize) -> Result<Vec<(RelPath, f32)>> {
        todo!()
    }

    pub fn stats(&self) -> Result<IndexStats> {
        todo!()
    }
}

/// f32 to little-endian f16 bytes, and back.
pub fn encode_vector(v: &[f32]) -> Vec<u8> {
    todo!("half::f16")
}

pub fn decode_vector(bytes: &[u8], out: &mut Vec<f32>) {
    todo!()
}
