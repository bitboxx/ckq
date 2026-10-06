//! All of an index's vectors in one file, so a semantic search reads one file
//! instead of one sidecar per indexed file.
//!
//! A search used to walk the index directory and decode every sidecar: about
//! 60 ms for a thousand files, most of a warm query. The cache holds the same
//! vectors packed together with the path and span of each chunk, raw little-endian
//! floats after a small header, and reads in a few milliseconds.
//!
//! It is only ever a copy. Validity is the manifest's identity (inode, mtime,
//! size): every index write ends by saving the manifest through an atomic
//! rename, so any change to the index changes that identity. The identity is
//! taken before the sidecars are read and the cache is saved only if it still
//! holds afterwards, so an index update racing a rebuild leaves no cache rather
//! than a wrong one. Anything unexpected on load is a miss, never an error.

use ck_core::Span;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

const FILE_NAME: &str = "vectors.cache";
const MAGIC: &[u8; 8] = b"CKQVEC01";

/// Every chunk of an index with its embedding, stored flat: one list of files,
/// one of chunks pointing into it, one buffer of floats. Loading it makes three
/// allocations rather than two per chunk, which was most of the load time.
#[derive(Debug, Clone, Default)]
pub struct VectorSet {
    dims: usize,
    pub files: Vec<PathBuf>,
    /// For each chunk, the index of its file in `files`, and its span.
    pub chunks: Vec<(u32, Span)>,
    data: Vec<f32>,
}

impl VectorSet {
    /// Add a chunk. A file's chunks must be added one after another. A vector
    /// whose length differs from the first one is skipped.
    pub fn push(&mut self, file: &Path, span: Span, embedding: &[f32]) {
        if self.chunks.is_empty() {
            self.dims = embedding.len();
        } else if embedding.len() != self.dims {
            return;
        }
        if self.files.last().map(PathBuf::as_path) != Some(file) {
            self.files.push(file.to_path_buf());
        }
        self.chunks.push(((self.files.len() - 1) as u32, span));
        self.data.extend_from_slice(embedding);
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// The embedding of chunk `i`.
    pub fn embedding(&self, i: usize) -> &[f32] {
        &self.data[i * self.dims..(i + 1) * self.dims]
    }
}

/// The identity of the manifest file when the cache was built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ManifestStamp {
    ino: u64,
    mtime_ns: u64,
    len: u64,
}

/// The manifest's current identity, or `None` when there is no manifest.
pub fn manifest_stamp(index_dir: &Path) -> Option<ManifestStamp> {
    let meta = fs::metadata(index_dir.join("manifest.json")).ok()?;
    let mtime_ns = meta
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_nanos() as u64;
    #[cfg(unix)]
    let ino = std::os::unix::fs::MetadataExt::ino(&meta);
    #[cfg(not(unix))]
    let ino = 0;
    Some(ManifestStamp {
        ino,
        mtime_ns,
        len: meta.len(),
    })
}

/// Drop the cache, for a change that might not touch the manifest.
pub fn invalidate(index_dir: &Path) {
    let _ = fs::remove_file(index_dir.join(FILE_NAME));
}

/// The cached vectors, or `None` when the cache is missing, stale or unreadable.
pub fn load(index_dir: &Path) -> Option<VectorSet> {
    let stamp = manifest_stamp(index_dir)?;
    let data = fs::read(index_dir.join(FILE_NAME)).ok()?;
    let mut r = Reader {
        data: &data,
        pos: 0,
    };
    if r.take(MAGIC.len())? != MAGIC {
        return None;
    }
    let stored = ManifestStamp {
        ino: r.u64()?,
        mtime_ns: r.u64()?,
        len: r.u64()?,
    };
    if stored != stamp {
        return None;
    }
    let dims = r.u32()? as usize;
    let n_files = r.u32()? as usize;
    let mut set = VectorSet {
        dims,
        ..Default::default()
    };
    for file in 0..n_files {
        let len = r.u32()? as usize;
        let path = std::str::from_utf8(r.take(len)?).ok()?;
        set.files.push(PathBuf::from(path));
        let n_chunks = r.u32()? as usize;
        for _ in 0..n_chunks {
            let span = Span {
                byte_start: r.u64()? as usize,
                byte_end: r.u64()? as usize,
                line_start: r.u64()? as usize,
                line_end: r.u64()? as usize,
            };
            set.chunks.push((file as u32, span));
        }
    }
    let floats = r.take(set.chunks.len().checked_mul(dims)?.checked_mul(4)?)?;
    set.data = floats
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    (r.pos == data.len()).then_some(set)
}

/// Write the cache for `set`, read while the manifest had identity `stamp`.
/// Does nothing if the manifest has changed since, or if a path is not valid
/// UTF-8.
pub fn save(index_dir: &Path, stamp: ManifestStamp, set: &VectorSet) -> std::io::Result<()> {
    if manifest_stamp(index_dir) != Some(stamp) {
        return Ok(());
    }
    let mut paths = Vec::with_capacity(set.files.len());
    for file in &set.files {
        let Some(path) = file.to_str() else {
            return Ok(());
        };
        paths.push(path);
    }

    let mut out = Vec::with_capacity(64 + set.chunks.len() * 36 + set.data.len() * 4);
    out.extend_from_slice(MAGIC);
    for v in [stamp.ino, stamp.mtime_ns, stamp.len] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&(set.dims as u32).to_le_bytes());
    out.extend_from_slice(&(paths.len() as u32).to_le_bytes());
    let mut next = 0;
    for (file, path) in paths.iter().enumerate() {
        let start = next;
        while next < set.chunks.len() && set.chunks[next].0 as usize == file {
            next += 1;
        }
        out.extend_from_slice(&(path.len() as u32).to_le_bytes());
        out.extend_from_slice(path.as_bytes());
        out.extend_from_slice(&((next - start) as u32).to_le_bytes());
        for (_, s) in &set.chunks[start..next] {
            for v in [s.byte_start, s.byte_end, s.line_start, s.line_end] {
                out.extend_from_slice(&(v as u64).to_le_bytes());
            }
        }
    }
    for x in &set.data {
        out.extend_from_slice(&x.to_le_bytes());
    }

    // Temp file and rename, so a reader never sees half a cache. No fsync: a
    // cache lost to a power cut is rebuilt on the next search.
    let mut tmp = tempfile::NamedTempFile::new_in(index_dir)?;
    tmp.write_all(&out)?;
    tmp.persist(index_dir.join(FILE_NAME))
        .map_err(|e| e.error)?;
    Ok(())
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let data: &'a [u8] = self.data;
        let slice = data.get(self.pos..end)?;
        self.pos = end;
        Some(slice)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn span(n: usize) -> Span {
        Span {
            byte_start: n,
            byte_end: n + 10,
            line_start: n + 1,
            line_end: n + 2,
        }
    }

    fn sample() -> VectorSet {
        let mut set = VectorSet::default();
        set.push(Path::new("/r/a.md"), span(0), &[0.5, -1.0, 2.25]);
        set.push(Path::new("/r/a.md"), span(40), &[1.0, 0.0, -0.125]);
        set.push(Path::new("/r/b.md"), span(7), &[3.0, 4.0, 5.0]);
        set.push(Path::new("/r/b.md"), span(9), &[1.0]); // wrong length, skipped
        set
    }

    #[test]
    fn round_trip() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("manifest.json"), "{}").unwrap();
        let stamp = manifest_stamp(dir.path()).unwrap();
        let original = sample();
        assert_eq!((original.len(), original.files.len()), (3, 2));
        save(dir.path(), stamp, &original).unwrap();
        let loaded = load(dir.path()).expect("cache hit");
        assert_eq!(loaded.files, original.files);
        assert_eq!(loaded.len(), 3);
        for i in 0..3 {
            let (fa, sa) = &loaded.chunks[i];
            let (fb, sb) = &original.chunks[i];
            assert_eq!(fa, fb);
            assert_eq!(
                (sa.byte_start, sa.byte_end, sa.line_start, sa.line_end),
                (sb.byte_start, sb.byte_end, sb.line_start, sb.line_end)
            );
            assert_eq!(loaded.embedding(i), original.embedding(i));
        }
    }

    #[test]
    fn a_manifest_change_is_a_miss() {
        let dir = TempDir::new().unwrap();
        let manifest = dir.path().join("manifest.json");
        fs::write(&manifest, "{}").unwrap();
        let stamp = manifest_stamp(dir.path()).unwrap();
        save(dir.path(), stamp, &sample()).unwrap();
        // An index write replaces the manifest through a rename.
        let tmp = dir.path().join("m.tmp");
        fs::write(&tmp, "{\"x\":1}").unwrap();
        fs::rename(&tmp, &manifest).unwrap();
        assert!(load(dir.path()).is_none());
    }

    #[test]
    fn no_save_when_the_manifest_moved_during_the_read() {
        let dir = TempDir::new().unwrap();
        let manifest = dir.path().join("manifest.json");
        fs::write(&manifest, "{}").unwrap();
        let stamp = manifest_stamp(dir.path()).unwrap();
        let tmp = dir.path().join("m.tmp");
        fs::write(&tmp, "{\"x\":1}").unwrap();
        fs::rename(&tmp, &manifest).unwrap();
        save(dir.path(), stamp, &sample()).unwrap();
        assert!(!dir.path().join(FILE_NAME).exists());
    }

    #[test]
    fn truncation_and_invalidation_are_misses() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("manifest.json"), "{}").unwrap();
        let stamp = manifest_stamp(dir.path()).unwrap();
        save(dir.path(), stamp, &sample()).unwrap();
        let path = dir.path().join(FILE_NAME);
        let data = fs::read(&path).unwrap();
        fs::write(&path, &data[..data.len() - 3]).unwrap();
        assert!(load(dir.path()).is_none());
        save(dir.path(), stamp, &sample()).unwrap();
        invalidate(dir.path());
        assert!(load(dir.path()).is_none());
    }
}
