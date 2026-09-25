//! A shared embedding daemon, so one model serves every ckq on the machine.
//!
//! Each CLI invocation is its own process, and the in-process worker cache
//! cannot cross that boundary: two agents searching at once loaded the model
//! twice (470 MB against 235 MB) and ran slower than doing it serially, because
//! they contended for the GPU. So the first invocation starts a daemon and every
//! later one connects to it.
//!
//! Lifetime: the daemon exits after `IDLE_TIMEOUT` with no requests. A client
//! can pin it with `"pin": true`; the pin lasts exactly as long as that client's
//! connection stays open, so a client that dies or moves on cannot leave the
//! machine with an immortal daemon.

use anyhow::{Context, Result, anyhow};
use parking_lot::Mutex;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::MetadataExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant, UNIX_EPOCH};

pub const IDLE_TIMEOUT: Duration = Duration::from_secs(15 * 60);
/// How long a connection may sit with no request before the daemon drops it.
/// Real clients write their request immediately; the timeout exists so a client
/// that connects and then goes silent cannot hold its thread (and its pin)
/// forever.
const CONN_READ_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const CONNECT_POLL: Duration = Duration::from_millis(50);

/// Identity of the running executable, folded into the socket key: a rebuild
/// can change pooling or truncation behaviour, and clients must not be served
/// by a stale daemon still running the old code.
fn exe_stamp() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|exe| {
            let meta = std::fs::metadata(&exe).ok()?;
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            Some(format!("{}:{}:{}", exe.display(), meta.len(), mtime))
        })
        // Unlikely (deleted binary, procfs unavailable); the version alone still
        // separates releases, just not same-version rebuilds.
        .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string())
}

/// Where sockets live. /tmp is world-writable: with a deterministic name,
/// another local user could bind the socket first and quietly receive
/// everything the machine indexes. XDG_RUNTIME_DIR is per-user by spec;
/// otherwise fall back to a 0700 directory under the user's own cache.
fn socket_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("XDG_RUNTIME_DIR")
        && !dir.is_empty()
    {
        return PathBuf::from(dir);
    }
    let cache = match std::env::var_os("XDG_CACHE_HOME") {
        Some(v) if !v.is_empty() => PathBuf::from(v),
        _ => match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home).join(".cache"),
            None => PathBuf::from(".cache"),
        },
    };
    let dir = cache.join("ck").join("sockets");
    if std::fs::create_dir_all(&dir).is_ok() {
        use std::os::unix::fs::PermissionsExt;
        // Only the first creation matters; a failed chmod still leaves the dir
        // owned by this user.
        let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700));
    }
    dir
}

/// One socket per (model, file, binary): a different model is a different
/// daemon, a stale socket from another model must never be mistaken for a
/// live one, and a rebuild must not be served by the old code.
pub fn socket_path(model: &str, gguf: &str) -> PathBuf {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in model
        .bytes()
        .chain(b"::".iter().copied())
        .chain(gguf.bytes())
        .chain(exe_stamp().bytes())
    {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    socket_dir().join(format!("ckq-embed-{hash:016x}.sock"))
}

/// Try an existing daemon. `None` means nothing is listening, what was
/// listening has gone and left the socket file behind, or the socket belongs
/// to another user — a deterministic name in a directory other users can write
/// to would otherwise hand them everything we index.
pub fn try_connect(path: &Path) -> Option<UnixStream> {
    if let Ok(meta) = std::fs::metadata(path)
        && meta.uid() != unsafe { libc::geteuid() }
    {
        return None;
    }
    UnixStream::connect(path).ok()
}

/// Claim the socket without loading anything. `Ok(None)` means another daemon
/// is already listening and won the startup race.
///
/// Binding before loading is deliberate: two processes starting together must
/// not both load the model — the very double-load this daemon exists to
/// prevent. The loser returns from here immediately and becomes a client of
/// the winner, and callers' requests queue on the socket while the winner
/// loads.
pub fn bind(path: &Path) -> Result<Option<UnixListener>> {
    // A socket file with nothing behind it blocks bind(); the previous daemon
    // was killed rather than shut down.
    if path.exists() && try_connect(path).is_none() {
        let _ = std::fs::remove_file(path);
    }
    match UnixListener::bind(path) {
        Ok(l) => Ok(Some(l)),
        // Lost the race to another invocation starting at the same moment; it won.
        Err(_) if try_connect(path).is_some() => Ok(None),
        Err(e) => Err(e).context("binding the embed daemon socket"),
    }
}

/// Ask a running daemon to embed. One JSON line out, one JSON line back.
/// There is deliberately no read timeout: the daemon binds before it loads
/// the model, so the first request may legitimately block for the whole load.
pub fn request(stream: &mut UnixStream, texts: &[String], pin: bool) -> Result<Vec<Vec<f32>>> {
    let payload = serde_json::json!({ "texts": texts, "pin": pin });
    writeln!(stream, "{payload}").context("writing to embed daemon")?;
    stream.flush()?;

    let mut line = String::new();
    BufReader::new(stream.try_clone()?)
        .read_line(&mut line)
        .context("reading from embed daemon")?;
    if line.trim().is_empty() {
        return Err(anyhow!("embed daemon closed the connection"));
    }
    let v: serde_json::Value = serde_json::from_str(&line)?;
    if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
        return Err(anyhow!("embed daemon: {err}"));
    }
    serde_json::from_value(v["embeddings"].clone()).context("decoding embeddings")
}

/// Re-exec ourselves as a detached daemon and wait for it to listen.
pub fn spawn(model_alias: &str, path: &Path) -> Result<UnixStream> {
    // A socket file with nothing behind it blocks bind(); the previous daemon
    // was killed rather than shut down.
    if path.exists() && try_connect(path).is_none() {
        let _ = std::fs::remove_file(path);
    }
    let exe = std::env::current_exe().context("locating the ckq binary")?;
    let mut child = std::process::Command::new(exe)
        .arg("--embed-daemon")
        .arg(model_alias)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        // stderr stays inherited: a failed start (bad alias, failed download,
        // model load error) must be visible in the terminal that triggered it,
        // not vanish into null and surface as a bare timeout.
        .spawn()
        .context("spawning the embed daemon")?;

    // The daemon binds its socket before it loads the model, so connecting is
    // fast and the model-load wait happens later, on the request itself. A
    // fixed attempt budget here used to give up while a first-use download of
    // hundreds of MB was still running, orphaning it; instead wait exactly as
    // long as the child is alive to try.
    loop {
        if let Some(stream) = try_connect(path) {
            return Ok(stream);
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                return Err(anyhow!("embed daemon exited during startup: {status}"));
            }
            Ok(None) => std::thread::sleep(CONNECT_POLL),
            Err(e) => return Err(anyhow!("waiting on the embed daemon: {e}")),
        }
    }
}

/// Run as the daemon on an already-bound listener. Exits after `IDLE_TIMEOUT`
/// with no requests and no pinned connection left.
///
/// Each connection runs on its own thread, so one stuck client cannot stall
/// every other ckq on the machine; the model itself is not thread-safe, so
/// generation is serialised on a mutex around `embed` instead of on the
/// accept loop.
/// Hold a connection open purely to pin the daemon, for the life of this process.
///
/// The pin counts live pinned connections, and an ordinary client opens one per
/// request and drops it, so a per-request pin lasts exactly one request. An MCP
/// server may go hours between queries and must not pay a reload, so it takes a
/// connection of its own and never closes it.
pub fn pin_for_process(stream: UnixStream) {
    // Deliberately never dropped: closing it would release the pin.
    std::mem::forget(stream);
}

pub fn serve_on<F>(listener: UnixListener, path: PathBuf, embed: F) -> Result<()>
where
    F: FnMut(&[String]) -> Result<Vec<Vec<f32>>> + Send + 'static,
{
    listener.set_nonblocking(true)?;

    let embed = Arc::new(Mutex::new(embed));
    let last = Arc::new(Mutex::new(Instant::now()));
    // Pin is a count of live pinned connections, not a latch: a one-bit "ever
    // pinned" flag kept the daemon alive long after its `--serve` client died.
    let pinned = Arc::new(AtomicUsize::new(0));

    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                let embed = Arc::clone(&embed);
                let last = Arc::clone(&last);
                let pinned = Arc::clone(&pinned);
                // Accepting must never block on serving: hand the stream to a
                // thread and go back to accept.
                if thread::Builder::new()
                    .name("ckq-embed-conn".into())
                    .spawn(move || handle_conn(stream, &embed, &last, &pinned))
                    .is_err()
                {
                    // Out of threads; drop this one client rather than the daemon.
                    eprintln!("ckq: warning: could not spawn a thread for an embed client");
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if pinned.load(Ordering::SeqCst) == 0 && last.lock().elapsed() > IDLE_TIMEOUT {
                    break;
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(e) => return Err(e).context("accepting on the embed daemon socket"),
        }
    }
    let _ = std::fs::remove_file(&path);
    Ok(())
}

/// Serve every request line on one connection until it closes or times out.
fn handle_conn<F>(stream: UnixStream, embed: &Mutex<F>, last: &Mutex<Instant>, pinned: &AtomicUsize)
where
    F: FnMut(&[String]) -> Result<Vec<Vec<f32>>>,
{
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(CONN_READ_TIMEOUT));
    let mut writer = match stream.try_clone() {
        Ok(w) => w,
        Err(_) => return,
    };
    let reader = BufReader::new(stream);

    let mut conn_pinned = false;
    for line in reader.lines() {
        // Err covers EOF, the read timeout and a reset connection alike.
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        *last.lock() = Instant::now();
        let reply = match serde_json::from_str::<serde_json::Value>(&line) {
            Ok(v) => {
                if v.get("pin").and_then(|p| p.as_bool()).unwrap_or(false) && !conn_pinned {
                    conn_pinned = true;
                    pinned.fetch_add(1, Ordering::SeqCst);
                }
                let texts: Vec<String> =
                    serde_json::from_value(v["texts"].clone()).unwrap_or_default();
                let mut embed = embed.lock();
                match embed(&texts) {
                    Ok(e) => serde_json::json!({ "embeddings": e }),
                    Err(e) => serde_json::json!({ "error": e.to_string() }),
                }
            }
            Err(e) => serde_json::json!({ "error": e.to_string() }),
        };
        if writeln!(writer, "{reply}").is_err() {
            break;
        }
        let _ = writer.flush();
        *last.lock() = Instant::now();
    }
    if conn_pinned {
        pinned.fetch_sub(1, Ordering::SeqCst);
    }
}
