//! The query daemon: one resident model that embeds search queries.
//!
//! Every `--sem` and `--hybrid` search needs its query embedded, and loading
//! the model costs about 0.4 s per process — the cost a wrapper that calls
//! ckq dozens of times a day pays on every call. So the first search starts a
//! daemon (this same executable re-run with `--query-daemon`) and every later
//! one connects to it.
//!
//! The daemon embeds queries and nothing else. The client sends the bare
//! query text and the daemon calls `embed_query` on its side, so the query
//! prompt is applied in `embed` and nowhere else; indexing never talks to the
//! daemon, it embeds documents in the process that runs `--index`. The old
//! engine's daemon embedded role-agnostically and its client prefixed too,
//! which is the double-prefix bug this split rules out.
//!
//! The socket key carries the executable's own stamp, so a rebuilt binary
//! never shares a daemon with the code it replaced. The daemon binds before
//! it loads the model: two clients starting at once cannot load it twice (the
//! bind loser exits before loading), and requests queue on the socket while
//! the winner loads.
//!
//! Lifetime: the daemon exits after [`idle_timeout`] with no request and
//! removes its socket. `CKQ_IN_PROCESS=1` skips the daemon entirely; if it
//! cannot be reached the search embeds in its own process, with a warning.

use crate::embed::{self, Embedder, LlamaEmbedder};
use crate::models::{self, ModelSpec};
use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, UNIX_EPOCH};

/// How long the daemon waits with no request before it exits: 15 minutes, or
/// `CKQ_DAEMON_IDLE_SECS`. The e2e tests set it low, because every test build
/// is a new executable and so a new daemon, and each one would otherwise hold
/// a model resident for a quarter of an hour after the run.
pub fn idle_timeout() -> Duration {
    std::env::var("CKQ_DAEMON_IDLE_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(900))
}

/// How long a connection may sit with no request before the daemon drops it:
/// a real client writes its query the moment it connects, and one that has
/// gone silent must not hold the daemon's only embedding thread.
const CONN_READ_TIMEOUT: Duration = Duration::from_secs(30);
/// How long the client that started a daemon waits for it to listen. The
/// daemon binds before it loads the model, so this covers a cold start and a
/// slow exec, not the load; the load is waited out on the request itself.
const STARTUP_WAIT: Duration = Duration::from_secs(30);
const CONNECT_POLL: Duration = Duration::from_millis(50);
const ACCEPT_POLL: Duration = Duration::from_millis(200);
/// How long a log with no socket beside it is kept: long enough to read why
/// a daemon failed to start, short enough that the directory does not grow.
const LOG_KEEP: Duration = Duration::from_secs(24 * 60 * 60);

/// The embedder a search uses: the daemon's when one can be reached — started
/// first if nothing answers — else this process's own model, with one warning
/// so the slowdown is never silent. `CKQ_IN_PROCESS=1` skips the daemon.
pub fn embedder(spec: &'static ModelSpec) -> Result<Box<dyn Embedder>> {
    if in_process() {
        return load_in_process(spec);
    }
    let path = socket_path(spec.alias);
    let stream = match try_connect(&path) {
        Some(stream) => stream,
        None => match start(&path, spec.alias) {
            Ok(stream) => stream,
            Err(e) => {
                eprintln!(
                    "ckq: warning: no query daemon ({e:#}); \
                     embedding the query in this process instead"
                );
                return load_in_process(spec);
            }
        },
    };
    Ok(Box::new(DaemonEmbedder { spec, stream }))
}

fn in_process() -> bool {
    std::env::var_os("CKQ_IN_PROCESS").is_some_and(|v| v == "1")
}

fn load_in_process(spec: &'static ModelSpec) -> Result<Box<dyn Embedder>> {
    Ok(Box::new(LlamaEmbedder::load(spec, embed::query_device())?))
}

/// `Embedder` over one daemon connection. Queries only: a document request is
/// a bug (indexing must load its own model), so it is an error rather than a
/// silent fallback.
struct DaemonEmbedder {
    spec: &'static ModelSpec,
    stream: UnixStream,
}

impl Embedder for DaemonEmbedder {
    fn dims(&self) -> usize {
        self.spec.dims
    }

    fn embed_documents(&mut self, _texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Err(anyhow!(
            "the query daemon embeds queries only; indexing loads its own model"
        ))
    }

    fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        request(&mut self.stream, text)
    }
}

/// One exchange: the bare query as a JSON line out, one JSON line back,
/// `{"vector": [...]}` or `{"error": "..."}`. No read timeout: the daemon
/// binds before it loads the model, so a first request may legitimately block
/// for the whole load. Daemon-side errors come back as errors, not as
/// silence.
fn request(stream: &mut UnixStream, query: &str) -> Result<Vec<f32>> {
    let payload = json!({ "query": query });
    writeln!(stream, "{payload}").context("writing to the query daemon")?;
    stream.flush()?;

    let mut reader = BufReader::new(stream.try_clone().context("sharing the daemon stream")?);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .context("reading from the query daemon")?;
    if line.trim().is_empty() {
        return Err(anyhow!("the query daemon closed the connection"));
    }
    let reply: Value = serde_json::from_str(&line)?;
    if let Some(err) = reply.get("error").and_then(Value::as_str) {
        return Err(anyhow!("query daemon: {err}"));
    }
    serde_json::from_value(reply["vector"].clone()).context("parsing the daemon's vector")
}

/// Where the sockets live: `~/.cache/ck/sockets`, beside the model cache, mode
/// 0700 — a daemon answers anyone who can connect, so no other user may reach
/// the directory.
fn socket_dir() -> PathBuf {
    let dir = dirs::home_dir()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
        .join(".cache")
        .join("ck")
        .join("sockets");
    if std::fs::create_dir_all(&dir).is_ok() {
        let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700));
    }
    dir
}

/// 16 hex of blake3 over the executable's path, size and mtime. A rebuild
/// changes it, so a client never talks to a daemon still running the code the
/// rebuild replaced — and a stale daemon never serves a new chunker or prompt
/// it predates.
fn exe_stamp() -> String {
    let mut hasher = blake3::Hasher::new();
    if let Ok(exe) = std::env::current_exe() {
        hasher.update(exe.to_string_lossy().as_bytes());
        if let Ok(meta) = std::fs::metadata(&exe) {
            hasher.update(&meta.len().to_le_bytes());
            if let Some(mtime) = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            {
                hasher.update(&mtime.as_nanos().to_le_bytes());
            }
        }
    }
    hasher.finalize().to_hex()[..16].to_string()
}

/// One socket per model and binary: `ckq2-<alias>-<stamp>.sock`.
pub fn socket_path(alias: &str) -> PathBuf {
    socket_dir().join(format!("ckq2-{alias}-{}.sock", exe_stamp()))
}

/// `Some(stream)` when a daemon answers on `path`.
fn try_connect(path: &Path) -> Option<UnixStream> {
    UnixStream::connect(path).ok()
}

/// A socket file with nothing behind it blocks bind(): the daemon before this
/// one was killed rather than left to idle out. Removed only when nothing
/// answers, so a live daemon's socket is never mistaken for dead.
fn remove_dead_socket(path: &Path) {
    if path.exists() && try_connect(path).is_none() {
        let _ = std::fs::remove_file(path);
    }
}

/// Where a daemon's stderr goes: beside its socket, one log per daemon, so
/// two models do not share. The client reads a failed start back out of it.
fn log_path(socket: &Path) -> PathBuf {
    let mut name = socket.as_os_str().to_os_string();
    name.push(".log");
    PathBuf::from(name)
}

/// The tail of a daemon log, for an error message. Empty when there is
/// nothing to say, so the caller can append it unconditionally.
fn last_log_lines(path: &Path) -> String {
    let Ok(text) = std::fs::read_to_string(path) else {
        return String::new();
    };
    let tail: Vec<&str> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .rev()
        .take(5)
        .collect();
    if tail.is_empty() {
        return String::new();
    }
    let body: Vec<&str> = tail.into_iter().rev().collect();
    format!("\n{}", body.join("\n"))
}

/// Start the daemon: re-exec this executable detached — its own process group,
/// stdin and stdout to null, stderr to the log beside the socket (inheriting
/// them would hold the caller's pipes open for the daemon's whole life) — and
/// wait up to [`STARTUP_WAIT`] for it to listen.
fn start(path: &Path, alias: &str) -> Result<UnixStream> {
    remove_dead_socket(path);
    let exe = std::env::current_exe().context("locating the ckq binary")?;
    let log = log_path(path);
    let log_file =
        std::fs::File::create(&log).with_context(|| format!("creating {}", log.display()))?;
    let mut child = {
        use std::os::unix::process::CommandExt;
        std::process::Command::new(exe)
            .arg("--query-daemon")
            .arg(alias)
            .process_group(0)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::from(log_file))
            .spawn()
            .context("spawning the query daemon")?
    };

    let deadline = Instant::now() + STARTUP_WAIT;
    loop {
        // A concurrent client's daemon may have won the bind race and had our
        // child exit quietly; connecting is the only fact that matters.
        if let Some(stream) = try_connect(path) {
            return Ok(stream);
        }
        if let Some(status) = child.try_wait().context("waiting on the query daemon")? {
            let detail = last_log_lines(&log);
            return Err(anyhow!(
                "the query daemon exited during startup: {status}{detail}"
            ));
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            let detail = last_log_lines(&log);
            return Err(anyhow!(
                "the query daemon did not listen within {} s{detail}",
                STARTUP_WAIT.as_secs()
            ));
        }
        std::thread::sleep(CONNECT_POLL);
    }
}

/// Run as the daemon (`ckq --query-daemon <alias>`). Binds the socket before
/// the model is loaded — that is what decides the race between two starting
/// daemons, and the loser exits here without loading anything — then serves
/// one query per connection until [`idle_timeout`] passes with no request.
/// An idle exit is a clean one: the socket goes, and the log with it.
pub fn serve(alias: &str) -> Result<()> {
    let spec = models::resolve(alias)?;
    let path = socket_path(spec.alias);
    // A write to a client that has already gone must not kill the daemon.
    unsafe { libc::signal(libc::SIGPIPE, libc::SIG_IGN) };
    let Some(listener) = bind(&path)? else {
        return Ok(()); // another daemon won the startup race and is serving
    };
    eprintln!(
        "ckq: query daemon {} pid {} on {}",
        spec.alias,
        std::process::id(),
        path.display()
    );

    // Loading after the bind is what makes one model out of two concurrent
    // starts; the requests that arrive while it loads queue on the socket.
    let mut embedder = match LlamaEmbedder::load(spec, embed::query_device()) {
        Ok(embedder) => embedder,
        Err(e) => {
            let _ = std::fs::remove_file(&path);
            return Err(e);
        }
    };

    listener.set_nonblocking(true)?;
    let mut last = Instant::now();
    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                let _ = serve_conn(stream, &mut embedder); // one client's failure is not the daemon's
                last = Instant::now();
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if last.elapsed() > idle_timeout() {
                    break;
                }
                std::thread::sleep(ACCEPT_POLL);
            }
            Err(e) => return Err(e).context("accepting on the query daemon socket"),
        }
    }
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(log_path(&path));
    Ok(())
}

/// Claim the socket without loading anything. `Ok(None)` means another daemon
/// is already listening and won the startup race.
fn bind(path: &Path) -> Result<Option<UnixListener>> {
    remove_dead_socket(path);
    match UnixListener::bind(path) {
        Ok(listener) => {
            reap_stale(path);
            Ok(Some(listener))
        }
        Err(_) if try_connect(path).is_some() => Ok(None),
        Err(e) => Err(e).context("binding the query daemon socket"),
    }
}

/// One connection: read one query line, embed it, write one reply line. The
/// protocol is one JSON line each way, so the connection is dropped after the
/// exchange; the read timeout drops a client that connects and says nothing.
fn serve_conn(stream: UnixStream, embedder: &mut LlamaEmbedder) -> std::io::Result<()> {
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(CONN_READ_TIMEOUT))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = stream;
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(()); // connected and left without asking
    }
    let reply = match serde_json::from_str::<Value>(line.trim()) {
        Ok(v) => match v.get("query").and_then(Value::as_str) {
            Some(query) => match embedder.embed_query(query) {
                Ok(vector) => json!({ "vector": vector }),
                Err(e) => json!({ "error": e.to_string() }),
            },
            None => json!({ "error": "malformed request: no query" }),
        },
        Err(e) => json!({ "error": format!("malformed request: {e}") }),
    };
    writeln!(writer, "{reply}")?;
    writer.flush()
}

/// Clear what dead daemons left in the socket directory. The socket key
/// carries the binary's stamp, so every rebuild leaves a new socket and log
/// pair, and [`bind`] only ever clears a dead socket at its own key; without
/// this the directory grows by a pair per build. A `ckq2-*` socket goes when
/// nothing answers on it, its log with it; a log alone goes once it is older
/// than [`LOG_KEEP`]. Nothing outside `ckq2-*` is ever touched.
fn reap_stale(own: &Path) {
    let Some(dir) = own.parent() else { return };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path == own || path == log_path(own) {
            continue;
        }
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with("ckq2-") {
            continue;
        }
        if name.ends_with(".sock") {
            if try_connect(&path).is_none() {
                let _ = std::fs::remove_file(&path);
                let _ = std::fs::remove_file(log_path(&path));
            }
        } else if let Some(socket) = name.strip_suffix(".log") {
            let orphan = !dir.join(format!("{socket}.sock")).exists();
            let old = entry
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.elapsed().ok())
                .is_some_and(|age| age > LOG_KEEP);
            if orphan && old {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
}
