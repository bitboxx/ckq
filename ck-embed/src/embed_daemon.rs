//! A shared embedding daemon, so one model serves every ckq on the machine.
//!
//! Each CLI invocation is its own process, and the in-process worker cache
//! cannot cross that boundary: two agents searching at once loaded the model
//! twice (470 MB against 235 MB) and ran slower than doing it serially, because
//! they contended for the GPU. So the first invocation starts a daemon and every
//! later one connects to it.
//!
//! Lifetime: the daemon exits after `IDLE_TIMEOUT` with no requests. `--serve`
//! pins it, because an MCP server is long-lived and its next request may be
//! hours away.

use anyhow::{Context, Result, anyhow};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const IDLE_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const CONNECT_ATTEMPTS: u32 = 100;
const CONNECT_WAIT: Duration = Duration::from_millis(100);

/// One socket per (model, file): a different model is a different daemon, and a
/// stale socket from another model must never be mistaken for a live one.
pub fn socket_path(model: &str, gguf: &str) -> PathBuf {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in model
        .bytes()
        .chain(b"::".iter().copied())
        .chain(gguf.bytes())
    {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    let dir = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(dir).join(format!("ckq-embed-{hash:016x}.sock"))
}

/// Try an existing daemon. `None` means nothing is listening, or what was
/// listening has gone and left the socket file behind.
pub fn try_connect(path: &PathBuf) -> Option<UnixStream> {
    UnixStream::connect(path).ok()
}

/// Ask a running daemon to embed. One JSON line out, one JSON line back.
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
pub fn spawn(model_alias: &str, path: &PathBuf) -> Result<UnixStream> {
    // A socket file with nothing behind it blocks bind(); the previous daemon
    // was killed rather than shut down.
    if path.exists() && try_connect(path).is_none() {
        let _ = std::fs::remove_file(path);
    }
    let exe = std::env::current_exe().context("locating the ckq binary")?;
    std::process::Command::new(exe)
        .arg("--embed-daemon")
        .arg(model_alias)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .context("spawning the embed daemon")?;

    for _ in 0..CONNECT_ATTEMPTS {
        if let Some(s) = try_connect(path) {
            return Ok(s);
        }
        std::thread::sleep(CONNECT_WAIT);
    }
    Err(anyhow!(
        "embed daemon did not start listening on {}",
        path.display()
    ))
}

/// Run as the daemon. Blocks until idle for `IDLE_TIMEOUT`, or forever if pinned.
pub fn serve<F>(path: PathBuf, mut embed: F) -> Result<()>
where
    F: FnMut(&[String]) -> Result<Vec<Vec<f32>>>,
{
    if path.exists() && try_connect(&path).is_none() {
        let _ = std::fs::remove_file(&path);
    }
    // Lost the race to another invocation starting at the same moment; it won.
    let listener = match UnixListener::bind(&path) {
        Ok(l) => l,
        Err(_) if try_connect(&path).is_some() => return Ok(()),
        Err(e) => return Err(e).context("binding the embed daemon socket"),
    };
    listener.set_nonblocking(true)?;

    let last = Arc::new(Mutex::new(Instant::now()));
    let pinned = Arc::new(AtomicBool::new(false));

    loop {
        match listener.accept() {
            Ok((mut stream, _)) => {
                stream.set_nonblocking(false)?;
                let reader = BufReader::new(stream.try_clone()?);
                for line in reader.lines() {
                    let Ok(line) = line else { break };
                    if line.trim().is_empty() {
                        continue;
                    }
                    *last.lock().unwrap() = Instant::now();
                    let reply = match serde_json::from_str::<serde_json::Value>(&line) {
                        Ok(v) => {
                            if v.get("pin").and_then(|p| p.as_bool()).unwrap_or(false) {
                                pinned.store(true, Ordering::SeqCst);
                            }
                            let texts: Vec<String> =
                                serde_json::from_value(v["texts"].clone()).unwrap_or_default();
                            match embed(&texts) {
                                Ok(e) => serde_json::json!({ "embeddings": e }),
                                Err(e) => serde_json::json!({ "error": e.to_string() }),
                            }
                        }
                        Err(e) => serde_json::json!({ "error": e.to_string() }),
                    };
                    if writeln!(stream, "{reply}").is_err() {
                        break;
                    }
                    let _ = stream.flush();
                    *last.lock().unwrap() = Instant::now();
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if !pinned.load(Ordering::SeqCst) && last.lock().unwrap().elapsed() > IDLE_TIMEOUT {
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
