//! Offline integration tests: they run the real binary on real files and need
//! no model and no network. Anything that has to index (and so embed) carries
//! an `e2e:` ignore and runs with `cargo e2e`.

use serial_test::serial;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;

fn ckq_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_ckq"))
}

/// Spawn ckq with the settings that change its behaviour cleared, so these
/// tests exercise the defaults regardless of the ambient environment.
fn ckq_command() -> Command {
    let mut cmd = Command::new(ckq_binary());
    for var in ["CKQ_GGUF_FILE", "CKQ_TIMING", "CKQ_QUERY_DEVICE"] {
        cmd.env_remove(var);
    }
    cmd
}

#[test]
fn test_basic_grep_functionality() {
    let temp_dir = TempDir::new().unwrap();

    fs::write(
        temp_dir.path().join("file1.txt"),
        "hello world\nrust programming\ntest line",
    )
    .unwrap();
    fs::write(
        temp_dir.path().join("file2.rs"),
        "fn main() {\n    println!(\"Hello Rust\");\n}",
    )
    .unwrap();
    fs::write(
        temp_dir.path().join("file3.py"),
        "print('Hello Python')\n# rust comment",
    )
    .unwrap();

    let output = ckq_command()
        .args(["rust", temp_dir.path().to_str().unwrap()])
        .output()
        .expect("Failed to run ckq");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("rust programming"));
    assert!(stdout.contains("# rust comment"));
}

#[test]
fn test_case_insensitive_search() {
    let temp_dir = TempDir::new().unwrap();
    fs::write(
        temp_dir.path().join("test.txt"),
        "Hello World\nHELLO WORLD\nhello world",
    )
    .unwrap();

    let output = ckq_command()
        .args(["-i", "HELLO", temp_dir.path().to_str().unwrap()])
        .output()
        .expect("Failed to run ckq");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let line_count = stdout.lines().count();
    assert_eq!(line_count, 6); // filename + content for each of three matches
}

#[test]
fn test_recursive_search() {
    let temp_dir = TempDir::new().unwrap();
    fs::create_dir(temp_dir.path().join("subdir")).unwrap();
    fs::write(temp_dir.path().join("root.txt"), "target text").unwrap();
    fs::write(
        temp_dir.path().join("subdir").join("nested.txt"),
        "target text",
    )
    .unwrap();

    let output = ckq_command()
        .args(["-r", "target", temp_dir.path().to_str().unwrap()])
        .output()
        .expect("Failed to run ckq");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let line_count = stdout.lines().count();
    assert_eq!(line_count, 4); // two files, filename + content each
}

#[test]
fn test_json_output() {
    let temp_dir = TempDir::new().unwrap();
    fs::write(temp_dir.path().join("test.txt"), "json test line").unwrap();

    let output = ckq_command()
        .args(["--json", "json", temp_dir.path().to_str().unwrap()])
        .output()
        .expect("Failed to run ckq");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();

    // Should be valid JSON
    let json_result: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert!(json_result["file"].is_string());
    assert!(json_result["score"].is_number());
    assert!(json_result["preview"].is_string());
}

#[test]
fn test_context_lines() {
    let temp_dir = TempDir::new().unwrap();
    fs::write(
        temp_dir.path().join("context.txt"),
        "line 1\nline 2\ntarget line\nline 4\nline 5",
    )
    .unwrap();

    let output = ckq_command()
        .args(["-C", "1", "target", temp_dir.path().to_str().unwrap()])
        .output()
        .expect("Failed to run ckq with context");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();

    // Should include context lines
    assert!(stdout.contains("line 2"));
    assert!(stdout.contains("target line"));
    assert!(stdout.contains("line 4"));
}

#[test]
fn test_topk_limit() {
    let temp_dir = TempDir::new().unwrap();

    for i in 1..=10 {
        fs::write(
            temp_dir.path().join(format!("file{i}.txt")),
            "match content",
        )
        .unwrap();
    }

    let output = ckq_command()
        .args(["--topk", "5", "match", temp_dir.path().to_str().unwrap()])
        .output()
        .expect("Failed to run ckq with topk");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let line_count = stdout.trim().lines().count();
    assert!(line_count <= 10); // up to 5 results, filename + content each
}

#[test]
fn test_line_numbers() {
    let temp_dir = TempDir::new().unwrap();
    fs::write(
        temp_dir.path().join("numbered.txt"),
        "line 1\nmatched line\nline 3",
    )
    .unwrap();

    let output = ckq_command()
        .args(["-n", "matched", temp_dir.path().to_str().unwrap()])
        .output()
        .expect("Failed to run ckq with line numbers");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("2:matched line"));
}

#[test]
fn test_no_matches_stderr_message() {
    let temp_dir = TempDir::new().unwrap();
    fs::write(temp_dir.path().join("test.txt"), "hello world").unwrap();

    let output = ckq_command()
        .args(["nonexistent_pattern", temp_dir.path().to_str().unwrap()])
        .output()
        .expect("Failed to run ckq");

    assert!(!output.status.success());
    assert_eq!(output.status.code(), Some(1));

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.trim().is_empty());

    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("No matches found"));
}

#[test]
fn test_nonexistent_directory_error() {
    let output = ckq_command()
        .args(["--sem", "test", "/nonexistent/directory"])
        .output()
        .expect("Failed to run ckq");

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("Path does not exist"));
    assert!(stderr.contains("/nonexistent/directory"));
}

#[test]
fn test_error_handling() {
    // An invalid regex must fail gracefully, not crash.
    let temp_dir = TempDir::new().unwrap();
    fs::write(temp_dir.path().join("test.txt"), "test content").unwrap();

    let output = ckq_command()
        .args(["[invalid", temp_dir.path().to_str().unwrap()])
        .output()
        .expect("Failed to run ckq");

    assert!(!output.status.success());
}

#[test]
fn test_invalid_regex_warning_during_highlighting() {
    let temp_dir = TempDir::new().unwrap();
    fs::write(
        temp_dir.path().join("test.txt"),
        "hello world\ntest content",
    )
    .unwrap();

    let output = ckq_command()
        .args(["[invalid", temp_dir.path().to_str().unwrap()])
        .output()
        .expect("Failed to run ckq");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("regex") || stderr.contains("pattern") || stderr.contains("invalid"));
}

#[test]
fn test_jsonl_basic_output() {
    let temp_dir = TempDir::new().unwrap();

    fs::write(
        temp_dir.path().join("test.rs"),
        "fn main() {\n    println!(\"Hello Rust\");\n}",
    )
    .unwrap();
    fs::write(
        temp_dir.path().join("test.py"),
        "print('Hello Python')\ndef main():\n    pass",
    )
    .unwrap();

    let output = ckq_command()
        .args(["fn main", "--jsonl", temp_dir.path().to_str().unwrap()])
        .output()
        .expect("Failed to run ckq");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(!stdout.trim().is_empty());

    for line in stdout.lines() {
        if !line.trim().is_empty() {
            let json: serde_json::Value = serde_json::from_str(line).expect("Invalid JSON line");
            assert!(json.get("path").is_some());
            assert!(json.get("span").is_some());
            assert!(json.get("language").is_some());
            assert!(json.get("snippet").is_some());
            assert!(json.get("score").is_some());

            let span = json.get("span").unwrap().as_object().unwrap();
            assert!(span.get("byte_start").is_some());
            assert!(span.get("byte_end").is_some());
            assert!(span.get("line_start").is_some());
            assert!(span.get("line_end").is_some());
        }
    }
}

#[test]
fn test_jsonl_no_snippet_flag() {
    let temp_dir = TempDir::new().unwrap();
    fs::write(
        temp_dir.path().join("test.rs"),
        "fn main() {\n    println!(\"Hello Rust\");\n}",
    )
    .unwrap();

    let output = ckq_command()
        .args([
            "fn main",
            "--jsonl",
            "--no-snippet",
            temp_dir.path().to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run ckq");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();

    for line in stdout.lines() {
        if !line.trim().is_empty() {
            let json: serde_json::Value = serde_json::from_str(line).expect("Invalid JSON line");
            assert!(json.get("snippet").is_none());
            assert!(json.get("path").is_some());
            assert!(json.get("span").is_some());
            assert!(json.get("language").is_some());
            assert!(json.get("score").is_some());
        }
    }
}

#[test]
fn test_jsonl_vs_regular_output() {
    let temp_dir = TempDir::new().unwrap();

    fs::write(
        temp_dir.path().join("test.rs"),
        "fn main() {\n    println!(\"Hello Rust\");\n}",
    )
    .unwrap();

    let regular_output = ckq_command()
        .args(["fn main", temp_dir.path().to_str().unwrap()])
        .output()
        .expect("Failed to run ckq");

    let jsonl_output = ckq_command()
        .args(["fn main", "--jsonl", temp_dir.path().to_str().unwrap()])
        .output()
        .expect("Failed to run ckq");

    assert!(regular_output.status.success());
    assert!(jsonl_output.status.success());

    let regular_stdout = String::from_utf8(regular_output.stdout).unwrap();
    let jsonl_stdout = String::from_utf8(jsonl_output.stdout).unwrap();

    assert!(!regular_stdout.contains("{\"path\":"));
    assert!(jsonl_stdout.contains("{\"path\":"));
    assert!(jsonl_stdout.contains("\"span\":"));
    assert!(jsonl_stdout.contains("\"language\":"));
}

#[test]
fn test_jsonl_with_different_languages() {
    let temp_dir = TempDir::new().unwrap();

    fs::write(
        temp_dir.path().join("test.rs"),
        "fn main() {\n    println!(\"Hello Rust\");\n}",
    )
    .unwrap();
    fs::write(
        temp_dir.path().join("test.py"),
        "print('Hello Python')\ndef main():\n    pass",
    )
    .unwrap();

    let output = ckq_command()
        .args(["main", "--jsonl", temp_dir.path().to_str().unwrap()])
        .output()
        .expect("Failed to run ckq");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();

    let languages: Vec<String> = stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let json: serde_json::Value = serde_json::from_str(l).expect("valid JSONL");
            json["language"]
                .as_str()
                .expect("language field")
                .to_string()
        })
        .collect();
    assert!(
        languages.contains(&"rust".to_string()),
        "the Rust file must be labelled: {languages:?}"
    );
    assert!(
        languages.contains(&"python".to_string()),
        "the Python file must be labelled: {languages:?}"
    );
}

#[test]
#[serial]
fn test_no_ckignore_flag_disables_hierarchical_ignore() {
    let temp_dir = TempDir::new().unwrap();
    let parent = temp_dir.path();
    let subdir = parent.join("subdir");
    fs::create_dir(&subdir).unwrap();

    // .ckqignore at parent level excluding *.tmp files
    fs::write(parent.join(".ckqignore"), "*.tmp\n").unwrap();

    fs::write(parent.join("test.txt"), "FINDME_TEXT").unwrap();
    fs::write(parent.join("ignored.tmp"), "FINDME_TMP").unwrap();
    fs::write(subdir.join("nested.txt"), "FINDME_TEXT").unwrap();
    fs::write(subdir.join("also_ignored.tmp"), "FINDME_TMP").unwrap();

    // With --no-ckqignore the .tmp files are included
    let output = ckq_command()
        .args(["-r", "--no-ckqignore", "FINDME", "."])
        .current_dir(parent)
        .output()
        .expect("Failed to run ckq search --no-ckqignore");

    assert!(
        output.status.success(),
        "Search with --no-ckqignore failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("ignored.tmp") || stdout.contains("also_ignored.tmp"),
        "Should find .tmp files when --no-ckqignore is used. Output: {stdout}"
    );

    // Without it (default), .ckqignore applies
    let output = ckq_command()
        .args(["-r", "FINDME", "."])
        .current_dir(parent)
        .output()
        .expect("Failed to run ckq search");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        !stdout.contains("ignored.tmp") && !stdout.contains("also_ignored.tmp"),
        "Should NOT find .tmp files when .ckqignore is active (default). Output: {stdout}"
    );
    assert!(
        stdout.contains("test.txt") || stdout.contains("nested.txt"),
        "Should find .txt files even with .ckqignore active"
    );
}

/// Writing to a closed pipe (`ck pattern | head`) must terminate ck silently
/// via SIGPIPE (exit status 141), matching grep — not panic with a
/// "failed printing to stdout: Broken pipe" message.
#[cfg(unix)]
#[test]
fn test_sigpipe_terminates_silently() {
    use std::io::Read;
    use std::os::unix::process::ExitStatusExt;
    use std::process::Stdio;

    let temp_dir = TempDir::new().unwrap();

    let line = "match: the quick brown fox jumps over the lazy dog\n";
    fs::write(temp_dir.path().join("big.txt"), line.repeat(50_000)).unwrap();

    let mut child = ckq_command()
        .args(["match", temp_dir.path().to_str().unwrap()])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn ckq");

    let mut stdout = child.stdout.take().unwrap();
    let mut buf = [0u8; 4096];
    let _ = stdout.read(&mut buf).unwrap();
    drop(stdout);

    let status = child.wait().expect("Failed to wait on ckq");
    let mut stderr_out = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr_out)
        .unwrap();

    assert!(
        !stderr_out.contains("panicked"),
        "ck panicked on broken pipe: {stderr_out}"
    );
    assert_eq!(
        status.signal(),
        Some(libc::SIGPIPE),
        "ck should be terminated by SIGPIPE, got status {status:?}"
    );
}

/// Helper: create a workspace with one visible file, one hidden dot-file, and
/// a hidden directory containing a file. All three share the search term "needle".
fn setup_hidden_workspace() -> TempDir {
    let temp_dir = TempDir::new().unwrap();
    fs::write(
        temp_dir.path().join("visible.txt"),
        "needle in visible file",
    )
    .unwrap();
    fs::write(
        temp_dir.path().join(".hidden-file.txt"),
        "needle in hidden file",
    )
    .unwrap();
    let hidden_dir = temp_dir.path().join(".hidden-dir");
    fs::create_dir(&hidden_dir).unwrap();
    fs::write(hidden_dir.join("inside.txt"), "needle inside hidden dir").unwrap();
    temp_dir
}

/// Regex search (the on-the-fly search path) must skip dot-prefixed files and
/// directories by default, and include them only when `--hidden` is passed.
#[test]
#[serial]
fn test_hidden_flag_regex_search() {
    let temp_dir = setup_hidden_workspace();

    let output = ckq_command()
        .args(["needle", "."])
        .current_dir(temp_dir.path())
        .output()
        .expect("Failed to run ckq search");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("visible.txt"),
        "visible file should always match; stdout: {stdout}"
    );
    assert!(
        !stdout.contains(".hidden-file.txt"),
        "hidden file should NOT match without --hidden; stdout: {stdout}"
    );
    assert!(
        !stdout.contains("inside.txt"),
        "file in hidden dir should NOT match without --hidden; stdout: {stdout}"
    );

    let output = ckq_command()
        .args(["--hidden", "needle", "."])
        .current_dir(temp_dir.path())
        .output()
        .expect("Failed to run ckq --hidden search");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("visible.txt"),
        "visible file should still match with --hidden; stdout: {stdout}"
    );
    assert!(
        stdout.contains(".hidden-file.txt"),
        "hidden file SHOULD match with --hidden; stdout: {stdout}"
    );
    assert!(
        stdout.contains("inside.txt"),
        "file in hidden dir SHOULD match with --hidden; stdout: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Everything below needs the index, and indexing embeds, so these run with
// `cargo e2e` and the default model.
// ---------------------------------------------------------------------------

#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn test_index_command() {
    let temp_dir = TempDir::new().unwrap();
    fs::write(temp_dir.path().join("test.txt"), "indexable content").unwrap();

    let output = ckq_command()
        .args(["--index", "."])
        .current_dir(temp_dir.path())
        .output()
        .expect("Failed to run ckq index");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stdout.contains("indexed") || stderr.contains("indexed"),
        "the run should say what it did; stdout: {stdout} stderr: {stderr}"
    );

    assert!(temp_dir.path().join(".ckq").exists());
}

#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn test_clean_command() {
    let temp_dir = TempDir::new().unwrap();
    fs::write(temp_dir.path().join("test.txt"), "test content").unwrap();

    let output = ckq_command()
        .args(["--index", "."])
        .current_dir(temp_dir.path())
        .output()
        .expect("Failed to run ckq index");
    assert!(
        output.status.success(),
        "Index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        temp_dir.path().join(".ckq").exists(),
        "Index directory not created"
    );

    let output = ckq_command()
        .args(["--clean", "."])
        .current_dir(temp_dir.path())
        .output()
        .expect("Failed to run ckq clean");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stdout.contains("Index cleaned") || stderr.contains("Index cleaned"),
        "Expected 'Index cleaned' in output, got stdout: {stdout}, stderr: {stderr}"
    );
}

#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn test_add_single_file_to_index() {
    let temp_dir = TempDir::new().unwrap();

    let test_file = temp_dir.path().join("test_file.txt");
    fs::write(&test_file, "This is test content for indexing").unwrap();

    let output = ckq_command()
        .args(["--index", "."])
        .current_dir(temp_dir.path())
        .output()
        .expect("Failed to create index");
    assert!(output.status.success(), "Failed to create initial index");

    let new_file = temp_dir.path().join("new_file.txt");
    fs::write(&new_file, "New file content to be added").unwrap();

    // Add the new file by absolute path, from outside the corpus.
    let output = ckq_command()
        .args(["--add", new_file.to_str().unwrap()])
        .output()
        .expect("Failed to run ckq --add");

    assert!(
        output.status.success(),
        "Failed to add file: stderr: {}, stdout: {}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stdout.contains("added") || stderr.contains("added"),
        "Expected the index summary, got stdout: {stdout}, stderr: {stderr}"
    );

    // Verify the file was actually added by searching for it
    let output = ckq_command()
        .args(["New file", "."])
        .current_dir(temp_dir.path())
        .output()
        .expect("Failed to search for added file");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("New file content"),
        "Added file content not found in search"
    );
}

#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn test_add_file_with_relative_path() {
    let temp_dir = TempDir::new().unwrap();

    let output = ckq_command()
        .args(["--index", "."])
        .current_dir(temp_dir.path())
        .output()
        .expect("Failed to create index");
    assert!(output.status.success());

    fs::write(
        temp_dir.path().join("relative_file.txt"),
        "Relative path content",
    )
    .unwrap();

    let output = ckq_command()
        .args(["--add", "relative_file.txt"])
        .current_dir(temp_dir.path())
        .output()
        .expect("Failed to run ckq --add with relative path");

    assert!(
        output.status.success(),
        "Failed to add file with relative path: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = ckq_command()
        .args(["Relative path", "."])
        .current_dir(temp_dir.path())
        .output()
        .expect("Failed to search");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Relative path content"));
}

/// Two ck processes indexing the same directory concurrently must queue on
/// the store's own write lock instead of interleaving transactions. Both
/// should succeed and the final index should contain every file.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn test_concurrent_indexing_is_serialized() {
    use std::process::Stdio;

    let temp_dir = TempDir::new().unwrap();
    let file_count = 30;
    for i in 0..file_count {
        fs::write(
            temp_dir.path().join(format!("file{i}.txt")),
            format!("searchable content number {i}\n").repeat(50),
        )
        .unwrap();
    }

    let spawn_indexer = || {
        ckq_command()
            .arg("--index")
            .arg("-q")
            .arg(".")
            .current_dir(temp_dir.path())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to spawn ckq --index")
    };

    let first = spawn_indexer();
    let second = spawn_indexer();

    for child in [first, second] {
        let output = child.wait_with_output().expect("Failed to wait on ckq");
        assert!(
            output.status.success(),
            "concurrent ck --index failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let out = ckq_command()
        .arg("--status-json")
        .current_dir(temp_dir.path())
        .output()
        .expect("ckq --status-json should run");
    assert!(
        out.status.success(),
        "status after concurrent indexing failed"
    );
    let status: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(
        status["total_files"].as_u64(),
        Some(file_count as u64),
        "the index lost entries under concurrent indexing: {status}"
    );
}

/// Command-mode flags must honor an explicit path argument: `ckq --index /dir`
/// must index /dir, not the cwd, from any working directory.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn test_command_flags_honor_explicit_path() {
    let target = TempDir::new().unwrap();
    let elsewhere = TempDir::new().unwrap();
    fs::write(target.path().join("a.txt"), "indexable content here").unwrap();

    let output = ckq_command()
        .args(["--index", "-q", target.path().to_str().unwrap()])
        .current_dir(elsewhere.path())
        .output()
        .expect("Failed to run ckq --index <path>");
    assert!(
        output.status.success(),
        "ck --index <path> failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        target.path().join(".ckq").exists(),
        "--index must create the index at the given path"
    );
    assert!(
        !elsewhere.path().join(".ckq").exists(),
        "--index must not index the cwd when a path is given"
    );

    let output = ckq_command()
        .args(["--status", target.path().to_str().unwrap()])
        .current_dir(elsewhere.path())
        .output()
        .expect("Failed to run ckq --status <path>");
    assert!(output.status.success());

    let output = ckq_command()
        .args(["--clean", target.path().to_str().unwrap()])
        .current_dir(elsewhere.path())
        .output()
        .expect("Failed to run ckq --clean <path>");
    assert!(output.status.success());
    assert!(
        !target.path().join(".ckq").exists(),
        "--clean must remove the index at the given path"
    );
}

/// The lexical index is the store itself, written by `--index`: files added
/// or edited afterwards are invisible to `--lex` until the next `--index`,
/// and never appear as stale content.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn test_lexical_search_reflects_file_changes() {
    let temp_dir = TempDir::new().unwrap();
    fs::write(
        temp_dir.path().join("a.txt"),
        "alphaterm appears in the original corpus",
    )
    .unwrap();

    let status = ckq_command()
        .args(["--index", "-q", "."])
        .current_dir(temp_dir.path())
        .status()
        .expect("ckq --index should run");
    assert!(status.success());

    let output = ckq_command()
        .args(["--lex", "alphaterm", ".", "--jsonl", "-q"])
        .current_dir(temp_dir.path())
        .output()
        .expect("ckq --lex should run");
    assert!(
        output.status.success(),
        "initial lexical search failed.\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // A file added after the index run is not in the store, so the next
    // search must not find it until the index runs again.
    fs::write(
        temp_dir.path().join("b.txt"),
        "zebraterm only exists in the new file",
    )
    .unwrap();

    let output = ckq_command()
        .args(["--lex", "zebraterm", ".", "--jsonl", "-q"])
        .current_dir(temp_dir.path())
        .output()
        .expect("ckq --lex should run");
    assert!(
        !output.status.success(),
        "search is read-only: a file the index never saw cannot be found yet"
    );

    let output = ckq_command()
        .args(["--index", "-q", "."])
        .current_dir(temp_dir.path())
        .status()
        .expect("ckq --index should run");
    assert!(output.success());
    let out = ckq_command()
        .args(["--lex", "zebraterm", ".", "--jsonl", "-q"])
        .current_dir(temp_dir.path())
        .output()
        .expect("ckq --lex should run");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && stdout.contains("b.txt"),
        "lexical search must see files added after a re-index; stdout: {stdout}"
    );

    // Modify the original file: removed content must stop matching
    fs::write(temp_dir.path().join("a.txt"), "completely different now").unwrap();
    let output = ckq_command()
        .args(["--index", "-q", "."])
        .current_dir(temp_dir.path())
        .status()
        .expect("ckq --index should run");
    assert!(output.success());
    let output = ckq_command()
        .args(["--lex", "alphaterm", ".", "--jsonl", "-q"])
        .current_dir(temp_dir.path())
        .output()
        .expect("ckq --lex should run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.contains("a.txt"),
        "lexical search returned stale content for a modified file; stdout: {stdout}"
    );
}

/// Lexical search reads what `--index` stored, so `--hidden` must control
/// whether dot-prefixed files land in the index.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn test_hidden_flag_lexical_index() {
    let temp_dir = setup_hidden_workspace();

    let output = ckq_command()
        .args(["--index", "-q", "."])
        .current_dir(temp_dir.path())
        .status()
        .expect("ckq --index should run");
    assert!(output.success());

    let output = ckq_command()
        .args(["--lex", "needle", "."])
        .current_dir(temp_dir.path())
        .output()
        .expect("Failed to run ckq --lex");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("visible.txt"),
        "lexical index should contain the visible file; stdout: {stdout}"
    );
    assert!(
        !stdout.contains(".hidden-file.txt") && !stdout.contains("inside.txt"),
        "hidden content should NOT be in the lexical index without --hidden; stdout: {stdout}"
    );

    let output = ckq_command()
        .args(["--index", "-q", "--hidden", "."])
        .current_dir(temp_dir.path())
        .status()
        .expect("ckq --index --hidden should run");
    assert!(output.success());

    let output = ckq_command()
        .args(["--hidden", "--lex", "needle", "."])
        .current_dir(temp_dir.path())
        .output()
        .expect("Failed to run ckq --hidden --lex");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains(".hidden-file.txt"),
        "hidden file SHOULD be in the lexical index with --hidden; stdout: {stdout}"
    );
    assert!(
        stdout.contains("inside.txt"),
        "file in hidden dir SHOULD be in the lexical index with --hidden; stdout: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Several directories in one session
//
// ckq keeps one index per directory, in a `.ckq` beside the corpus. That makes
// a session which searches corpus A and then corpus B the normal case rather
// than an edge case, and these cover the ways it can break: an index written
// to the wrong place, and a result from the wrong corpus.
// ---------------------------------------------------------------------------

/// Two corpora with no vocabulary in common. Each carries a nonsense marker so
/// a leak is unambiguous: the word cannot occur in the other corpus by chance,
/// and it cannot be stemmed into the other's vocabulary.
fn write_two_corpora(a: &Path, b: &Path) {
    fs::write(
        a.join("sailing.md"),
        "# Sailing\n\nHarbours, tides and moorings for small boats. Marker zephyrquux.\n",
    )
    .unwrap();
    fs::write(
        b.join("baking.md"),
        "# Baking\n\nSourdough starters, proving and ovens. Marker quibblewomp.\n",
    )
    .unwrap();
}

/// Run a search and hand back stdout; the tests that call this have indexed
/// `dir` first, and the marker queries always match.
fn search_stdout(mode: &str, query: &str, dir: &Path) -> Option<String> {
    let output = ckq_command()
        .args([mode, query, dir.to_str().unwrap()])
        .output()
        .expect("Failed to run ckq search");
    if output.status.success() {
        Some(String::from_utf8(output.stdout).unwrap())
    } else {
        eprintln!("search failed: {}", String::from_utf8_lossy(&output.stderr));
        None
    }
}

#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn test_each_directory_gets_its_own_index() {
    let a = TempDir::new().unwrap();
    let b = TempDir::new().unwrap();
    let elsewhere = TempDir::new().unwrap();
    write_two_corpora(a.path(), b.path());

    // Index both by path argument, from a directory that is neither of them.
    for corpus in [a.path(), b.path()] {
        let output = ckq_command()
            .args(["--index", "-q", corpus.to_str().unwrap()])
            .current_dir(elsewhere.path())
            .output()
            .expect("Failed to run ckq index");
        assert!(output.status.success());
    }

    // The path argument decides where the index lands, not the current directory.
    assert!(a.path().join(".ckq").is_dir());
    assert!(b.path().join(".ckq").is_dir());
    assert!(
        !elsewhere.path().join(".ckq").exists(),
        "indexing by path argument must not index the current directory"
    );

    // Each search answers from its own corpus and nothing from the other.
    assert!(
        search_stdout("--lex", "zephyrquux", a.path()).is_some_and(|s| s.contains("sailing.md")),
        "corpus A must find its own marker"
    );
    assert!(
        search_stdout("--lex", "quibblewomp", b.path()).is_some_and(|s| s.contains("baking.md")),
        "corpus B must find its own marker"
    );
}

#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn test_lexical_search_does_not_leak_between_directories() {
    let a = TempDir::new().unwrap();
    let b = TempDir::new().unwrap();
    write_two_corpora(a.path(), b.path());

    for corpus in [a.path(), b.path()] {
        let output = ckq_command()
            .args(["--index", "-q", corpus.to_str().unwrap()])
            .output()
            .expect("Failed to run ckq index");
        assert!(output.status.success());
    }

    // Positive control. Without this, an empty result below would prove nothing:
    // a stale or missing index also returns nothing.
    let Some(own) = search_stdout("--lex", "zephyrquux", a.path()) else {
        panic!("corpus A's own marker must be findable, or the leak check is vacuous");
    };
    assert!(
        own.contains("sailing.md"),
        "corpus A must find its own marker, otherwise the leak check is vacuous"
    );

    // The leak itself: A must not answer with B's content.
    let foreign = search_stdout("--lex", "quibblewomp", a.path());
    assert!(
        foreign.as_deref().is_none_or(|f| !f.contains("baking.md")),
        "searching corpus A returned a file from corpus B: {foreign:?}"
    );
}

#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn test_alternating_semantic_searches_stay_scoped() {
    let a = TempDir::new().unwrap();
    let b = TempDir::new().unwrap();
    write_two_corpora(a.path(), b.path());

    for corpus in [a.path(), b.path()] {
        let output = ckq_command()
            .args(["--index", "-q", corpus.to_str().unwrap()])
            .output()
            .expect("Failed to run ckq index");
        assert!(output.status.success());
    }

    // Two rounds, so the second visit to each corpus meets state left by the
    // other. One round would not catch carried-over state.
    for round in 1..=2 {
        if let Some(hits) = search_stdout("--sem", "boats and the sea", a.path()) {
            assert!(
                !hits.contains("baking.md"),
                "round {round}: corpus A returned a file from corpus B"
            );
        }
        if let Some(hits) = search_stdout("--sem", "bread and yeast", b.path()) {
            assert!(
                !hits.contains("sailing.md"),
                "round {round}: corpus B returned a file from corpus A"
            );
        }
    }
}
