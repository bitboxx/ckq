//! `scripts/resolve-gguf.sh`: the snapshot a one-file build embeds must be
//! the one hf-hub (and so a normal rogo download) would serve — the commit
//! `refs/main` names — never whichever snapshot happens to sort first. These
//! run against a fake cache and need no model, no network and no build.

#[cfg(unix)]
mod resolve {
    use std::path::PathBuf;
    use std::process::Command;

    fn script() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/resolve-gguf.sh")
    }

    /// The Hugging Face cache layout for one repo: `refs/main` naming a
    /// commit, and snapshots holding (or not holding) the file.
    /// The registry row the alias `gemma-q4` resolves to.
    const REPO: &str = "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF";
    const FILE: &str = "embeddinggemma-300M-qat-Q4_0.gguf";

    fn fake_cache(dir: &std::path::Path, revs: &[(&str, bool)]) -> PathBuf {
        let repo_dir = dir.join(format!("models--{}", REPO.replace('/', "--")));
        std::fs::create_dir_all(repo_dir.join("refs")).unwrap();
        std::fs::create_dir_all(repo_dir.join("snapshots")).unwrap();
        std::fs::write(repo_dir.join("refs").join("main"), revs[0].0).unwrap();
        for (rev, has_file) in revs {
            let snap = repo_dir.join("snapshots").join(rev);
            std::fs::create_dir_all(&snap).unwrap();
            if *has_file {
                std::fs::write(snap.join(FILE), "gguf bytes").unwrap();
            }
        }
        repo_dir
    }

    fn run(cache: &std::path::Path, alias: &str) -> (bool, String, String) {
        let out = Command::new("bash")
            .arg(script())
            .arg(alias)
            .env("ROGO_MODELS_CACHE", cache)
            .output()
            .expect("run the resolver");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    }

    /// Two revisions hold the file and the lexically-first is not the one
    /// `refs/main` points at: the ref wins. The old glob picked the first.
    #[test]
    fn refs_main_picks_the_revision_not_the_lexically_first() {
        let dir = tempfile::TempDir::new().unwrap();
        // `aaaa` sorts before `ffff`; the ref names `ffff`.
        fake_cache(dir.path(), &[("ffff0000", true), ("aaaa1111", true)]);
        let (ok, stdout, stderr) = run(dir.path(), "gemma-q4");
        assert!(ok, "stderr: {stderr}");
        let want = dir.path().join(format!(
            "models--{}/snapshots/ffff0000/{}",
            REPO.replace('/', "--"),
            FILE
        ));
        assert_eq!(stdout.trim(), want.display().to_string());
    }

    /// Without a ref, more than one candidate is ambiguity, not a choice.
    #[test]
    fn two_candidates_without_a_ref_are_an_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let repo_dir = fake_cache(dir.path(), &[("ffff0000", true), ("aaaa1111", true)]);
        std::fs::remove_file(repo_dir.join("refs").join("main")).unwrap();
        let (ok, _, stderr) = run(dir.path(), "gemma-q4");
        assert!(!ok, "ambiguity must fail the build");
        assert!(stderr.contains("aaaa1111"), "stderr: {stderr}");
        assert!(stderr.contains("ffff0000"), "stderr: {stderr}");
    }

    /// Nothing downloaded yet: the error says how to get it.
    #[test]
    fn an_empty_cache_names_the_download_step() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(
            dir.path()
                .join("models--ggml-org--embeddinggemma-300M-qat-q4_0-GGUF/snapshots"),
        )
        .unwrap();
        let (ok, _, stderr) = run(dir.path(), "gemma-q4");
        assert!(!ok);
        assert!(stderr.contains("not in"), "stderr: {stderr}");
        assert!(stderr.contains("rogo --index"), "stderr: {stderr}");
    }

    /// A ref whose snapshot lost the file is a broken cache, not a fallback
    /// to the glob.
    #[test]
    fn a_ref_without_its_file_is_an_error() {
        let dir = tempfile::TempDir::new().unwrap();
        fake_cache(dir.path(), &[("ffff0000", false), ("aaaa1111", true)]);
        let (ok, _, stderr) = run(dir.path(), "gemma-q4");
        assert!(!ok, "a dangling ref must not fall back to the glob");
        assert!(stderr.contains("ffff0000"), "stderr: {stderr}");
    }
}
