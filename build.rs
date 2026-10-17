//! The `embedded-model` build: one GGUF inside the binary.
//!
//! With the feature on, two environment variables drive the build:
//!
//! - `CKQ_EMBED_GGUF`: path of the GGUF file to embed,
//! - `CKQ_EMBED_ALIAS`: its registry alias in `src/models.rs`.
//!
//! Both must be set and the alias must be one the registry knows; anything
//! else fails the build with the message that says so. The alias list is read
//! out of `src/models.rs`, so an alias added there is embeddable without
//! touching this file. `models.rs` gets the path and alias (for
//! `include_bytes!` and the default alias) plus the first 16 hex of the
//! GGUF's blake3, so a run never has to hash 278 MB again.
//!
//! Without the feature this script emits nothing and does nothing at all.

use std::env;
use std::fs;
use std::path::Path;

fn main() {
    if env::var_os("CARGO_FEATURE_EMBEDDED_MODEL").is_none() {
        return;
    }

    let gguf = require_var("CKQ_EMBED_GGUF");
    let alias = require_var("CKQ_EMBED_ALIAS");

    let gguf_path = fs::canonicalize(&gguf).unwrap_or_else(|e| {
        fail(&format!(
            "CKQ_EMBED_GGUF does not point at a readable file: {gguf} ({e})"
        ))
    });

    let models_rs = Path::new(&env::var("CARGO_MANIFEST_DIR").expect("cargo sets it"))
        .join("src")
        .join("models.rs");
    let src = fs::read_to_string(&models_rs)
        .unwrap_or_else(|e| fail(&format!("cannot read {} ({e})", models_rs.display())));
    let known: Vec<&str> = src
        .lines()
        .filter_map(|l| {
            l.trim()
                .strip_prefix("alias: \"")
                .and_then(|l| l.strip_suffix("\","))
        })
        .collect();
    if !known.contains(&alias.as_str()) {
        fail(&format!(
            "CKQ_EMBED_ALIAS '{alias}' is not in the registry ({}). \
             Available models: {}",
            models_rs.display(),
            known.join(", ")
        ));
    }

    let bytes = fs::read(&gguf_path)
        .unwrap_or_else(|e| fail(&format!("cannot read {} ({e})", gguf_path.display())));
    let hex16 = &blake3::hash(&bytes).to_hex()[..16];

    println!("cargo:rustc-env=CKQ_EMBED_GGUF={}", gguf_path.display());
    println!("cargo:rustc-env=CKQ_EMBED_ALIAS={alias}");
    println!("cargo:rustc-env=CKQ_EMBED_BLAKE3={hex16}");
    println!("cargo:rerun-if-changed={}", gguf_path.display());
    println!("cargo:rerun-if-changed={}", models_rs.display());
    println!("cargo:rerun-if-env-changed=CKQ_EMBED_GGUF");
    println!("cargo:rerun-if-env-changed=CKQ_EMBED_ALIAS");
}

fn require_var(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| {
        fail(&format!(
            "{name} is not set. An embedded-model build needs \
             CKQ_EMBED_GGUF (path of the GGUF) and CKQ_EMBED_ALIAS (its \
             registry alias); scripts/build-onefile.sh sets both."
        ))
    })
}

fn fail(message: &str) -> ! {
    eprintln!("error: {message}");
    std::process::exit(1);
}
