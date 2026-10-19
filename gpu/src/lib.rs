//! A feature forwarder, not code.
//!
//! Cargo cannot make a crate's default feature platform-specific, and it
//! refuses the same crate twice under two names, so `rogo` cannot ask for
//! `llama-cpp-2/vulkan` by default on Linux and Windows but not macOS through
//! its own `[features]`. Instead `rogo` depends on this crate — target-gated to
//! `not(target_os = "macos")` and optional — and enabling rogo's `vulkan` or
//! `cuda` feature activates it, whose features then reach `llama-cpp-2`
//! through feature unification. macOS never sees this dependency, so its
//! default build stays exactly as it was.
//!
//! Nothing lives here by design; the manifest does all the work.
