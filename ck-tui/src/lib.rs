pub mod app;
pub mod chunks;
pub mod colors;
pub mod commands;
pub mod config;
pub mod events;
pub mod preview;
pub mod rendering;
pub mod state;
pub mod utils;

use anyhow::Result;
use std::path::PathBuf;

// Re-export main types for public API
pub use app::TuiApp;
pub use chunks::{
    ChunkColumnChar, ChunkDisplayLine, IndexedChunkMeta, chunk_display_line_to_string,
    chunk_file_live,
};
pub use config::{PreviewMode, TuiConfig};
pub use preview::dump_chunk_view_internal;

/// Main entry point to run the TUI application
pub async fn run_tui(search_path: PathBuf, initial_query: Option<String>) -> Result<()> {
    let app = TuiApp::new(search_path, initial_query);
    app.run().await
}

/// Dump chunk view for a file (used by --dump-chunks CLI)
#[allow(dead_code)]
pub fn dump_chunk_view(
    path: &std::path::Path,
    match_line: Option<usize>,
    full_file_mode: bool,
) -> Result<Vec<String>, String> {
    dump_chunk_view_internal(path, match_line, full_file_mode)
}
