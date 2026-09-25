use crate::chunks::IndexedChunkMeta;
use crate::config::PreviewMode;
use ck_core::SearchMode;
use ck_core::SearchResult;
use ck_index::IndexStats;
use ratatui::text::Line;
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

pub struct TuiState {
    pub query: String,
    pub mode: SearchMode,
    pub results: Vec<SearchResult>,
    pub selected_idx: usize,
    pub preview_content: String,
    pub preview_lines: Vec<Line<'static>>, // Colored preview
    pub preview_mode: PreviewMode,
    pub full_file_mode: bool, // false = snippet (±5 lines), true = full file
    pub scroll_offset: usize, // For scrolling in full file mode
    pub status_message: String,
    pub search_path: PathBuf,
    pub selected_files: HashSet<PathBuf>, // For multi-select
    pub search_history: Vec<String>,      // Search history
    pub history_index: usize,             // Current position in history
    pub command_mode: bool,               // true when query starts with /
    pub index_stats: Option<IndexStats>,
    pub last_index_stats_refresh: Option<Instant>,
    pub index_stats_error: Option<String>,
    pub preview_cache: Option<PreviewCache>,
    pub indexing_message: Option<String>,
    pub indexing_progress: Option<f32>,
    pub indexing_active: bool,
    pub indexing_started_at: Option<Instant>,
    pub last_indexing_update: Option<Instant>,
    pub search_in_progress: bool,
}

pub struct PreviewCache {
    pub file: PathBuf,
    pub lines: Vec<String>,
    pub is_pdf: bool,
    pub chunks: Vec<IndexedChunkMeta>,
}
