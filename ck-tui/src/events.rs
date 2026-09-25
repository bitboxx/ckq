use ck_core::SearchResult;

#[derive(Debug)]
pub enum UiEvent {
    Indexing {
        generation: u64,
        message: String,
        progress: Option<f32>,
    },
    IndexingDone {
        generation: u64,
    },
    SearchProgress {
        generation: u64,
        message: String,
    },
    SearchCompleted {
        generation: u64,
        results: Vec<SearchResult>,
        summary: String,
        query: String,
    },
    SearchFailed {
        generation: u64,
        error: String,
    },
}
