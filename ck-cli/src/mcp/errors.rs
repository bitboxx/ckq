use thiserror::Error;

pub type McpResult<T> = Result<T, McpError>;

#[derive(Error, Debug)]
pub enum McpError {
    #[error("Search error: {0}")]
    #[allow(dead_code)]
    Search(String),

    #[error("Index error: {0}")]
    #[allow(dead_code)]
    Index(String),

    #[error("Model error: {0}")]
    #[allow(dead_code)]
    Model(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid path: {0}")]
    #[allow(dead_code)]
    InvalidPath(String),

    #[error("Operation cancelled")]
    #[allow(dead_code)]
    Cancelled,

    #[error("Internal error: {0}")]
    Internal(#[from] anyhow::Error),
}

impl McpError {
    #[allow(dead_code)]
    pub fn error_code(&self) -> &'static str {
        match self {
            McpError::Search(_) => "SEARCH_ERROR",
            McpError::Index(_) => "INDEX_ERROR",
            McpError::Model(_) => "MODEL_ERROR",
            McpError::Io(_) => "IO_ERROR",
            McpError::InvalidPath(_) => "INVALID_PATH",
            McpError::Cancelled => "CANCELLED",
            McpError::Internal(_) => "INTERNAL_ERROR",
        }
    }
}
