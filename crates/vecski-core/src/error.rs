use thiserror::Error;

/// Errors produced by vecski-core.
#[derive(Debug, Error)]
pub enum CoreError {
    #[error("invalid input: {0}")]
    Invalid(String),
    #[error("shape mismatch: {0}")]
    Shape(String),
    #[error("linear algebra failed: {0}")]
    Linalg(String),
    #[error("serialization failed: {0}")]
    Serialization(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, CoreError>;
