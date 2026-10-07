use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use utoipa::ToSchema;
use vecski_core::CoreError;

/// Error envelope returned by every failing request.
#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorDetail {
    /// Stable machine-readable code: `bad_request`, `not_found`, `conflict`,
    /// `unauthorized`, `unsupported_media_type`, `payload_too_large`, `internal`.
    pub code: &'static str,
    pub message: String,
}

#[derive(Debug)]
pub enum ApiError {
    BadRequest(String),
    NotFound(String),
    Conflict(String),
    Unauthorized,
    UnsupportedMediaType(String),
    Internal(String),
}

impl ApiError {
    fn parts(&self) -> (StatusCode, &'static str, String) {
        match self {
            ApiError::BadRequest(m) => (StatusCode::BAD_REQUEST, "bad_request", m.clone()),
            ApiError::NotFound(m) => (StatusCode::NOT_FOUND, "not_found", m.clone()),
            ApiError::Conflict(m) => (StatusCode::CONFLICT, "conflict", m.clone()),
            ApiError::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                "unauthorized",
                "missing or invalid API key; send `Authorization: Bearer <key>` or `X-API-Key: <key>`".into(),
            ),
            ApiError::UnsupportedMediaType(m) => (StatusCode::UNSUPPORTED_MEDIA_TYPE, "unsupported_media_type", m.clone()),
            ApiError::Internal(m) => (StatusCode::INTERNAL_SERVER_ERROR, "internal", m.clone()),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code, message) = self.parts();
        if status.is_server_error() {
            tracing::error!(%message, "request failed");
        }
        (
            status,
            Json(ErrorBody {
                error: ErrorDetail { code, message },
            }),
        )
            .into_response()
    }
}

impl From<CoreError> for ApiError {
    fn from(e: CoreError) -> Self {
        match e {
            CoreError::Invalid(_) | CoreError::Shape(_) => ApiError::BadRequest(e.to_string()),
            CoreError::Serialization(_) => ApiError::BadRequest(e.to_string()),
            CoreError::Linalg(_) | CoreError::Io(_) => ApiError::Internal(e.to_string()),
        }
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(e: serde_json::Error) -> Self {
        ApiError::BadRequest(format!("invalid JSON: {e}"))
    }
}

impl From<tokio::task::JoinError> for ApiError {
    fn from(e: tokio::task::JoinError) -> Self {
        ApiError::Internal(format!("worker task failed: {e}"))
    }
}
