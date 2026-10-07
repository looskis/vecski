use crate::error::ApiError;
use crate::state::AppState;
use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::Response;
use std::sync::Arc;

/// Require a configured API key on `/v1/*` when any keys are configured.
pub async fn require_api_key(
    State(st): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Result<Response, ApiError> {
    if st.config.api_keys.is_empty() {
        return Ok(next.run(req).await);
    }
    let presented = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| {
            v.strip_prefix("Bearer ")
                .or_else(|| v.strip_prefix("bearer "))
        })
        .or_else(|| req.headers().get("x-api-key").and_then(|v| v.to_str().ok()))
        .map(str::trim);
    match presented {
        Some(k)
            if st
                .config
                .api_keys
                .iter()
                .any(|known| constant_time_eq(known.as_bytes(), k.as_bytes())) =>
        {
            Ok(next.run(req).await)
        }
        _ => Err(ApiError::Unauthorized),
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}
