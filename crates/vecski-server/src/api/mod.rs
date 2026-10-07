pub mod docs;
pub mod translators;

use crate::auth::require_api_key;
use crate::state::AppState;
use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::routing::{get, post};
use std::sync::Arc;
use tower_http::compression::CompressionLayer;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

pub fn router(state: Arc<AppState>) -> Router {
    let v1 = Router::new()
        .route(
            "/translators",
            post(translators::create).get(translators::list),
        )
        .route("/translators/import", post(translators::import))
        .route(
            "/translators/{id}",
            get(translators::get).delete(translators::delete),
        )
        .route("/translators/{id}/translate", post(translators::translate))
        .route(
            "/translators/{id}/evaluate",
            post(translators::evaluate_handler),
        )
        .route("/translators/{id}/safetensors", get(translators::export))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            require_api_key,
        ));

    Router::new()
        .route("/", get(docs::manifest))
        .route("/healthz", get(docs::healthz))
        .route("/llms.txt", get(docs::llms_txt))
        .route("/openapi.json", get(docs::openapi))
        .nest("/v1", v1)
        .layer(DefaultBodyLimit::max(state.config.max_body_bytes))
        .layer(CompressionLayer::new())
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
