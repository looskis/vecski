//! `/v1/translators` handlers: fit, list, inspect, translate, evaluate, export, import, delete.

use crate::error::ApiError;
use crate::state::{AppState, Entry, TranslatorMeta};
use axum::Json;
use axum::body::Bytes;
use axum::extract::{FromRequest, Multipart, Path, Query, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use utoipa::ToSchema;
use vecski_core::fit::{FitOptions, FitReport};
use vecski_core::{EvalMetrics, Matrix, Method, evaluate, fit, io};

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// Compact held-out metrics shown in listings.
#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct HoldoutSummary {
    pub n: usize,
    pub mean_cosine: f64,
    pub mean_vector_baseline_cosine: f64,
    pub top1_accuracy: f64,
    pub neighbor_overlap_at_k: f64,
    pub k: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct TranslatorSummary {
    /// Stable id (`tr_...`). Use it or `name` in URLs.
    pub id: String,
    #[serde(flatten)]
    pub meta: TranslatorMeta,
    pub source_dim: usize,
    pub target_dim: usize,
    pub method: Option<Method>,
    pub normalize_output: bool,
    pub created_at: String,
    pub n_train: Option<usize>,
    pub holdout: Option<HoldoutSummary>,
    pub warnings: Vec<String>,
    /// Bytes of weights held in memory / written to safetensors.
    pub weight_bytes: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct TranslatorDetail {
    #[serde(flatten)]
    pub summary: TranslatorSummary,
    /// Full fit report (absent for imported translators).
    pub report: Option<FitReport>,
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct ListResponse {
    pub translators: Vec<TranslatorSummary>,
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct DeleteResponse {
    pub id: String,
    pub deleted: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct TranslateResponse {
    pub translator: String,
    pub rows: usize,
    pub dim: usize,
    /// Translated vectors, same order as the input.
    pub vectors: Vec<Vec<f32>>,
}

pub fn summary(e: &Entry) -> TranslatorSummary {
    let r = &e.record;
    let t = &e.translator;
    let rep = r.report.as_ref();
    TranslatorSummary {
        id: r.id.clone(),
        meta: r.meta.clone(),
        source_dim: t.source_dim(),
        target_dim: t.target_dim(),
        method: t.info().method,
        normalize_output: t.normalize_output(),
        created_at: r.created_at.clone(),
        n_train: t.info().n_train,
        holdout: rep
            .and_then(|r| r.holdout.as_ref())
            .map(|h| HoldoutSummary {
                n: h.n,
                mean_cosine: h.mean_cosine,
                mean_vector_baseline_cosine: h.mean_vector_baseline_cosine,
                top1_accuracy: h.top1_accuracy,
                neighbor_overlap_at_k: h.neighbor_overlap_at_k,
                k: h.k,
            }),
        warnings: rep.map(|r| r.warnings.clone()).unwrap_or_default(),
        weight_bytes: t.weight_bytes(),
    }
}

fn detail(e: &Entry) -> TranslatorDetail {
    TranslatorDetail {
        summary: summary(e),
        report: e.record.report.clone(),
    }
}

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// JSON body for fitting. For large pair sets prefer `multipart/form-data`
/// with raw little-endian f32 parts (see `FitMultipartMeta`).
#[derive(Clone, Debug, Deserialize, ToSchema)]
pub struct FitRequest {
    #[serde(flatten)]
    pub meta: TranslatorMeta,
    #[serde(default)]
    pub options: FitOptions,
    /// Vectors from the source model; row `i` pairs with `target[i]`.
    pub source: Vec<Vec<f32>>,
    /// Vectors from the target model for the same texts.
    pub target: Vec<Vec<f32>>,
}

/// JSON `meta` part of a multipart fit request. The other two parts, `source`
/// and `target`, are raw little-endian f32 buffers (`numpy.float32.tobytes()`)
/// of shape `[n, source_dim]` and `[n, target_dim]`.
#[derive(Clone, Debug, Deserialize, ToSchema)]
pub struct FitMultipartMeta {
    #[serde(flatten)]
    pub meta: TranslatorMeta,
    #[serde(default)]
    pub options: FitOptions,
    pub source_dim: usize,
    pub target_dim: usize,
}

/// JSON body for translation. Binary alternative: send `application/octet-stream`
/// of `n * source_dim` little-endian f32 and receive the same for the target.
#[derive(Clone, Debug, Deserialize, ToSchema)]
pub struct TranslateRequest {
    pub vectors: Vec<Vec<f32>>,
}

/// JSON body for evaluation; row `i` of `source` and `target` embed the same text.
#[derive(Clone, Debug, Deserialize, ToSchema)]
pub struct EvaluateRequest {
    /// Neighborhood size for overlap (default 10).
    #[serde(default = "default_k")]
    pub k: usize,
    pub source: Vec<Vec<f32>>,
    pub target: Vec<Vec<f32>>,
}

/// JSON `meta` part of a multipart evaluate request (parts `source`, `target` as in fitting).
#[derive(Clone, Debug, Deserialize, ToSchema)]
pub struct EvaluateMultipartMeta {
    #[serde(default = "default_k")]
    pub k: usize,
    pub source_dim: usize,
    pub target_dim: usize,
}

fn default_k() -> usize {
    10
}

// ---------------------------------------------------------------------------
// Pair ingestion shared by fit and evaluate
// ---------------------------------------------------------------------------

struct Pairs {
    meta: serde_json::Value,
    source: Matrix,
    target: Matrix,
}

#[derive(Deserialize)]
struct PairsJson {
    source: Vec<Vec<f32>>,
    target: Vec<Vec<f32>>,
    #[serde(flatten)]
    rest: serde_json::Map<String, serde_json::Value>,
}

#[derive(Deserialize)]
struct DimsOnly {
    source_dim: usize,
    target_dim: usize,
}

fn content_type(headers: &HeaderMap) -> String {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| {
            s.split(';')
                .next()
                .unwrap_or("")
                .trim()
                .to_ascii_lowercase()
        })
        .unwrap_or_default()
}

async fn read_pairs(req: Request) -> Result<Pairs, ApiError> {
    let ct = content_type(req.headers());
    if ct == "application/json" || ct.is_empty() {
        let bytes = Bytes::from_request(req, &())
            .await
            .map_err(|e| ApiError::BadRequest(e.to_string()))?;
        let body: PairsJson = serde_json::from_slice(&bytes)?;
        let source = Matrix::from_rows(&body.source)?;
        let target = Matrix::from_rows(&body.target)?;
        return Ok(Pairs {
            meta: serde_json::Value::Object(body.rest),
            source,
            target,
        });
    }
    if ct == "multipart/form-data" {
        let mut mp = Multipart::from_request(req, &())
            .await
            .map_err(|e| ApiError::BadRequest(e.to_string()))?;
        let (mut meta, mut source, mut target) = (None, None, None);
        while let Some(field) = mp
            .next_field()
            .await
            .map_err(|e| ApiError::BadRequest(e.to_string()))?
        {
            let name = field.name().unwrap_or("").to_string();
            let data = field
                .bytes()
                .await
                .map_err(|e| ApiError::BadRequest(format!("part '{name}': {e}")))?;
            match name.as_str() {
                "meta" => meta = Some(data),
                "source" => source = Some(data),
                "target" => target = Some(data),
                other => {
                    return Err(ApiError::BadRequest(format!(
                        "unexpected multipart part '{other}'; expected meta, source, target"
                    )));
                }
            }
        }
        let meta = meta.ok_or_else(|| {
            ApiError::BadRequest("multipart part 'meta' (JSON) is required".into())
        })?;
        let source = source.ok_or_else(|| {
            ApiError::BadRequest("multipart part 'source' (f32 LE bytes) is required".into())
        })?;
        let target = target.ok_or_else(|| {
            ApiError::BadRequest("multipart part 'target' (f32 LE bytes) is required".into())
        })?;
        let meta: serde_json::Value = serde_json::from_slice(&meta)?;
        let dims: DimsOnly = serde_json::from_value(meta.clone()).map_err(|e| {
            ApiError::BadRequest(format!("meta must include source_dim and target_dim: {e}"))
        })?;
        let source = Matrix::from_le_bytes(&source, dims.source_dim)?;
        let target = Matrix::from_le_bytes(&target, dims.target_dim)?;
        return Ok(Pairs {
            meta,
            source,
            target,
        });
    }
    Err(ApiError::UnsupportedMediaType(format!(
        "content type '{ct}' not supported; send application/json or multipart/form-data"
    )))
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// Fit a new translator from paired vectors.
///
/// Send `application/json` (`FitRequest`) for small sets, or `multipart/form-data`
/// with parts `meta` (`FitMultipartMeta` JSON), `source` and `target` (raw
/// little-endian f32) for anything beyond a few thousand pairs. The response
/// carries held-out metrics; read `report.warnings` before trusting the map.
#[utoipa::path(
    post,
    path = "/v1/translators",
    tag = "translators",
    request_body(content = FitRequest, content_type = "application/json"),
    responses(
        (status = 201, description = "Translator fitted and stored", body = TranslatorDetail),
        (status = 400, description = "Malformed pairs or options", body = crate::error::ErrorBody),
        (status = 409, description = "Name already in use", body = crate::error::ErrorBody),
    )
)]
pub async fn create(State(st): State<Arc<AppState>>, req: Request) -> Result<Response, ApiError> {
    let pairs = read_pairs(req).await?;
    #[derive(Deserialize)]
    struct Head {
        #[serde(flatten)]
        meta: TranslatorMeta,
        #[serde(default)]
        options: FitOptions,
    }
    let head: Head = serde_json::from_value(pairs.meta)?;
    if let Some(n) = &head.meta.name {
        AppState::check_name(n)?;
    }
    let (source, target, options) = (pairs.source, pairs.target, head.options);
    tracing::info!(n = source.rows(), d1 = source.cols(), d2 = target.cols(), method = ?options.method, "fitting translator");
    let (translator, report) =
        tokio::task::spawn_blocking(move || fit(&source, &target, &options)).await??;
    tracing::info!(
        method = ?report.method,
        ms = report.fit_ms,
        holdout_cosine = report.holdout.as_ref().map(|h| h.mean_cosine),
        overlap = report.holdout.as_ref().map(|h| h.neighbor_overlap_at_k),
        "fit complete"
    );
    let entry = st.insert(head.meta, Some(report), translator)?;
    Ok((StatusCode::CREATED, Json(detail(&entry))).into_response())
}

/// List stored translators, newest first.
#[utoipa::path(get, path = "/v1/translators", tag = "translators",
    responses((status = 200, body = ListResponse)))]
pub async fn list(State(st): State<Arc<AppState>>) -> Json<ListResponse> {
    Json(ListResponse {
        translators: st.list().iter().map(|e| summary(e)).collect(),
    })
}

/// Fetch one translator (by id or name) including its full fit report.
#[utoipa::path(get, path = "/v1/translators/{id}", tag = "translators",
    params(("id" = String, Path, description = "Translator id (`tr_...`) or name")),
    responses((status = 200, body = TranslatorDetail), (status = 404, body = crate::error::ErrorBody)))]
pub async fn get(
    State(st): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<TranslatorDetail>, ApiError> {
    let e = st.get(&id)?;
    Ok(Json(detail(&e)))
}

/// Delete a translator and its on-disk file.
#[utoipa::path(delete, path = "/v1/translators/{id}", tag = "translators",
    params(("id" = String, Path, description = "Translator id or name")),
    responses((status = 200, body = DeleteResponse), (status = 404, body = crate::error::ErrorBody)))]
pub async fn delete(
    State(st): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<DeleteResponse>, ApiError> {
    let e = st.delete(&id)?;
    Ok(Json(DeleteResponse {
        id: e.record.id.clone(),
        deleted: true,
    }))
}

fn wants_binary(headers: &HeaderMap, input_was_binary: bool) -> bool {
    match headers.get(header::ACCEPT).and_then(|v| v.to_str().ok()) {
        Some(a) if a.contains("application/octet-stream") => true,
        Some(a) if a.contains("application/json") => false,
        _ => input_was_binary,
    }
}

/// Translate vectors from the source space into the target space.
///
/// JSON in → JSON out by default. For throughput send `Content-Type:
/// application/octet-stream` with `n * source_dim` little-endian f32 values and
/// (optionally) `Accept: application/octet-stream` to get `n * target_dim`
/// f32 back; the response headers `X-Vecski-Rows` and `X-Vecski-Dim` give the shape.
#[utoipa::path(post, path = "/v1/translators/{id}/translate", tag = "translate",
    params(("id" = String, Path, description = "Translator id or name")),
    request_body(content = TranslateRequest, content_type = "application/json"),
    responses(
        (status = 200, description = "Translated vectors", body = TranslateResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 404, body = crate::error::ErrorBody),
    ))]
pub async fn translate(
    State(st): State<Arc<AppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let entry = st.get(&id)?;
    let tr = entry.translator.clone();
    let ct = content_type(&headers);
    let binary_in = ct == "application/octet-stream";
    let input = if binary_in {
        Matrix::from_le_bytes(&body, tr.source_dim())?
    } else if ct == "application/json" || ct.is_empty() {
        let req: TranslateRequest = serde_json::from_slice(&body)?;
        Matrix::from_rows(&req.vectors)?
    } else {
        return Err(ApiError::UnsupportedMediaType(format!(
            "content type '{ct}' not supported; send application/json or application/octet-stream"
        )));
    };
    if input.cols() != tr.source_dim() {
        return Err(ApiError::BadRequest(format!(
            "vectors have {} dimensions but translator '{}' expects {}",
            input.cols(),
            entry.record.id,
            tr.source_dim()
        )));
    }
    let rows = input.rows();
    // Tiny requests run inline: a thread hop costs more than the GEMM.
    let out = if rows * tr.source_dim() * tr.target_dim() < 4_000_000 {
        tr.translate(&input)?
    } else {
        tokio::task::spawn_blocking(move || tr.translate(&input)).await??
    };
    let dim = out.cols();
    if wants_binary(&headers, binary_in) {
        let mut resp = out.to_le_bytes().into_response();
        let h = resp.headers_mut();
        h.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/octet-stream"),
        );
        h.insert("x-vecski-rows", HeaderValue::from(rows));
        h.insert("x-vecski-dim", HeaderValue::from(dim));
        h.insert(
            "x-vecski-translator",
            HeaderValue::from_str(&entry.record.id).unwrap_or(HeaderValue::from_static("")),
        );
        Ok(resp)
    } else {
        Ok(Json(TranslateResponse {
            translator: entry.record.id.clone(),
            rows,
            dim,
            vectors: out.to_rows(),
        })
        .into_response())
    }
}

/// Evaluate a translator on held-out pairs you supply.
///
/// Same body formats as fitting (`application/json` with `source`/`target`
/// arrays, or multipart with `meta` = `EvaluateMultipartMeta`). Compare
/// `mean_cosine` against `mean_vector_baseline_cosine`, and watch
/// `neighbor_overlap_at_k`: it is the best cheap predictor of retrieval quality.
#[utoipa::path(post, path = "/v1/translators/{id}/evaluate", tag = "translate",
    params(("id" = String, Path, description = "Translator id or name")),
    request_body(content = EvaluateRequest, content_type = "application/json"),
    responses((status = 200, body = EvalMetrics), (status = 400, body = crate::error::ErrorBody), (status = 404, body = crate::error::ErrorBody)))]
pub async fn evaluate_handler(
    State(st): State<Arc<AppState>>,
    Path(id): Path<String>,
    req: Request,
) -> Result<Json<EvalMetrics>, ApiError> {
    let entry = st.get(&id)?;
    let tr = entry.translator.clone();
    let pairs = read_pairs(req).await?;
    #[derive(Deserialize)]
    struct Head {
        #[serde(default = "default_k")]
        k: usize,
    }
    let head: Head = serde_json::from_value(pairs.meta)?;
    if head.k == 0 {
        return Err(ApiError::BadRequest("k must be positive".into()));
    }
    let (source, target) = (pairs.source, pairs.target);
    let metrics =
        tokio::task::spawn_blocking(move || evaluate(&tr, &source, &target, head.k)).await??;
    Ok(Json(metrics))
}

/// Download the translator as a safetensors file (`weight` [source_dim, target_dim],
/// `bias` [target_dim], `target_mean` [target_dim]; apply `y = x @ weight + bias`,
/// then L2-normalize if metadata `normalize_output` is `true`).
#[utoipa::path(get, path = "/v1/translators/{id}/safetensors", tag = "translators",
    params(("id" = String, Path, description = "Translator id or name")),
    responses((status = 200, description = "safetensors bytes", content_type = "application/octet-stream"), (status = 404, body = crate::error::ErrorBody)))]
pub async fn export(
    State(st): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let entry = st.get(&id)?;
    let bytes = AppState::to_bytes(&entry)?;
    let fname = format!(
        "{}.safetensors",
        entry
            .record
            .meta
            .name
            .clone()
            .unwrap_or_else(|| entry.record.id.clone())
    );
    let mut resp = bytes.into_response();
    let h = resp.headers_mut();
    h.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/octet-stream"),
    );
    if let Ok(v) = HeaderValue::from_str(&format!("attachment; filename=\"{fname}\"")) {
        h.insert(header::CONTENT_DISPOSITION, v);
    }
    Ok(resp)
}

/// Import a translator from safetensors bytes (raw request body).
///
/// Accepts files exported by this server or any file with an F32 `weight`
/// tensor of shape `[source_dim, target_dim]` plus optional `bias` and
/// `target_mean`. Metadata fields go in the query string.
#[utoipa::path(post, path = "/v1/translators/import", tag = "translators",
    params(TranslatorMeta),
    request_body(content = Vec<u8>, content_type = "application/octet-stream"),
    responses((status = 201, body = TranslatorDetail), (status = 400, body = crate::error::ErrorBody), (status = 409, body = crate::error::ErrorBody)))]
pub async fn import(
    State(st): State<Arc<AppState>>,
    Query(meta): Query<TranslatorMeta>,
    body: Bytes,
) -> Result<Response, ApiError> {
    if body.is_empty() {
        return Err(ApiError::BadRequest(
            "request body must contain safetensors bytes".into(),
        ));
    }
    let (translator, file_meta) = io::from_safetensors(&body)?;
    // Recover the embedded record (if this came from a vecski export) for its report.
    let report = file_meta
        .get("vecski.record")
        .and_then(|j| serde_json::from_str::<crate::state::Record>(j).ok())
        .and_then(|r| r.report);
    let meta = TranslatorMeta {
        name: meta.name.or_else(|| file_meta.get("name").cloned()),
        ..meta
    };
    let entry = st.insert(meta, report, translator)?;
    Ok((StatusCode::CREATED, Json(detail(&entry))).into_response())
}
