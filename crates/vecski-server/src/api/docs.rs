//! Discovery endpoints for humans and agents: `/`, `/healthz`, `/llms.txt`, `/openapi.json`.

use crate::state::AppState;
use axum::Json;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use serde_json::json;
use std::sync::Arc;
use utoipa::OpenApi;
use utoipa::ToSchema;

#[derive(Serialize, ToSchema)]
pub struct Health {
    pub status: &'static str,
    pub version: &'static str,
    pub translators: usize,
}

/// Liveness + a count of loaded translators.
#[utoipa::path(get, path = "/healthz", tag = "meta", responses((status = 200, body = Health)))]
pub async fn healthz(State(st): State<Arc<AppState>>) -> Json<Health> {
    Json(Health {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        translators: st.len(),
    })
}

/// Machine-readable manifest of what this server does and how to call it.
#[utoipa::path(get, path = "/", tag = "meta", responses((status = 200, description = "Service manifest")))]
pub async fn manifest(State(st): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let base = st.config.public_url.clone().unwrap_or_default();
    Json(json!({
        "name": "vecski",
        "version": env!("CARGO_PKG_VERSION"),
        "summary": "Fit and apply closed-form translators between embedding spaces (old model -> new model) from a few thousand paired vectors. Translation is one matrix multiply; quality is reported against a mean-vector baseline so you can decide whether to trust it.",
        "auth": if st.config.api_keys.is_empty() { "none" } else { "bearer: Authorization: Bearer <key> (or X-API-Key) on /v1/*" },
        "docs": { "openapi": format!("{base}/openapi.json"), "llms_txt": format!("{base}/llms.txt") },
        "endpoints": [
            { "method": "POST",   "path": "/v1/translators",                   "what": "fit from pairs (JSON or multipart f32 LE)" },
            { "method": "GET",    "path": "/v1/translators",                   "what": "list" },
            { "method": "GET",    "path": "/v1/translators/{id}",              "what": "detail + fit report" },
            { "method": "DELETE", "path": "/v1/translators/{id}",              "what": "delete" },
            { "method": "POST",   "path": "/v1/translators/{id}/translate",    "what": "translate vectors (JSON or application/octet-stream f32 LE)" },
            { "method": "POST",   "path": "/v1/translators/{id}/evaluate",     "what": "held-out metrics on pairs you supply" },
            { "method": "GET",    "path": "/v1/translators/{id}/safetensors",  "what": "download weights" },
            { "method": "POST",   "path": "/v1/translators/import",            "what": "upload safetensors weights" },
            { "method": "GET",    "path": "/healthz",                          "what": "liveness" }
        ],
        "limits": { "max_body_bytes": st.config.max_body_bytes },
        "guidance": [
            "Pairs: embed the same ~10,000 texts with both models; more than 10-20x the dimension gives diminishing returns.",
            "Fit on a sample of the corpus you will actually translate; translators degrade off-distribution.",
            "Query-side translation (new queries -> old index) retains ~98% of recall; document-side (old docs -> new space) recovers only part of the new model's gain.",
            "Always compare holdout.mean_cosine with holdout.mean_vector_baseline_cosine, and look at neighbor_overlap_at_k.",
            "Do not mix translated and natively embedded vectors in one index without checking score distributions."
        ]
    }))
}

/// Plain-language quickstart for LLM agents.
#[utoipa::path(get, path = "/llms.txt", tag = "meta", responses((status = 200, description = "Markdown quickstart", content_type = "text/markdown")))]
pub async fn llms_txt(State(st): State<Arc<AppState>>) -> Response {
    let base = st
        .config
        .public_url
        .clone()
        .unwrap_or_else(|| "http://localhost:8080".into());
    let body = LLMS_TXT.replace("{BASE}", &base);
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/markdown; charset=utf-8",
        )],
        body,
    )
        .into_response()
}

/// OpenAPI 3.1 document.
pub async fn openapi() -> Json<utoipa::openapi::OpenApi> {
    Json(ApiDoc::openapi())
}

#[derive(OpenApi)]
#[openapi(
    info(
        title = "vecski",
        description = "Fit and apply translators between embedding spaces. Closed-form (Procrustes / ridge), one GEMM per batch, retrieval-style evaluation with a mean-vector baseline.",
        license(name = "Apache-2.0")
    ),
    paths(
        crate::api::docs::manifest,
        crate::api::docs::healthz,
        crate::api::docs::llms_txt,
        crate::api::translators::create,
        crate::api::translators::list,
        crate::api::translators::get,
        crate::api::translators::delete,
        crate::api::translators::translate,
        crate::api::translators::evaluate_handler,
        crate::api::translators::export,
        crate::api::translators::import,
    ),
    components(schemas(
        crate::api::translators::FitMultipartMeta,
        crate::api::translators::EvaluateMultipartMeta,
        vecski_core::fit::Candidate,
        vecski_core::fit::LambdaPoint,
        vecski_core::TranslatorInfo,
    )),
    tags(
        (name = "translators", description = "Fit, inspect, export, import, delete"),
        (name = "translate", description = "Apply and evaluate"),
        (name = "meta", description = "Discovery and health")
    )
)]
pub struct ApiDoc;

const LLMS_TXT: &str = r#"# vecski

vecski fits a linear map between two embedding models' vector spaces from paired
examples and applies it fast. Use it when you have vectors from an old model and
want them usable with a new model (or new queries usable against an old index)
without re-embedding everything.

Base URL: {BASE}
OpenAPI: {BASE}/openapi.json
Auth: if the server was started with API keys, send `Authorization: Bearer <key>` on /v1/*.

## Workflow

1. Pick ~10,000 texts that look like your corpus. Embed each with BOTH models.
   (Minimum useful: about the embedding dimension. Saturation: 10-20x the dimension.)
2. POST the pairs to /v1/translators. Read `report.holdout` and `report.warnings`.
3. POST vectors to /v1/translators/{id}/translate. Store the output.
4. Optionally GET /v1/translators/{id}/safetensors and apply `y = x @ weight + bias`
   (then L2-normalize) yourself, offline, at any scale.

## Fit (JSON, fine up to a few thousand pairs)

```bash
curl -s {BASE}/v1/translators -H 'content-type: application/json' -d '{
  "name": "ada002-to-te3large",
  "source_model": "text-embedding-ada-002",
  "target_model": "text-embedding-3-large",
  "options": {"method": "auto", "holdout_fraction": 0.1, "k": 10},
  "source": [[...1536 floats...], ...],
  "target": [[...3072 floats...], ...]
}'
```

`options.method`: `procrustes` (orthogonal; preserves source geometry; safest),
`ridge` (lower error, distorts geometry), `auto` (fits both, keeps the better
held-out neighbor overlap). `options.normalize_output` defaults to true when the
targets are unit norm.

## Fit (multipart, for large pair sets)

Parts: `meta` (JSON with `source_dim`, `target_dim`, optional `name`,
`source_model`, `target_model`, `options`), `source` and `target` (raw
little-endian float32, row-major, e.g. `np.asarray(X, np.float32).tobytes()`).

```bash
curl -s {BASE}/v1/translators \
  -F 'meta={"name":"minilm-to-mpnet","source_dim":384,"target_dim":768};type=application/json' \
  -F 'source=@source.f32;type=application/octet-stream' \
  -F 'target=@target.f32;type=application/octet-stream'
```

## Translate

JSON:
```bash
curl -s {BASE}/v1/translators/ada002-to-te3large/translate \
  -H 'content-type: application/json' -d '{"vectors": [[...1536 floats...]]}'
```
Binary (fastest): send `application/octet-stream` of n*source_dim float32 LE and
`Accept: application/octet-stream`; the reply is n*target_dim float32 LE with
headers `X-Vecski-Rows` and `X-Vecski-Dim`.

Python:
```python
import numpy as np, urllib.request
x = np.asarray(vectors, np.float32)
req = urllib.request.Request(f"{BASE}/v1/translators/ada002-to-te3large/translate",
    data=x.tobytes(), method="POST",
    headers={"content-type": "application/octet-stream", "accept": "application/octet-stream"})
with urllib.request.urlopen(req) as r:
    y = np.frombuffer(r.read(), np.float32).reshape(int(r.headers["X-Vecski-Rows"]), int(r.headers["X-Vecski-Dim"]))
```

## Reading the report

- `holdout.mean_cosine` vs `holdout.mean_vector_baseline_cosine`: if the gap is
  small, the map learned almost nothing (text embeddings cluster, so the mean
  vector alone often scores 0.8+).
- `holdout.neighbor_overlap_at_k`: fraction of a native query's top-k neighbors
  recovered when the translated vector is used instead. This tracks retrieval
  quality after migration. Expect 0.7-0.95 for good pairs; below 0.5 means
  re-embed instead.
- `holdout.top1_accuracy` / `mrr`: whether each translated vector lands closest
  to its own native counterpart among the held-out set.
- `warnings`: data-size and baseline caveats; treat them as blocking.

## Limits worth knowing

- Translation cannot add information the old model did not encode. Mapping old
  documents into a new model's space recovers only part of the new model's
  quality; mapping new queries into an old index keeps ~98% of recall but caps
  quality at the old model.
- Translators are fit on a distribution. Fit on a sample of the corpus you will
  translate, and re-evaluate on new domains.
- Do not mix translated vectors with natively embedded ones in a single index
  without checking that score distributions match.

## Other endpoints

- GET /v1/translators — list; GET /v1/translators/{id} — detail and report
- POST /v1/translators/{id}/evaluate — body like fit (source/target pairs, `k`)
- GET /v1/translators/{id}/safetensors — download; POST /v1/translators/import?name=... — upload raw safetensors
- DELETE /v1/translators/{id}
- GET /healthz

Errors are `{"error": {"code": "...", "message": "..."}}` with a matching HTTP status.
"#;
