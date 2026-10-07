# vecski

A small, fast Rust API server that **translates embeddings between models**.
Give it a few thousand texts embedded by both an old model and a new model, and
it fits a closed-form linear map (centered orthogonal Procrustes or ridge
regression), reports held-out retrieval metrics next to a mean-vector baseline,
and then applies the map at ~100k vectors/s per node. Humans use it with `curl`;
agents use it through `/llms.txt` and `/openapi.json`.

The design follows the findings in [`research/`](research/): orthogonal maps
are the safe default at ≤50k pairs, ~10–20× the embedding dimension in pairs is
where quality saturates, zero-padding handles mismatched dimensions, cosine
alone is a misleading metric (so neighbor overlap and the mean-vector baseline
are always reported), and translation buys time but not parity with re-embedding.

## Install

Homebrew (macOS and Linux):

```bash
brew tap looskis/vecski https://github.com/looskis/vecski
brew install vecski
```

`brew services start vecski` runs it on `127.0.0.1:8080` with data under `$(brew --prefix)/var/vecski`.

Prebuilt binaries for macOS (arm64, x86_64) and Linux (x86_64, arm64) are on the
[releases page](https://github.com/looskis/vecski/releases). From source:

```bash
cargo install --locked --git https://github.com/looskis/vecski vecski-server
```

## Quick start

```bash
vecski            # or: cargo run --release -p vecski-server
```

```bash
curl -s localhost:8080/            # manifest
curl -s localhost:8080/llms.txt    # agent-oriented guide
curl -s localhost:8080/openapi.json | head -c 400
```

Fit from JSON (fine up to a few thousand pairs):

```bash
curl -s localhost:8080/v1/translators -H 'content-type: application/json' -d '{
  "name": "ada002-to-te3large",
  "source_model": "text-embedding-ada-002",
  "target_model": "text-embedding-3-large",
  "options": {"method": "auto"},
  "source": [[...1536 floats...], ...],
  "target": [[...3072 floats...], ...]
}'
```

Fit from raw float32 (anything larger):

```bash
curl -s localhost:8080/v1/translators \
  -F 'meta={"name":"minilm-to-mpnet","source_dim":384,"target_dim":768};type=application/json' \
  -F 'source=@source.f32;type=application/octet-stream' \
  -F 'target=@target.f32;type=application/octet-stream'
```

Translate:

```bash
curl -s localhost:8080/v1/translators/minilm-to-mpnet/translate \
  -H 'content-type: application/octet-stream' -H 'accept: application/octet-stream' \
  --data-binary @vectors.f32 -o translated.f32 -D -
```

`python examples/quickstart.py` runs the whole loop (fit → translate → download
weights → apply offline) on synthetic pairs.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/translators` | Fit from pairs. JSON body or multipart (`meta`, `source`, `target`). Returns 201 with the fit report. |
| `GET` | `/v1/translators` | List. |
| `GET` | `/v1/translators/{id}` | Detail and full report. `{id}` is the `tr_…` id or the `name`. |
| `DELETE` | `/v1/translators/{id}` | Remove from memory and disk. |
| `POST` | `/v1/translators/{id}/translate` | JSON `{"vectors": [[…]]}` → JSON, or `application/octet-stream` f32 LE → f32 LE (`X-Vecski-Rows`, `X-Vecski-Dim`). |
| `POST` | `/v1/translators/{id}/evaluate` | Metrics on pairs you supply (same body shapes as fit, plus `k`). |
| `GET` | `/v1/translators/{id}/safetensors` | Download `weight`, `bias`, `target_mean`. Apply `y = x @ weight + bias`, then L2-normalize. |
| `POST` | `/v1/translators/import?name=…` | Upload safetensors (ours, or any F32 `weight` `[d_in, d_out]`). |
| `GET` | `/`, `/healthz`, `/llms.txt`, `/openapi.json` | Discovery. |

Errors are `{"error": {"code", "message"}}` with the matching HTTP status.

### Fit options

| Field | Default | Meaning |
| --- | --- | --- |
| `method` | `auto` | `procrustes` (orthogonal, preserves source geometry), `ridge` (lower error, distorts geometry), or `auto` (fit both, keep the better held-out neighbor overlap; ties go to Procrustes). |
| `scale` | `true` | Procrustes: fit an isotropic scale. |
| `lambda` | sweep | Ridge: absolute penalty. Otherwise a 5-point grid relative to the mean Gram eigenvalue is scored on the holdout. |
| `normalize_output` | auto | L2-normalize outputs; defaults to whether targets were unit norm. |
| `holdout_fraction` | `0.1` | Share of pairs held out (capped by `max_holdout`, default 2048). |
| `k` | `10` | Neighborhood size for overlap. |

### Reading a report

- `holdout.mean_cosine` vs `holdout.mean_vector_baseline_cosine`: text
  embeddings cluster, so predicting the mean alone often scores 0.8+. The gap is
  what the translator learned.
- `holdout.neighbor_overlap_at_k`: of a native query's top-k neighbors, the
  fraction recovered when querying with the translated vector instead. This is
  the number that tracks retrieval quality after migration.
- `holdout.top1_accuracy`, `mrr`: does each translated vector land nearest its
  own native counterpart.
- `pairwise_cosine_rmse`: how faithfully the translated set reproduces the
  target space's internal geometry.
- `warnings`: data-size, baseline, and normalization caveats.

## Performance

Apple M-series laptop, 12 cores, release build, 1536 → 3072 dimensions:

| Operation | Time |
| --- | --- |
| Fit Procrustes, 10,000 pairs | ~3.1 s |
| Fit ridge (with 5-λ sweep), 10,000 pairs | ~1.6 s |
| Translate, batch 1 | ~250 µs |
| Translate, batch 1,024 | ~12 ms (~87k vectors/s) |
| Translate, batch 16,384 | ~173 ms (~95k vectors/s) |

Reproduce with `cargo run --release -p vecski-core --example bench -- 1536 3072 10000`.

Why it is fast: every estimator collapses to one fused affine map, so a batch is
a single `faer` GEMM (multithreaded above a small threshold) plus a per-row
normalize; fitting accumulates second moments in f64 with chunked GEMMs and
solves with one SVD or one eigendecomposition; the binary wire format is raw
float32 with no parsing; and the release profile uses fat LTO. For a further
5–15 % on a known host build with `RUSTFLAGS="-C target-cpu=native"`.

## Running

```
vecski [--bind 0.0.0.0:8080] [--data-dir ./data] [--api-keys k1,k2]
       [--max-body-mb 2048] [--threads 0] [--public-url https://…] [--log-json]
```

Every flag has a `VECSKI_*` environment variable. Translators persist as one
safetensors file each under `--data-dir` and reload on start. With `--api-keys`
set, `/v1/*` requires `Authorization: Bearer <key>` or `X-API-Key`.

```bash
docker build -t vecski . && docker run -p 8080:8080 -v vecski-data:/data vecski
```

## What translation can and cannot do

- **Query-side** (new queries → old index) keeps ~98 % of recall with a small
  adapter but caps quality at the old model.
- **Document-side** (old vectors → new space, native new queries) recovers only
  part of the new model's gain; a translator cannot add information the old
  encoder never captured.
- Fit on a sample of the corpus you will translate. Generic translators degrade
  off-distribution.
- Do not mix translated and natively embedded vectors in one index without
  checking that score distributions match.

See [`research/Embedding translation feasibility study.md`](research/Embedding%20translation%20feasibility%20study.md)
for the sources behind these statements.

## Layout

- `crates/vecski-core` — matrix type, fitting (`fit.rs`), the fused translator
  (`translator.rs`), evaluation (`eval.rs`), safetensors I/O (`io.rs`).
- `crates/vecski-server` — axum server: handlers in `src/api/translators.rs`,
  discovery in `src/api/docs.rs`, persistence in `src/state.rs`.
- `examples/quickstart.py` — Python client walkthrough.

```bash
cargo test --workspace
cargo clippy --workspace --all-targets
```

## License

Apache-2.0.
