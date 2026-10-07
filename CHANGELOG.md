# Changelog

## 0.1.0 — 2026-10-06

Initial release.

- Closed-form translators between embedding spaces: centered orthogonal
  Procrustes (zero-padded, optional isotropic scale) and ridge regression with a
  held-out lambda sweep; `auto` picks by held-out neighbor overlap.
- Retrieval-style evaluation: cosine next to the mean-vector baseline, top-1,
  MRR, neighbor overlap@k, pairwise-cosine RMSE, plus data-size warnings.
- HTTP API (axum): fit from JSON or multipart float32, translate via JSON or raw
  float32, evaluate, export/import safetensors, list/get/delete.
- Agent discovery: `/`, `/llms.txt`, `/openapi.json`.
- Persistence as one safetensors file per translator; optional API keys.
- Homebrew formula and GitHub release binaries for macOS and Linux.
