//! Retrieval-oriented evaluation of a translator on held-out pairs.
//!
//! Cosine to the native target is reported *alongside* the "predict the mean
//! vector" baseline because, for text embeddings, that baseline alone often
//! scores 0.85+ and makes raw cosine look far better than it is.

use crate::error::{CoreError, Result};
use crate::matrix::{Matrix, normalize_rows_in_place};
use crate::translator::Translator;
use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Metrics comparing translated source vectors against native target vectors.
#[derive(Clone, Debug, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct EvalMetrics {
    /// Number of held-out pairs evaluated.
    pub n: usize,
    /// `k` used for neighbor overlap.
    pub k: usize,
    /// Mean cosine between `translate(x_i)` and the native `y_i`.
    pub mean_cosine: f64,
    /// Mean cosine between the fitted target mean and `y_i`. If `mean_cosine`
    /// is not well above this, the translator has learned little.
    pub mean_vector_baseline_cosine: f64,
    /// Fraction of items whose nearest native target (by cosine) is their own pair.
    pub top1_accuracy: f64,
    /// Mean reciprocal rank of the paired target among all held-out targets.
    pub mrr: f64,
    /// Mean |top-k(translated query vs native docs) ∩ top-k(native query vs native docs)| / k,
    /// self excluded. This is the number that predicts retrieval quality after migration.
    pub neighbor_overlap_at_k: f64,
    /// RMSE between pairwise cosines among translated vectors and among native
    /// targets. Measures how faithfully the translated set reproduces target geometry.
    pub pairwise_cosine_rmse: f64,
    /// Median L2 norm of translated vectors before normalization is irrelevant to
    /// retrieval when outputs are normalized; reported for score-scale checks.
    pub median_output_norm: f64,
    /// Median L2 norm of the native targets, for the same check.
    pub median_target_norm: f64,
}

/// Evaluate `tr` on held-out pairs `(x, y)`.
pub fn evaluate(tr: &Translator, x: &Matrix, y: &Matrix, k: usize) -> Result<EvalMetrics> {
    if x.rows() != y.rows() {
        return Err(CoreError::Shape(format!(
            "source has {} vectors but target has {}",
            x.rows(),
            y.rows()
        )));
    }
    if x.cols() != tr.source_dim() || y.cols() != tr.target_dim() {
        return Err(CoreError::Shape(format!(
            "pairs are {}->{} but translator is {}->{}",
            x.cols(),
            y.cols(),
            tr.source_dim(),
            tr.target_dim()
        )));
    }
    let n = x.rows();
    if n < 2 {
        return Err(CoreError::Invalid(
            "evaluation needs at least 2 pairs".into(),
        ));
    }
    let k = k.clamp(1, n - 1);
    let d = tr.target_dim();

    // Translate without normalization so output norms can be reported, then normalize copies.
    let mut raw = tr.clone();
    raw.set_normalize_output(false);
    let yhat = raw.translate(x)?;
    let median_output_norm = median(yhat.row_norms());
    let median_target_norm = median(y.row_norms());

    let mut yh = yhat.into_data();
    normalize_rows_in_place(&mut yh, d);
    let mut yn = y.data().to_vec();
    normalize_rows_in_place(&mut yn, d);

    // Mean cosine and mean-vector baseline.
    let mut mean_t = tr.target_mean().to_vec();
    crate::translator::normalize_row(&mut mean_t);
    let (sum_cos, sum_base) = yh
        .par_chunks_exact(d)
        .zip(yn.par_chunks_exact(d))
        .map(|(a, b)| (dot(a, b) as f64, dot(&mean_t, b) as f64))
        .reduce(|| (0.0, 0.0), |p, q| (p.0 + q.0, p.1 + q.1));

    let par = if n * n * d > 4_000_000 {
        Par::rayon(0)
    } else {
        Par::Seq
    };
    // S = Ŷn · Ynᵀ (translated queries vs native docs), T = Yn · Ynᵀ (native vs native),
    // P = Ŷn · Ŷnᵀ (translated vs translated).
    let s = gram(&yh, &yn, n, d, par);
    let t = gram(&yn, &yn, n, d, par);
    let p = gram(&yh, &yh, n, d, par);

    let (top1, rr, overlap) = (0..n)
        .into_par_iter()
        .map(|i| {
            let srow = &s[i * n..(i + 1) * n];
            let trow = &t[i * n..(i + 1) * n];
            // Rank of self among translated-vs-native scores.
            let own = srow[i];
            let better = srow
                .iter()
                .enumerate()
                .filter(|(j, v)| *j != i && **v > own)
                .count();
            let top1 = if better == 0 { 1.0 } else { 0.0 };
            let rr = 1.0 / (better as f64 + 1.0);
            let a = topk_excluding(srow, i, k);
            let b = topk_excluding(trow, i, k);
            let inter = a.iter().filter(|j| b.contains(j)).count();
            (top1, rr, inter as f64 / k as f64)
        })
        .reduce(|| (0.0, 0.0, 0.0), |u, v| (u.0 + v.0, u.1 + v.1, u.2 + v.2));

    // Pairwise cosine RMSE over i<j.
    let sq: f64 = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut acc = 0.0f64;
            for j in (i + 1)..n {
                let dlt = (p[i * n + j] - t[i * n + j]) as f64;
                acc += dlt * dlt;
            }
            acc
        })
        .sum();
    let pairs = (n * (n - 1) / 2) as f64;

    Ok(EvalMetrics {
        n,
        k,
        mean_cosine: sum_cos / n as f64,
        mean_vector_baseline_cosine: sum_base / n as f64,
        top1_accuracy: top1 / n as f64,
        mrr: rr / n as f64,
        neighbor_overlap_at_k: overlap / n as f64,
        pairwise_cosine_rmse: (sq / pairs).sqrt(),
        median_output_norm,
        median_target_norm,
    })
}

/// `A · Bᵀ` for two row-major `n x d` buffers, returned row-major `n x n`.
fn gram(a: &[f32], b: &[f32], n: usize, d: usize, par: Par) -> Vec<f32> {
    let mut out = vec![0.0f32; n * n];
    let am = MatRef::from_row_major_slice(a, n, d);
    let bm = MatRef::from_row_major_slice(b, n, d);
    let om = MatMut::from_row_major_slice_mut(&mut out, n, n);
    matmul(om, Accum::Replace, am, bm.transpose(), 1.0f32, par);
    out
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Indices of the `k` largest entries of `row`, excluding index `skip`.
fn topk_excluding(row: &[f32], skip: usize, k: usize) -> Vec<usize> {
    // Partial selection: keep a small sorted buffer; k is tiny relative to n.
    let mut best: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    for (j, &v) in row.iter().enumerate() {
        if j == skip {
            continue;
        }
        if best.len() < k || v > best[best.len() - 1].0 {
            let pos = best.partition_point(|(bv, _)| *bv > v);
            best.insert(pos, (v, j));
            if best.len() > k {
                best.pop();
            }
        }
    }
    best.into_iter().map(|(_, j)| j).collect()
}

fn median(mut v: Vec<f32>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2] as f64
}
