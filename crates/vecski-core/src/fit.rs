//! Closed-form fitting of an embedding-space translator from paired vectors.
//!
//! Recipe (consistent with the 2025 literature on cross-model alignment):
//! center both spaces, zero-pad the narrower one, solve orthogonal Procrustes by
//! SVD of the cross-covariance (optionally with an isotropic scale), or solve
//! ridge regression via the eigendecomposition of the Gram matrix, then report
//! held-out retrieval metrics next to the mean-vector baseline.

use crate::error::{CoreError, Result};
use crate::eval::{EvalMetrics, evaluate};
use crate::matrix::{Matrix, normalize_rows_in_place};
use crate::translator::{Method, Translator, TranslatorInfo};
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::{SelfAdjointEigen, Svd};
use faer::{Accum, Mat, MatRef, Par, Side};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use utoipa::ToSchema;

/// Options controlling the fit.
#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct FitOptions {
    /// Estimator. `auto` fits both and keeps the one with better held-out neighbor overlap.
    pub method: Method,
    /// Procrustes only: also fit an isotropic scale (recommended when norms differ).
    pub scale: bool,
    /// Ridge only: absolute penalty. When omitted a small grid is swept on the holdout.
    pub lambda: Option<f64>,
    /// L2-normalize translated outputs. Defaults to whether the targets are unit norm.
    pub normalize_output: Option<bool>,
    /// Fraction of pairs held out for evaluation (0 disables).
    pub holdout_fraction: f64,
    /// Upper bound on the holdout size (evaluation is O(n²·d)).
    pub max_holdout: usize,
    /// `k` for neighbor overlap.
    pub k: usize,
    /// Seed for the train/holdout shuffle.
    pub seed: u64,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            method: Method::Auto,
            scale: true,
            lambda: None,
            normalize_output: None,
            holdout_fraction: 0.1,
            max_holdout: 2048,
            k: 10,
            seed: 42,
        }
    }
}

/// One `(lambda, holdout mean cosine)` point from the ridge sweep.
#[derive(Clone, Debug, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct LambdaPoint {
    pub lambda: f64,
    pub holdout_mean_cosine: f64,
}

/// Held-out metrics for one candidate estimator (populated by `auto`).
#[derive(Clone, Debug, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct Candidate {
    pub method: Method,
    pub holdout: EvalMetrics,
}

/// Everything a caller needs to decide whether to trust the fitted map.
#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct FitReport {
    /// Estimator that produced the returned weights.
    pub method: Method,
    /// Estimator that was requested (may be `auto`).
    pub requested_method: Method,
    pub source_dim: usize,
    pub target_dim: usize,
    pub n_pairs: usize,
    pub n_train: usize,
    pub n_holdout: usize,
    /// Isotropic Procrustes scale (1.0 if disabled or ridge).
    pub scale: Option<f64>,
    /// Ridge penalty used (absolute).
    pub lambda: Option<f64>,
    /// Ridge sweep results when lambda was chosen automatically.
    pub lambda_sweep: Option<Vec<LambdaPoint>>,
    pub normalize_output: bool,
    /// Whether the target vectors were (median) unit norm.
    pub targets_unit_norm: bool,
    /// Mean cosine on (a sample of) the training pairs. Optimistic by construction.
    pub train_mean_cosine: f64,
    /// Held-out metrics for the returned translator, if a holdout existed.
    pub holdout: Option<EvalMetrics>,
    /// Held-out metrics for every candidate (only `auto` fits more than one).
    pub candidates: Vec<Candidate>,
    /// Human-readable caveats derived from the data and metrics.
    pub warnings: Vec<String>,
    pub fit_ms: u64,
}

/// Fit a translator from paired vectors: row `i` of `source` and row `i` of
/// `target` must embed the same text.
pub fn fit(source: &Matrix, target: &Matrix, opts: &FitOptions) -> Result<(Translator, FitReport)> {
    let t0 = Instant::now();
    validate(source, target, opts)?;
    let n = source.rows();
    let (d1, d2) = (source.cols(), target.cols());
    let dmax = d1.max(d2);

    // Train / holdout split.
    let (train_idx, hold_idx) = split(n, opts);
    let xt = source.select(&train_idx);
    let yt = target.select(&train_idx);
    let (xh, yh) = if hold_idx.is_empty() {
        (None, None)
    } else {
        (
            Some(source.select(&hold_idx)),
            Some(target.select(&hold_idx)),
        )
    };
    let n_train = xt.rows();

    // Output normalization policy.
    let targets_unit_norm = is_unit_norm(&yt);
    let normalize_output = opts.normalize_output.unwrap_or(targets_unit_norm);

    // Centering + second moments, accumulated in f64, zero-padded to dmax.
    let mu_x = xt.col_means();
    let mu_y = yt.col_means();
    let want_gram = matches!(opts.method, Method::Ridge | Method::Auto);
    let mom = moments(&xt, &yt, &mu_x, &mu_y, dmax, want_gram);
    let mean_y: Vec<f32> = mu_y.iter().map(|v| *v as f32).collect();

    let mut warnings = Vec::new();
    let mut candidates = Vec::new();
    let mut lambda_sweep = None;

    let build = |w: &Mat<f64>, info: TranslatorInfo| -> Result<Translator> {
        assemble(
            w,
            &mu_x,
            &mu_y,
            d1,
            d2,
            &mean_y,
            normalize_output,
            info,
            n_train,
        )
    };

    // --- Procrustes -------------------------------------------------------
    let procrustes = if matches!(opts.method, Method::Procrustes | Method::Auto) {
        let (q, scale) = solve_procrustes(&mom.cross, mom.x_fro2, opts.scale)?;
        let tr = build(
            &q,
            TranslatorInfo {
                method: Some(Method::Procrustes),
                scale: Some(scale),
                lambda: None,
                n_train: None,
            },
        )?;
        Some((tr, scale))
    } else {
        None
    };

    // --- Ridge -------------------------------------------------------------
    let ridge = if matches!(opts.method, Method::Ridge | Method::Auto) {
        let gram = mom.gram.as_ref().expect("gram requested");
        let solver = RidgeSolver::new(
            gram.as_ref().submatrix(0, 0, d1, d1),
            mom.cross.as_ref().submatrix(0, 0, d1, d2),
        )?;
        let mean_eig = solver.mean_eigenvalue();
        let (lambda, sweep) = match (opts.lambda, &xh, &yh) {
            (Some(l), _, _) => (l, None),
            (None, Some(xh), Some(yh)) => {
                let mut pts = Vec::new();
                let mut best = (f64::NEG_INFINITY, 0.0);
                for f in [1e-4, 1e-3, 1e-2, 1e-1, 1.0] {
                    let l = f * mean_eig;
                    let w = solver.solve(l);
                    let tr = build(&w, TranslatorInfo::default())?;
                    let c = mean_cosine(&tr, xh, yh)?;
                    pts.push(LambdaPoint {
                        lambda: l,
                        holdout_mean_cosine: c,
                    });
                    if c > best.0 {
                        best = (c, l);
                    }
                }
                (best.1, Some(pts))
            }
            (None, _, _) => {
                warnings.push(
                    "ridge: no holdout available to choose lambda; using 1e-2 x mean eigenvalue"
                        .into(),
                );
                (1e-2 * mean_eig, None)
            }
        };
        lambda_sweep = sweep;
        let w = solver.solve(lambda);
        let tr = build(
            &w,
            TranslatorInfo {
                method: Some(Method::Ridge),
                scale: None,
                lambda: Some(lambda),
                n_train: None,
            },
        )?;
        Some((tr, lambda))
    } else {
        None
    };

    // --- Choose --------------------------------------------------------------
    let (translator, method) = match (procrustes, ridge) {
        (Some((p, _)), None) => (p, Method::Procrustes),
        (None, Some((r, _))) => (r, Method::Ridge),
        (Some((p, _)), Some((r, _))) => {
            match (&xh, &yh) {
                (Some(xh), Some(yh)) => {
                    let ep = evaluate(&p, xh, yh, opts.k)?;
                    let er = evaluate(&r, xh, yh, opts.k)?;
                    candidates.push(Candidate {
                        method: Method::Procrustes,
                        holdout: ep.clone(),
                    });
                    candidates.push(Candidate {
                        method: Method::Ridge,
                        holdout: er.clone(),
                    });
                    // Prefer orthogonal unless ridge is clearly better on neighbor overlap.
                    if er.neighbor_overlap_at_k > ep.neighbor_overlap_at_k + 0.005 {
                        (r, Method::Ridge)
                    } else {
                        (p, Method::Procrustes)
                    }
                }
                _ => {
                    warnings.push("auto: no holdout available; defaulting to procrustes".into());
                    (p, Method::Procrustes)
                }
            }
        }
        (None, None) => unreachable!("method resolves to at least one estimator"),
    };

    // --- Report ---------------------------------------------------------------
    let holdout = match (&xh, &yh) {
        (Some(xh), Some(yh)) => Some(
            candidates
                .iter()
                .find(|c| c.method == method)
                .map(|c| c.holdout.clone())
                .map_or_else(|| evaluate(&translator, xh, yh, opts.k), Ok)?,
        ),
        _ => None,
    };
    let train_sample = sample_indices(n_train, 4096, opts.seed ^ 0x9e37_79b9);
    let train_mean_cosine = mean_cosine(
        &translator,
        &xt.select(&train_sample),
        &yt.select(&train_sample),
    )?;

    if n_train < dmax {
        warnings.push(format!(
            "only {n_train} training pairs for {dmax}-dimensional spaces (fewer than the dimension): the map is under-determined and will not generalize; aim for at least {} pairs",
            10 * dmax
        ));
    } else if n_train < 10 * dmax {
        warnings.push(format!(
            "{n_train} training pairs is below the ~10-20x dimension regime where alignment quality saturates (about {}-{} pairs)",
            10 * dmax,
            20 * dmax
        ));
    }
    if holdout.is_none() {
        warnings.push("no holdout split (too few pairs); only training-set cosine is reported, which is optimistic".into());
    }
    if let Some(h) = &holdout {
        if h.mean_cosine - h.mean_vector_baseline_cosine < 0.05 {
            warnings.push(format!(
                "held-out cosine {:.3} barely beats the mean-vector baseline {:.3}: the translator carries little information",
                h.mean_cosine, h.mean_vector_baseline_cosine
            ));
        }
        if h.neighbor_overlap_at_k < 0.5 {
            warnings.push(format!(
                "held-out neighbor overlap@{} is {:.2}; expect substantial retrieval degradation versus re-embedding",
                h.k, h.neighbor_overlap_at_k
            ));
        }
    }
    if normalize_output && !targets_unit_norm {
        warnings.push("outputs are L2-normalized but the target vectors are not unit norm; disable normalize_output if the target model's norms carry information".into());
    }

    let (scale, lambda) = (translator.info().scale, translator.info().lambda);
    let report = FitReport {
        method,
        requested_method: opts.method,
        source_dim: d1,
        target_dim: d2,
        n_pairs: n,
        n_train,
        n_holdout: hold_idx.len(),
        scale,
        lambda,
        lambda_sweep,
        normalize_output,
        targets_unit_norm,
        train_mean_cosine,
        holdout,
        candidates,
        warnings,
        fit_ms: t0.elapsed().as_millis() as u64,
    };
    Ok((translator, report))
}

fn validate(source: &Matrix, target: &Matrix, opts: &FitOptions) -> Result<()> {
    if source.rows() != target.rows() {
        return Err(CoreError::Shape(format!(
            "source has {} vectors but target has {}; pairs must align row by row",
            source.rows(),
            target.rows()
        )));
    }
    if source.rows() < 2 {
        return Err(CoreError::Invalid("need at least 2 pairs to fit".into()));
    }
    source.check_finite("source")?;
    target.check_finite("target")?;
    if !(0.0..=0.5).contains(&opts.holdout_fraction) {
        return Err(CoreError::Invalid(
            "holdout_fraction must be within [0, 0.5]".into(),
        ));
    }
    if opts.k == 0 {
        return Err(CoreError::Invalid("k must be positive".into()));
    }
    if let Some(l) = opts.lambda
        && !(l.is_finite() && l >= 0.0)
    {
        return Err(CoreError::Invalid(
            "lambda must be a finite non-negative number".into(),
        ));
    }
    Ok(())
}

/// Deterministic shuffle (splitmix64 + Fisher-Yates); no external RNG needed.
fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut next = move || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    for i in (1..n).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        idx.swap(i, j);
    }
    idx
}

fn sample_indices(n: usize, max: usize, seed: u64) -> Vec<usize> {
    if n <= max {
        (0..n).collect()
    } else {
        let mut s = shuffled(n, seed);
        s.truncate(max);
        s
    }
}

fn split(n: usize, opts: &FitOptions) -> (Vec<usize>, Vec<usize>) {
    let mut h = ((n as f64) * opts.holdout_fraction).floor() as usize;
    h = h.min(opts.max_holdout);
    if h < 20 || n - h < 2 {
        h = 0;
    }
    if h == 0 {
        return ((0..n).collect(), Vec::new());
    }
    let idx = shuffled(n, opts.seed);
    let (hold, train) = idx.split_at(h);
    (train.to_vec(), hold.to_vec())
}

fn is_unit_norm(y: &Matrix) -> bool {
    let mut dev: Vec<f32> = y.row_norms().into_iter().map(|n| (n - 1.0).abs()).collect();
    dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    dev[dev.len() / 2] < 1e-3
}

struct Moments {
    /// `Xcᵀ · Yc`, zero-padded to `dmax x dmax`.
    cross: Mat<f64>,
    /// `Xcᵀ · Xc`, zero-padded to `dmax x dmax` (only when requested).
    gram: Option<Mat<f64>>,
    /// `‖Xc‖²_F`.
    x_fro2: f64,
}

/// Accumulate centered second moments in f64 without materializing the full
/// f64 copies of the inputs (chunked GEMMs).
fn moments(
    x: &Matrix,
    y: &Matrix,
    mu_x: &[f64],
    mu_y: &[f64],
    dmax: usize,
    want_gram: bool,
) -> Moments {
    let n = x.rows();
    let (d1, d2) = (x.cols(), y.cols());
    let par = Par::rayon(0);
    let mut cross = Mat::<f64>::zeros(dmax, dmax);
    let mut gram = want_gram.then(|| Mat::<f64>::zeros(dmax, dmax));
    let mut x_fro2 = 0.0f64;
    let chunk = (1 << 22) / dmax.max(1); // ~32 MB of f64 per chunk matrix
    let chunk = chunk.clamp(64, 8192);
    let mut start = 0;
    while start < n {
        let rows = (n - start).min(chunk);
        let xc = Mat::<f64>::from_fn(rows, dmax, |i, j| {
            if j < d1 {
                x.row(start + i)[j] as f64 - mu_x[j]
            } else {
                0.0
            }
        });
        let yc = Mat::<f64>::from_fn(rows, dmax, |i, j| {
            if j < d2 {
                y.row(start + i)[j] as f64 - mu_y[j]
            } else {
                0.0
            }
        });
        matmul(
            cross.as_mut(),
            Accum::Add,
            xc.transpose(),
            yc.as_ref(),
            1.0f64,
            par,
        );
        if let Some(g) = gram.as_mut() {
            matmul(
                g.as_mut(),
                Accum::Add,
                xc.transpose(),
                xc.as_ref(),
                1.0f64,
                par,
            );
        }
        for j in 0..dmax {
            x_fro2 += xc.col_as_slice(j).iter().map(|v| v * v).sum::<f64>();
        }
        start += rows;
    }
    Moments {
        cross,
        gram,
        x_fro2,
    }
}

/// Orthogonal Procrustes: `Q = U Vᵀ` from `SVD(Xcᵀ Yc) = U Σ Vᵀ`; optional
/// isotropic scale `s = tr(Σ) / ‖Xc‖²_F`.
fn solve_procrustes(cross: &Mat<f64>, x_fro2: f64, with_scale: bool) -> Result<(Mat<f64>, f64)> {
    let d = cross.nrows();
    let svd = Svd::new(cross.as_ref()).map_err(|e| CoreError::Linalg(format!("svd: {e:?}")))?;
    let mut q = Mat::<f64>::zeros(d, d);
    matmul(
        q.as_mut(),
        Accum::Replace,
        svd.U(),
        svd.V().transpose(),
        1.0f64,
        Par::rayon(0),
    );
    let scale = if with_scale && x_fro2 > 0.0 {
        let trace: f64 = svd.S().column_vector().iter().sum();
        trace / x_fro2
    } else {
        1.0
    };
    if scale != 1.0 {
        for j in 0..d {
            q.col_as_slice_mut(j).iter_mut().for_each(|v| *v *= scale);
        }
    }
    Ok((q, scale))
}

/// Ridge via eigendecomposition so that any number of penalties can be solved
/// with two GEMMs each: `W_λ = P diag(1/(e_i+λ)) Pᵀ C`.
struct RidgeSolver {
    p: Mat<f64>,
    eig: Vec<f64>,
    ptc: Mat<f64>,
}

impl RidgeSolver {
    fn new(gram: MatRef<'_, f64>, cross: MatRef<'_, f64>) -> Result<Self> {
        let d1 = gram.nrows();
        let d2 = cross.ncols();
        let e = SelfAdjointEigen::new(gram, Side::Lower)
            .map_err(|e| CoreError::Linalg(format!("eigendecomposition: {e:?}")))?;
        let p = e.U().to_owned();
        let eig: Vec<f64> = e.S().column_vector().iter().map(|v| v.max(0.0)).collect();
        let mut ptc = Mat::<f64>::zeros(d1, d2);
        matmul(
            ptc.as_mut(),
            Accum::Replace,
            p.transpose(),
            cross,
            1.0f64,
            Par::rayon(0),
        );
        Ok(Self { p, eig, ptc })
    }

    fn mean_eigenvalue(&self) -> f64 {
        let s: f64 = self.eig.iter().sum();
        (s / self.eig.len().max(1) as f64).max(f64::MIN_POSITIVE)
    }

    fn solve(&self, lambda: f64) -> Mat<f64> {
        let (d1, d2) = (self.ptc.nrows(), self.ptc.ncols());
        let mut scaled = Mat::<f64>::zeros(d1, d2);
        for j in 0..d2 {
            let src = self.ptc.col_as_slice(j);
            let dst = scaled.col_as_slice_mut(j);
            for i in 0..d1 {
                dst[i] = src[i] / (self.eig[i] + lambda);
            }
        }
        let mut w = Mat::<f64>::zeros(d1, d2);
        matmul(
            w.as_mut(),
            Accum::Replace,
            self.p.as_ref(),
            scaled.as_ref(),
            1.0f64,
            Par::rayon(0),
        );
        w
    }
}

/// Turn a (possibly padded) f64 linear map plus the centering means into the
/// fused affine translator `y = x·W + b`, with `b = μ_y − μ_x·W`.
#[allow(clippy::too_many_arguments)]
fn assemble(
    w: &Mat<f64>,
    mu_x: &[f64],
    mu_y: &[f64],
    d1: usize,
    d2: usize,
    mean_y: &[f32],
    normalize_output: bool,
    mut info: TranslatorInfo,
    n_train: usize,
) -> Result<Translator> {
    let mut weight = vec![0.0f32; d1 * d2];
    let mut bias = vec![0.0f32; d2];
    for j in 0..d2 {
        let col = w.col_as_slice(j);
        let mut b = mu_y[j];
        for i in 0..d1 {
            weight[i * d2 + j] = col[i] as f32;
            b -= mu_x[i] * col[i];
        }
        bias[j] = b as f32;
    }
    info.n_train = Some(n_train);
    Translator::new(
        d1,
        d2,
        weight,
        bias,
        mean_y.to_vec(),
        normalize_output,
        info,
    )
}

/// Mean cosine between translated `x` and native `y`; cheap (no n×n grams).
pub fn mean_cosine(tr: &Translator, x: &Matrix, y: &Matrix) -> Result<f64> {
    let yhat = tr.translate(x)?;
    let d = tr.target_dim();
    let mut a = yhat.into_data();
    normalize_rows_in_place(&mut a, d);
    let mut b = y.data().to_vec();
    normalize_rows_in_place(&mut b, d);
    let s: f64 = a
        .chunks_exact(d)
        .zip(b.chunks_exact(d))
        .map(|(p, q)| p.iter().zip(q).map(|(u, v)| u * v).sum::<f32>() as f64)
        .sum();
    Ok(s / x.rows().max(1) as f64)
}
