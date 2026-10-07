//! The fitted map. Every estimator in vecski collapses to a single affine map
//! `y = x · W + b` (optionally followed by L2 normalization), so the hot path is
//! one GEMM plus a cheap per-row pass regardless of how the map was fitted.

use crate::error::{CoreError, Result};
use crate::matrix::Matrix;
use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Which closed-form estimator produced the map.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    /// Centered orthogonal Procrustes (SVD of the cross-covariance), zero-padded
    /// when the dimensions differ, with an optional isotropic scale.
    /// Preserves the geometry of the source space exactly.
    Procrustes,
    /// Centered ridge regression (closed form via eigendecomposition of the Gram
    /// matrix). Lower cross-space error, but distorts the source geometry.
    Ridge,
    /// Fit both and keep whichever scores higher on held-out neighbor overlap.
    Auto,
}

/// Facts about how a translator was fitted that are needed to interpret it.
#[derive(Clone, Debug, Serialize, Deserialize, ToSchema, Default)]
pub struct TranslatorInfo {
    /// Estimator that produced the weights (never `auto`; that resolves to one of the others).
    pub method: Option<Method>,
    /// Isotropic scale applied by Procrustes (1.0 when disabled).
    pub scale: Option<f64>,
    /// Ridge penalty actually used (absolute, not relative).
    pub lambda: Option<f64>,
    /// Number of pairs the map was fitted on (excluding holdout).
    pub n_train: Option<usize>,
}

/// A fitted affine translator between two embedding spaces.
#[derive(Clone, Debug)]
pub struct Translator {
    source_dim: usize,
    target_dim: usize,
    /// Row-major `source_dim x target_dim`.
    weight: Vec<f32>,
    /// `target_dim`.
    bias: Vec<f32>,
    /// Mean of the target vectors seen during fitting; the "predict the mean" baseline.
    target_mean: Vec<f32>,
    normalize_output: bool,
    info: TranslatorInfo,
}

impl Translator {
    pub fn new(
        source_dim: usize,
        target_dim: usize,
        weight: Vec<f32>,
        bias: Vec<f32>,
        target_mean: Vec<f32>,
        normalize_output: bool,
        info: TranslatorInfo,
    ) -> Result<Self> {
        if source_dim == 0 || target_dim == 0 {
            return Err(CoreError::Invalid("dimensions must be positive".into()));
        }
        if weight.len() != source_dim * target_dim {
            return Err(CoreError::Shape(format!(
                "weight has {} values, expected {source_dim} x {target_dim}",
                weight.len()
            )));
        }
        if bias.len() != target_dim || target_mean.len() != target_dim {
            return Err(CoreError::Shape(
                "bias / target_mean must have target_dim values".into(),
            ));
        }
        Ok(Self {
            source_dim,
            target_dim,
            weight,
            bias,
            target_mean,
            normalize_output,
            info,
        })
    }

    #[inline]
    pub fn source_dim(&self) -> usize {
        self.source_dim
    }
    #[inline]
    pub fn target_dim(&self) -> usize {
        self.target_dim
    }
    pub fn weight(&self) -> &[f32] {
        &self.weight
    }
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }
    pub fn target_mean(&self) -> &[f32] {
        &self.target_mean
    }
    pub fn normalize_output(&self) -> bool {
        self.normalize_output
    }
    pub fn set_normalize_output(&mut self, v: bool) {
        self.normalize_output = v;
    }
    pub fn info(&self) -> &TranslatorInfo {
        &self.info
    }
    pub fn info_mut(&mut self) -> &mut TranslatorInfo {
        &mut self.info
    }

    /// Size of the weights in bytes (what gets shipped around).
    pub fn weight_bytes(&self) -> usize {
        (self.weight.len() + self.bias.len() + self.target_mean.len()) * 4
    }

    /// Pick a parallelism level for a batch of `n` rows. Small batches stay on
    /// the calling thread: thread hand-off costs more than the GEMM itself.
    #[inline]
    pub fn par_for(&self, n: usize) -> Par {
        let flops = n * self.source_dim * self.target_dim;
        if flops < 8_000_000 {
            Par::Seq
        } else {
            Par::rayon(0)
        }
    }

    /// Translate `n` row-major vectors in `x` (length `n * source_dim`) into
    /// `out` (length `n * target_dim`).
    pub fn translate_into(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        let n = self.check_batch(x.len(), out.len())?;
        if n == 0 {
            return Ok(());
        }
        let par = self.par_for(n);
        let xm = MatRef::from_row_major_slice(x, n, self.source_dim);
        let wm = MatRef::from_row_major_slice(&self.weight, self.source_dim, self.target_dim);
        let om = MatMut::from_row_major_slice_mut(out, n, self.target_dim);
        matmul(om, Accum::Replace, xm, wm, 1.0f32, par);
        self.finish_rows(out, par);
        Ok(())
    }

    /// Translate a batch and return a freshly allocated matrix.
    pub fn translate(&self, x: &Matrix) -> Result<Matrix> {
        if x.cols() != self.source_dim {
            return Err(CoreError::Shape(format!(
                "input vectors have {} dimensions but this translator expects {}",
                x.cols(),
                self.source_dim
            )));
        }
        let mut out = vec![0.0f32; x.rows() * self.target_dim];
        self.translate_into(x.data(), &mut out)?;
        Matrix::new(x.rows(), self.target_dim, out)
    }

    /// Translate one vector (latency path: no thread hand-off, no faer dispatch).
    pub fn translate_one(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        self.check_batch(x.len(), out.len())?;
        out.copy_from_slice(&self.bias);
        for (xi, wrow) in x.iter().zip(self.weight.chunks_exact(self.target_dim)) {
            let xi = *xi;
            if xi != 0.0 {
                for (o, w) in out.iter_mut().zip(wrow) {
                    *o += xi * w;
                }
            }
        }
        if self.normalize_output {
            normalize_row(out);
        }
        Ok(())
    }

    fn check_batch(&self, x_len: usize, out_len: usize) -> Result<usize> {
        if !x_len.is_multiple_of(self.source_dim) {
            return Err(CoreError::Shape(format!(
                "input length {x_len} is not a multiple of source_dim {}",
                self.source_dim
            )));
        }
        let n = x_len / self.source_dim;
        if out_len != n * self.target_dim {
            return Err(CoreError::Shape(format!(
                "output buffer has {out_len} values, expected {} ({n} x {})",
                n * self.target_dim,
                self.target_dim
            )));
        }
        Ok(n)
    }

    #[inline]
    fn finish_rows(&self, out: &mut [f32], par: Par) {
        let d = self.target_dim;
        let bias = &self.bias;
        let norm = self.normalize_output;
        let f = |row: &mut [f32]| {
            for (o, b) in row.iter_mut().zip(bias) {
                *o += *b;
            }
            if norm {
                normalize_row(row);
            }
        };
        match par {
            Par::Seq => out.chunks_exact_mut(d).for_each(f),
            _ => out.par_chunks_exact_mut(d).for_each(f),
        }
    }
}

#[inline]
pub(crate) fn normalize_row(row: &mut [f32]) {
    let n = row.iter().map(|v| v * v).sum::<f32>().sqrt();
    if n > 1e-12 {
        let inv = 1.0 / n;
        row.iter_mut().for_each(|v| *v *= inv);
    }
}
