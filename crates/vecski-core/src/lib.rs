//! # vecski-core
//!
//! Closed-form translators between embedding spaces, built for the case where
//! you have a few thousand texts embedded by both an old and a new model and
//! want to map the old vectors (or new queries) across without re-embedding.
//!
//! * [`fit::fit`] — centered orthogonal Procrustes (SVD) or ridge regression,
//!   with zero-padding for mismatched dimensions and an automatic holdout.
//! * [`Translator`] — the fused affine map; one GEMM per batch.
//! * [`eval::evaluate`] — cosine, mean-vector baseline, top-1, MRR, neighbor
//!   overlap@k, pairwise-cosine RMSE.
//! * [`io`] — safetensors round-trip.

pub mod error;
pub mod eval;
pub mod fit;
pub mod io;
pub mod matrix;
pub mod translator;

pub use error::{CoreError, Result};
pub use eval::{EvalMetrics, evaluate};
pub use fit::{FitOptions, FitReport, fit};
pub use matrix::Matrix;
pub use translator::{Method, Translator, TranslatorInfo};

/// Configure faer's global thread pool size (0 = all cores).
pub fn set_threads(n: usize) {
    if n == 1 {
        faer::set_global_parallelism(faer::Par::Seq);
    } else {
        faer::set_global_parallelism(faer::Par::rayon(n));
    }
}
