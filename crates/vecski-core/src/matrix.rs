//! A minimal, allocation-friendly row-major `f32` matrix used for embedding batches.

use crate::error::{CoreError, Result};
use faer::{MatMut, MatRef};

/// Row-major dense `f32` matrix. Each row is one embedding.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Matrix {
    /// Wrap an existing row-major buffer. `data.len()` must equal `rows * cols`.
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self> {
        if rows.checked_mul(cols) != Some(data.len()) {
            return Err(CoreError::Shape(format!(
                "buffer has {} values but {rows} x {cols} = {} were expected",
                data.len(),
                rows.saturating_mul(cols)
            )));
        }
        Ok(Self { rows, cols, data })
    }

    /// Zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Build from a list of equal-length rows (e.g. decoded JSON).
    pub fn from_rows(rows: &[Vec<f32>]) -> Result<Self> {
        let n = rows.len();
        if n == 0 {
            return Err(CoreError::Invalid("expected at least one vector".into()));
        }
        let d = rows[0].len();
        if d == 0 {
            return Err(CoreError::Invalid(
                "vectors must have at least one dimension".into(),
            ));
        }
        let mut data = Vec::with_capacity(n * d);
        for (i, r) in rows.iter().enumerate() {
            if r.len() != d {
                return Err(CoreError::Shape(format!(
                    "vector {i} has {} dimensions but vector 0 has {d}",
                    r.len()
                )));
            }
            data.extend_from_slice(r);
        }
        Ok(Self {
            rows: n,
            cols: d,
            data,
        })
    }

    /// Decode little-endian `f32` bytes into a matrix with `cols` columns.
    pub fn from_le_bytes(bytes: &[u8], cols: usize) -> Result<Self> {
        if cols == 0 {
            return Err(CoreError::Invalid("dimension must be positive".into()));
        }
        if !bytes.len().is_multiple_of(4) {
            return Err(CoreError::Shape(format!(
                "byte length {} is not a multiple of 4 (expected little-endian f32)",
                bytes.len()
            )));
        }
        let total = bytes.len() / 4;
        if !total.is_multiple_of(cols) {
            return Err(CoreError::Shape(format!(
                "{total} floats is not a multiple of the dimension {cols}"
            )));
        }
        if total == 0 {
            return Err(CoreError::Invalid("expected at least one vector".into()));
        }
        let data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(Self {
            rows: total / cols,
            cols,
            data,
        })
    }

    /// Encode as little-endian `f32` bytes, row-major.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.data.len() * 4);
        for v in &self.data {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// Copy the rows listed in `idx` into a new matrix.
    pub fn select(&self, idx: &[usize]) -> Self {
        let mut data = Vec::with_capacity(idx.len() * self.cols);
        for &i in idx {
            data.extend_from_slice(self.row(i));
        }
        Self {
            rows: idx.len(),
            cols: self.cols,
            data,
        }
    }

    /// Convert to nested vectors (for JSON responses).
    pub fn to_rows(&self) -> Vec<Vec<f32>> {
        self.data
            .chunks_exact(self.cols)
            .map(<[f32]>::to_vec)
            .collect()
    }

    /// Reject NaN / infinity so they cannot poison a fit.
    pub fn check_finite(&self, what: &str) -> Result<()> {
        if let Some(pos) = self.data.iter().position(|v| !v.is_finite()) {
            return Err(CoreError::Invalid(format!(
                "{what} contains a non-finite value at vector {}, dimension {}",
                pos / self.cols,
                pos % self.cols
            )));
        }
        Ok(())
    }

    /// Borrow as a faer matrix view (zero-copy).
    #[inline]
    pub fn as_faer(&self) -> MatRef<'_, f32> {
        MatRef::from_row_major_slice(&self.data, self.rows, self.cols)
    }

    /// Mutably borrow as a faer matrix view (zero-copy).
    #[inline]
    pub fn as_faer_mut(&mut self) -> MatMut<'_, f32> {
        MatMut::from_row_major_slice_mut(&mut self.data, self.rows, self.cols)
    }

    /// Per-row L2 norms.
    pub fn row_norms(&self) -> Vec<f32> {
        self.data
            .chunks_exact(self.cols)
            .map(|r| r.iter().map(|v| v * v).sum::<f32>().sqrt())
            .collect()
    }

    /// Column means accumulated in f64.
    pub fn col_means(&self) -> Vec<f64> {
        let mut acc = vec![0.0f64; self.cols];
        for r in self.data.chunks_exact(self.cols) {
            for (a, v) in acc.iter_mut().zip(r) {
                *a += *v as f64;
            }
        }
        let n = self.rows.max(1) as f64;
        acc.iter_mut().for_each(|a| *a /= n);
        acc
    }
}

/// Normalize every row of a row-major buffer to unit L2 norm in place.
/// Rows with (near) zero norm are left untouched.
pub fn normalize_rows_in_place(data: &mut [f32], cols: usize) {
    for r in data.chunks_exact_mut(cols) {
        let n = r.iter().map(|v| v * v).sum::<f32>().sqrt();
        if n > 1e-12 {
            let inv = 1.0 / n;
            r.iter_mut().for_each(|v| *v *= inv);
        }
    }
}
