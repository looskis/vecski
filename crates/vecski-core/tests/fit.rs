use vecski_core::fit::FitOptions;
use vecski_core::matrix::normalize_rows_in_place;
use vecski_core::{Matrix, Method, evaluate, fit, io};

/// splitmix64-based deterministic pseudo-random floats in [-1, 1).
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn f(&mut self) -> f32 {
        (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
    }
    /// Approximately Gaussian via sum of uniforms.
    fn g(&mut self) -> f32 {
        (0..4).map(|_| self.f()).sum::<f32>() * 0.5
    }
}

/// Random orthogonal matrix via Gram-Schmidt.
fn random_orthogonal(d: usize, rng: &mut Rng) -> Vec<Vec<f32>> {
    let mut q: Vec<Vec<f32>> = Vec::with_capacity(d);
    for _ in 0..d {
        let mut v: Vec<f32> = (0..d).map(|_| rng.g()).collect();
        for u in &q {
            let p: f32 = v.iter().zip(u).map(|(a, b)| a * b).sum();
            v.iter_mut().zip(u).for_each(|(a, b)| *a -= p * b);
        }
        let n = v.iter().map(|a| a * a).sum::<f32>().sqrt();
        v.iter_mut().for_each(|a| *a /= n);
        q.push(v);
    }
    q
}

/// Build (source, target) where target = normalize(scale * (source - mu) @ Q[:d1,:d2] + shift + noise).
fn synthetic(n: usize, d1: usize, d2: usize, noise: f32, seed: u64) -> (Matrix, Matrix) {
    let mut rng = Rng(seed);
    let d = d1.max(d2);
    let q = random_orthogonal(d, &mut rng);
    let mu: Vec<f32> = (0..d1).map(|_| 0.1 * rng.g()).collect();
    let shift: Vec<f32> = (0..d2).map(|_| 0.05 * rng.g()).collect();
    let mut xs = Vec::with_capacity(n * d1);
    let mut ys = Vec::with_capacity(n * d2);
    for _ in 0..n {
        let x: Vec<f32> = (0..d1).map(|i| rng.g() + mu[i]).collect();
        let mut y = vec![0.0f32; d2];
        for (i, xi) in x.iter().enumerate() {
            let xc = xi - mu[i];
            for j in 0..d2 {
                y[j] += 1.7 * xc * q[i][j];
            }
        }
        for j in 0..d2 {
            y[j] += shift[j] + noise * rng.g();
        }
        xs.extend_from_slice(&x);
        ys.extend_from_slice(&y);
    }
    normalize_rows_in_place(&mut xs, d1);
    normalize_rows_in_place(&mut ys, d2);
    (
        Matrix::new(n, d1, xs).unwrap(),
        Matrix::new(n, d2, ys).unwrap(),
    )
}

#[test]
fn procrustes_recovers_a_rotation() {
    let (x, y) = synthetic(3000, 64, 64, 0.0, 1);
    let opts = FitOptions {
        method: Method::Procrustes,
        ..Default::default()
    };
    let (tr, rep) = fit(&x, &y, &opts).unwrap();
    assert_eq!(rep.method, Method::Procrustes);
    assert_eq!(rep.n_holdout, 300);
    let h = rep.holdout.as_ref().unwrap();
    assert!(h.mean_cosine > 0.99, "cosine {}", h.mean_cosine);
    assert!(h.top1_accuracy > 0.99, "top1 {}", h.top1_accuracy);
    assert!(
        h.neighbor_overlap_at_k > 0.95,
        "overlap {}",
        h.neighbor_overlap_at_k
    );
    assert!(h.mean_vector_baseline_cosine < 0.9);
    assert!(rep.normalize_output && rep.targets_unit_norm);
    // single-vector and batch paths agree
    let mut one = vec![0.0; 64];
    tr.translate_one(x.row(0), &mut one).unwrap();
    let batch = tr.translate(&x).unwrap();
    for (a, b) in one.iter().zip(batch.row(0)) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn handles_dimension_mismatch_both_ways() {
    for (d1, d2) in [(48, 96), (96, 48)] {
        let (x, y) = synthetic(2500, d1, d2, 0.01, 7);
        let (tr, rep) = fit(&x, &y, &FitOptions::default()).unwrap();
        assert_eq!((tr.source_dim(), tr.target_dim()), (d1, d2));
        let h = rep.holdout.as_ref().unwrap();
        assert!(h.mean_cosine > 0.9, "{d1}->{d2} cosine {}", h.mean_cosine);
        assert!(h.top1_accuracy > 0.9, "{d1}->{d2} top1 {}", h.top1_accuracy);
    }
}

#[test]
fn ridge_fits_and_sweeps_lambda() {
    let (x, y) = synthetic(3000, 32, 32, 0.02, 3);
    let opts = FitOptions {
        method: Method::Ridge,
        ..Default::default()
    };
    let (_tr, rep) = fit(&x, &y, &opts).unwrap();
    assert_eq!(rep.method, Method::Ridge);
    assert!(rep.lambda.unwrap() > 0.0);
    assert_eq!(rep.lambda_sweep.as_ref().unwrap().len(), 5);
    assert!(rep.holdout.unwrap().mean_cosine > 0.98);
}

#[test]
fn auto_reports_both_candidates() {
    let (x, y) = synthetic(2000, 32, 32, 0.05, 11);
    let (_tr, rep) = fit(&x, &y, &FitOptions::default()).unwrap();
    assert_eq!(rep.requested_method, Method::Auto);
    assert_eq!(rep.candidates.len(), 2);
    assert!(matches!(rep.method, Method::Procrustes | Method::Ridge));
}

#[test]
fn warns_when_underdetermined_and_without_holdout() {
    let (x, y) = synthetic(40, 64, 64, 0.0, 5);
    let opts = FitOptions {
        method: Method::Procrustes,
        ..Default::default()
    };
    let (_tr, rep) = fit(&x, &y, &opts).unwrap();
    assert!(rep.holdout.is_none());
    assert!(rep.warnings.iter().any(|w| w.contains("under-determined")));
    assert!(rep.warnings.iter().any(|w| w.contains("no holdout")));
}

#[test]
fn rejects_bad_input() {
    let x = Matrix::new(3, 4, vec![0.0; 12]).unwrap();
    let y = Matrix::new(2, 4, vec![0.0; 8]).unwrap();
    assert!(fit(&x, &y, &FitOptions::default()).is_err());
    let bad = Matrix::new(3, 4, vec![f32::NAN; 12]).unwrap();
    let y3 = Matrix::new(3, 4, vec![0.0; 12]).unwrap();
    assert!(fit(&bad, &y3, &FitOptions::default()).is_err());
    assert!(Matrix::from_le_bytes(&[0u8; 10], 4).is_err());
    assert!(Matrix::from_rows(&[vec![1.0, 2.0], vec![1.0]]).is_err());
}

#[test]
fn safetensors_round_trip_is_exact() {
    let (x, y) = synthetic(1500, 24, 40, 0.01, 9);
    let (tr, _) = fit(&x, &y, &FitOptions::default()).unwrap();
    let mut extra = std::collections::HashMap::new();
    extra.insert("name".to_string(), "test".to_string());
    let bytes = io::to_safetensors(&tr, &extra).unwrap();
    let (back, meta) = io::from_safetensors(&bytes).unwrap();
    assert_eq!(meta.get("name").unwrap(), "test");
    assert_eq!(meta.get("format").unwrap(), io::FORMAT);
    assert_eq!(back.weight(), tr.weight());
    assert_eq!(back.bias(), tr.bias());
    assert_eq!(back.normalize_output(), tr.normalize_output());
    assert_eq!(back.info().method, tr.info().method);
    let a = tr.translate(&x).unwrap();
    let b = back.translate(&x).unwrap();
    assert_eq!(a, b);
}

#[test]
fn evaluate_identity_is_perfect() {
    let (x, _) = synthetic(300, 16, 16, 0.0, 2);
    let d = 16;
    let mut w = vec![0.0f32; d * d];
    for i in 0..d {
        w[i * d + i] = 1.0;
    }
    let tr = vecski_core::Translator::new(
        d,
        d,
        w,
        vec![0.0; d],
        vec![0.0; d],
        true,
        Default::default(),
    )
    .unwrap();
    let m = evaluate(&tr, &x, &x, 10).unwrap();
    assert!((m.mean_cosine - 1.0).abs() < 1e-5);
    assert_eq!(m.top1_accuracy, 1.0);
    assert_eq!(m.neighbor_overlap_at_k, 1.0);
    assert!(m.pairwise_cosine_rmse < 1e-5);
}
