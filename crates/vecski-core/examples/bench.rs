//! Rough timing of fit and translate at realistic dimensions.
//! `cargo run --release -p vecski-core --example bench -- 1536 3072 10000`
use std::time::Instant;
use vecski_core::fit::FitOptions;
use vecski_core::{Matrix, Method, fit};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let d1: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1536);
    let d2: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3072);
    let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10_000);

    let mut s = 1u64;
    let mut f = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f32 / (1u64 << 53) as f32 - 0.5
    };
    let x = Matrix::new(n, d1, (0..n * d1).map(|_| f()).collect()).unwrap();
    let y = Matrix::new(n, d2, (0..n * d2).map(|_| f()).collect()).unwrap();

    for method in [Method::Procrustes, Method::Ridge] {
        let t = Instant::now();
        let (tr, rep) = fit(
            &x,
            &y,
            &FitOptions {
                method,
                ..Default::default()
            },
        )
        .unwrap();
        println!(
            "fit {method:?} {d1}->{d2} n={n}: {:?} (reported {} ms)",
            t.elapsed(),
            rep.fit_ms
        );

        for batch in [1usize, 32, 1024, 16384] {
            let xb = Matrix::new(batch, d1, (0..batch * d1).map(|_| f()).collect()).unwrap();
            let mut out = vec![0.0f32; batch * d2];
            let iters = if batch >= 1024 { 5 } else { 200 };
            let t = Instant::now();
            for _ in 0..iters {
                if batch == 1 {
                    tr.translate_one(xb.data(), &mut out).unwrap();
                } else {
                    tr.translate_into(xb.data(), &mut out).unwrap();
                }
            }
            let per = t.elapsed() / iters as u32;
            let vps = batch as f64 / per.as_secs_f64();
            println!("  translate batch={batch:>6}: {per:?} per call, {vps:.0} vectors/s");
        }
    }
}
