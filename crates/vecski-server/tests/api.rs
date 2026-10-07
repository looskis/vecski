use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;
use tower::ServiceExt;
use vecski_server::api::router;
use vecski_server::{AppState, Config};

fn temp_dir(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("vecski-test-{tag}-{nanos}"));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn app(data_dir: PathBuf, keys: Vec<&str>) -> (Router, Arc<AppState>) {
    let st = Arc::new(AppState::new(Config {
        data_dir,
        api_keys: keys.into_iter().map(String::from).collect(),
        max_body_bytes: 64 << 20,
        public_url: None,
    }));
    st.load_all().unwrap();
    (router(st.clone()), st)
}

/// Pairs related by a fixed rotation+shift in 2D-blocks so the fit is exact.
fn pairs(n: usize, d: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut s = seed;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f32 / (1u64 << 53) as f32 - 0.5
    };
    let (c, sn) = (0.6f32, 0.8f32);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for _ in 0..n {
        let x: Vec<f32> = (0..d).map(|_| rnd()).collect();
        let mut y = vec![0.0; d];
        for k in (0..d).step_by(2) {
            y[k] = c * x[k] - sn * x[k + 1] + 0.01;
            y[k + 1] = sn * x[k] + c * x[k + 1] - 0.02;
        }
        let norm = |v: &mut Vec<f32>| {
            let n = v.iter().map(|a| a * a).sum::<f32>().sqrt();
            v.iter_mut().for_each(|a| *a /= n);
        };
        let mut x = x;
        norm(&mut x);
        norm(&mut y);
        xs.push(x);
        ys.push(y);
    }
    (xs, ys)
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, Vec<u8>, axum::http::HeaderMap) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let body = resp
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes()
        .to_vec();
    (status, body, headers)
}

async fn send_json(app: &Router, req: Request<Body>) -> (StatusCode, Value) {
    let (s, b, _) = send(app, req).await;
    let v: Value =
        serde_json::from_slice(&b).unwrap_or_else(|_| json!({"raw": String::from_utf8_lossy(&b)}));
    (s, v)
}

fn json_req(method: &str, uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

#[tokio::test]
async fn full_lifecycle_json_and_binary() {
    let dir = temp_dir("lifecycle");
    let (app, _st) = app(dir.clone(), vec![]);

    // Discovery endpoints.
    let (s, v) = send_json(&app, Request::get("/").body(Body::empty()).unwrap()).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["name"], "vecski");
    let (s, v) = send_json(
        &app,
        Request::get("/openapi.json").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert!(v["paths"]["/v1/translators/{id}/translate"].is_object());
    let (s, b, h) = send(&app, Request::get("/llms.txt").body(Body::empty()).unwrap()).await;
    assert_eq!(s, StatusCode::OK);
    assert!(
        h[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("text/markdown")
    );
    assert!(String::from_utf8_lossy(&b).contains("/v1/translators"));

    // Fit.
    let (src, tgt) = pairs(600, 8, 99);
    let (s, v) = send_json(
        &app,
        json_req(
            "POST",
            "/v1/translators",
            json!({"name": "demo", "source_model": "a", "target_model": "b",
                   "options": {"method": "procrustes"}, "source": src, "target": tgt}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    let id = v["id"].as_str().unwrap().to_string();
    assert!(id.starts_with("tr_"));
    assert_eq!(v["name"], "demo");
    assert_eq!(v["method"], "procrustes");
    assert_eq!(v["source_dim"], 8);
    assert!(v["holdout"]["mean_cosine"].as_f64().unwrap() > 0.99, "{v}");
    assert!(
        v["report"]["holdout"]["neighbor_overlap_at_k"]
            .as_f64()
            .unwrap()
            > 0.9
    );
    assert!(dir.join(format!("{id}.safetensors")).exists());

    // Duplicate name -> 409.
    let (s, _) = send_json(
        &app,
        json_req(
            "POST",
            "/v1/translators",
            json!({"name": "demo", "source": src, "target": tgt}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::CONFLICT);

    // List + get by name and id.
    let (s, v) = send_json(
        &app,
        Request::get("/v1/translators").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["translators"].as_array().unwrap().len(), 1);
    let (s, v) = send_json(
        &app,
        Request::get("/v1/translators/demo")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["id"], id);
    assert!(v["report"]["n_train"].as_u64().unwrap() > 0);

    // Translate JSON.
    let (s, v) = send_json(
        &app,
        json_req(
            "POST",
            &format!("/v1/translators/{id}/translate"),
            json!({"vectors": [src[0].clone(), src[1].clone()]}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["rows"], 2);
    assert_eq!(v["dim"], 8);
    let y0: Vec<f32> = v["vectors"][0]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    let cos: f32 = y0.iter().zip(&tgt[0]).map(|(a, b)| a * b).sum();
    assert!(cos > 0.99, "cos {cos}");

    // Translate binary in / binary out.
    let mut raw = Vec::new();
    for v in &src[..3] {
        for f in v {
            raw.extend_from_slice(&f.to_le_bytes());
        }
    }
    let (s, b, h) = send(
        &app,
        Request::post("/v1/translators/demo/translate")
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .header(header::ACCEPT, "application/octet-stream")
            .body(Body::from(raw.clone()))
            .unwrap(),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(h["x-vecski-rows"], "3");
    assert_eq!(h["x-vecski-dim"], "8");
    assert_eq!(b.len(), 3 * 8 * 4);
    let y: Vec<f32> = b
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    for (i, y) in y.chunks_exact(8).enumerate() {
        assert!((y.iter().zip(&y0).map(|(a, b)| a * b).sum::<f32>() - 1.0).abs() < 1e-4 || i > 0);
    }
    // Binary in, JSON out when Accept says so.
    let (s, v) = send_json(
        &app,
        Request::post("/v1/translators/demo/translate")
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .header(header::ACCEPT, "application/json")
            .body(Body::from(raw))
            .unwrap(),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["rows"], 3);

    // Wrong dimension -> 400.
    let (s, v) = send_json(
        &app,
        json_req(
            "POST",
            "/v1/translators/demo/translate",
            json!({"vectors": [[1.0, 2.0]]}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["code"], "bad_request");

    // Evaluate.
    let (s, v) = send_json(
        &app,
        json_req(
            "POST",
            "/v1/translators/demo/evaluate",
            json!({"k": 5, "source": src[..100], "target": tgt[..100]}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["k"], 5);
    assert!(v["top1_accuracy"].as_f64().unwrap() > 0.95);

    // Export, delete, re-import with a new name.
    let (s, bytes, h) = send(
        &app,
        Request::get("/v1/translators/demo/safetensors")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert!(
        h[header::CONTENT_DISPOSITION]
            .to_str()
            .unwrap()
            .contains("demo.safetensors")
    );
    let (s, v) = send_json(
        &app,
        Request::delete("/v1/translators/demo")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["deleted"], true);
    assert!(!dir.join(format!("{id}.safetensors")).exists());
    let (s, _) = send_json(
        &app,
        Request::get("/v1/translators/demo")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    let (s, v) = send_json(
        &app,
        Request::post("/v1/translators/import?name=demo2&source_model=a")
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .body(Body::from(bytes))
            .unwrap(),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    assert_eq!(v["name"], "demo2");
    assert_eq!(v["method"], "procrustes");
    assert!(
        v["report"]["holdout"].is_object(),
        "report should survive export/import"
    );

    std::fs::remove_dir_all(dir).ok();
}

#[tokio::test]
async fn multipart_fit_and_restart_reload() {
    let dir = temp_dir("multipart");
    let (app, _st) = app(dir.clone(), vec![]);
    let (src, tgt) = pairs(500, 6, 7);
    let mut sb = Vec::new();
    let mut tb = Vec::new();
    for (x, y) in src.iter().zip(&tgt) {
        x.iter()
            .for_each(|f| sb.extend_from_slice(&f.to_le_bytes()));
        y.iter()
            .for_each(|f| tb.extend_from_slice(&f.to_le_bytes()));
    }
    let boundary = "vecskiboundary";
    let mut body = Vec::new();
    let mut part = |name: &str, ct: &str, data: &[u8]| {
        body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\nContent-Type: {ct}\r\n\r\n").as_bytes());
        body.extend_from_slice(data);
        body.extend_from_slice(b"\r\n");
    };
    part(
        "meta",
        "application/json",
        json!({"name": "mp", "source_dim": 6, "target_dim": 6, "options": {"method": "ridge"}})
            .to_string()
            .as_bytes(),
    );
    part("source", "application/octet-stream", &sb);
    part("target", "application/octet-stream", &tb);
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());

    let (s, v) = send_json(
        &app,
        Request::post("/v1/translators")
            .header(
                header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from(body))
            .unwrap(),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    assert_eq!(v["method"], "ridge");
    assert!(v["report"]["lambda_sweep"].is_array());
    let id = v["id"].as_str().unwrap().to_string();

    // "Restart": a fresh state over the same directory sees the translator.
    let (app2, st2) = self::app(dir.clone(), vec![]);
    assert_eq!(st2.len(), 1);
    let (s, v) = send_json(
        &app2,
        Request::get("/v1/translators/mp")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["id"], id);
    assert_eq!(v["report"]["method"], "ridge");
    std::fs::remove_dir_all(dir).ok();
}

#[tokio::test]
async fn api_keys_guard_v1_only() {
    let dir = temp_dir("auth");
    let (app, _st) = app(dir.clone(), vec!["secret-1", "secret-2"]);
    let (s, _) = send_json(&app, Request::get("/healthz").body(Body::empty()).unwrap()).await;
    assert_eq!(s, StatusCode::OK);
    let (s, v) = send_json(
        &app,
        Request::get("/v1/translators").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    assert_eq!(v["error"]["code"], "unauthorized");
    let (s, _) = send_json(
        &app,
        Request::get("/v1/translators")
            .header(header::AUTHORIZATION, "Bearer secret-2")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    let (s, _) = send_json(
        &app,
        Request::get("/v1/translators")
            .header("x-api-key", "secret-1")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    let (s, _) = send_json(
        &app,
        Request::get("/v1/translators")
            .header("x-api-key", "nope")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    std::fs::remove_dir_all(dir).ok();
}

#[tokio::test]
async fn bad_names_and_bodies_are_rejected() {
    let dir = temp_dir("bad");
    let (app, _st) = app(dir.clone(), vec![]);
    let (src, tgt) = pairs(50, 4, 3);
    let (s, v) = send_json(
        &app,
        json_req(
            "POST",
            "/v1/translators",
            json!({"name": "has space", "source": src, "target": tgt}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    let (s, _) = send_json(
        &app,
        json_req(
            "POST",
            "/v1/translators",
            json!({"name": "tr_x", "source": src, "target": tgt}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    let (s, v) = send_json(
        &app,
        json_req(
            "POST",
            "/v1/translators",
            json!({"source": src, "target": tgt[..10]}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("pairs must align")
    );
    let (s, _) = send_json(
        &app,
        Request::post("/v1/translators")
            .header(header::CONTENT_TYPE, "text/plain")
            .body(Body::from("x"))
            .unwrap(),
    )
    .await;
    assert_eq!(s, StatusCode::UNSUPPORTED_MEDIA_TYPE);
    std::fs::remove_dir_all(dir).ok();
}
