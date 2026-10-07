#!/usr/bin/env python3
"""End-to-end vecski demo with synthetic pairs (numpy only, no extra packages).

    python examples/quickstart.py http://localhost:8080

Replace `make_pairs` with real embeddings: the same texts embedded by the old
model (source) and the new model (target).
"""
import json
import sys
import urllib.request
import uuid

import numpy as np

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080").rstrip("/")
API_KEY = None  # set if the server runs with --api-keys


def call(method, path, data=None, headers=None):
    h = dict(headers or {})
    if API_KEY:
        h["authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(f"{BASE}{path}", data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def make_pairs(n=6000, d1=384, d2=768, seed=0):
    """Target = unit-normalized (rotation of source into a wider space + shift + noise)."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d2, d2)))
    src = rng.standard_normal((n, d1)).astype(np.float32)
    src /= np.linalg.norm(src, axis=1, keepdims=True)
    padded = np.concatenate([src, np.zeros((n, d2 - d1), np.float32)], axis=1)
    tgt = 1.3 * padded @ q[:, :d2] + 0.05 * rng.standard_normal(d2) + 0.02 * rng.standard_normal((n, d2))
    tgt = (tgt / np.linalg.norm(tgt, axis=1, keepdims=True)).astype(np.float32)
    return src, tgt


def main():
    src, tgt = make_pairs()
    n, d1 = src.shape
    d2 = tgt.shape[1]
    name = f"demo-{uuid.uuid4().hex[:6]}"

    # --- fit via multipart (raw float32 bytes) ------------------------------
    boundary = "vecski" + uuid.uuid4().hex
    meta = {"name": name, "source_model": "old-model", "target_model": "new-model",
            "source_dim": d1, "target_dim": d2, "options": {"method": "auto", "k": 10}}
    parts = [("meta", "application/json", json.dumps(meta).encode()),
             ("source", "application/octet-stream", src.tobytes()),
             ("target", "application/octet-stream", tgt.tobytes())]
    body = b""
    for pname, ctype, data in parts:
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{pname}\"\r\n"
                 f"Content-Type: {ctype}\r\n\r\n").encode() + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    status, _, raw = call("POST", "/v1/translators", body,
                          {"content-type": f"multipart/form-data; boundary={boundary}"})
    rep = json.loads(raw)
    if status != 201:
        sys.exit(f"fit failed: {status} {rep}")
    h = rep["report"]["holdout"]
    print(f"fitted {rep['id']} ({rep['method']}) in {rep['report']['fit_ms']} ms on {rep['report']['n_train']} pairs")
    print(f"  holdout cosine {h['mean_cosine']:.3f} vs mean-vector baseline {h['mean_vector_baseline_cosine']:.3f}")
    print(f"  top-1 {h['top1_accuracy']:.3f}  MRR {h['mrr']:.3f}  neighbor overlap@{h['k']} {h['neighbor_overlap_at_k']:.3f}")
    for w in rep["report"]["warnings"]:
        print("  warning:", w)

    # --- translate via binary ------------------------------------------------
    batch = src[:1000]
    status, headers, raw = call("POST", f"/v1/translators/{name}/translate", batch.tobytes(),
                                {"content-type": "application/octet-stream",
                                 "accept": "application/octet-stream"})
    assert status == 200, raw
    y = np.frombuffer(raw, np.float32).reshape(int(headers["X-Vecski-Rows"]), int(headers["X-Vecski-Dim"]))
    cos = (y * tgt[:1000]).sum(axis=1)
    print(f"translated {y.shape[0]} vectors; mean cosine to native target {cos.mean():.3f}")

    # --- translate via JSON --------------------------------------------------
    status, _, raw = call("POST", f"/v1/translators/{name}/translate",
                          json.dumps({"vectors": src[:2].tolist()}).encode(),
                          {"content-type": "application/json"})
    assert status == 200, raw
    print("JSON translate rows:", json.loads(raw)["rows"])

    # --- download weights and apply offline ---------------------------------
    status, _, raw = call("GET", f"/v1/translators/{name}/safetensors")
    assert status == 200
    try:
        from safetensors.numpy import load as st_load
        t = st_load(raw)
        y2 = batch @ t["weight"] + t["bias"]
        y2 /= np.linalg.norm(y2, axis=1, keepdims=True)
        print(f"offline apply matches server: max abs diff {np.abs(y2 - y).max():.2e}")
    except ImportError:
        print(f"downloaded {len(raw)} bytes of safetensors (pip install safetensors to apply offline)")

    call("DELETE", f"/v1/translators/{name}")
    print("deleted", name)


if __name__ == "__main__":
    main()
