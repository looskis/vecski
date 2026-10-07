//! safetensors serialization. The file is loadable from Python with
//! `safetensors.numpy.load_file`; apply `y = x @ weight + bias` (then normalize).

use crate::error::{CoreError, Result};
use crate::translator::{Translator, TranslatorInfo};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use std::collections::HashMap;

pub const FORMAT: &str = "vecski/v1";

fn f32s_to_le(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn le_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Serialize a translator (plus arbitrary string metadata) to safetensors bytes.
pub fn to_safetensors(tr: &Translator, extra: &HashMap<String, String>) -> Result<Vec<u8>> {
    let w = f32s_to_le(tr.weight());
    let b = f32s_to_le(tr.bias());
    let m = f32s_to_le(tr.target_mean());
    let tensors = vec![
        (
            "weight".to_string(),
            TensorView::new(Dtype::F32, vec![tr.source_dim(), tr.target_dim()], &w)
                .map_err(|e| CoreError::Serialization(e.to_string()))?,
        ),
        (
            "bias".to_string(),
            TensorView::new(Dtype::F32, vec![tr.target_dim()], &b)
                .map_err(|e| CoreError::Serialization(e.to_string()))?,
        ),
        (
            "target_mean".to_string(),
            TensorView::new(Dtype::F32, vec![tr.target_dim()], &m)
                .map_err(|e| CoreError::Serialization(e.to_string()))?,
        ),
    ];
    let mut meta = extra.clone();
    meta.insert("format".into(), FORMAT.into());
    meta.insert("source_dim".into(), tr.source_dim().to_string());
    meta.insert("target_dim".into(), tr.target_dim().to_string());
    meta.insert("normalize_output".into(), tr.normalize_output().to_string());
    meta.insert(
        "info".into(),
        serde_json::to_string(tr.info()).map_err(|e| CoreError::Serialization(e.to_string()))?,
    );
    safetensors::serialize(tensors, Some(meta)).map_err(|e| CoreError::Serialization(e.to_string()))
}

/// Load a translator from safetensors bytes. Accepts files written by
/// `to_safetensors` and plain `weight` (+ optional `bias`) files from elsewhere.
pub fn from_safetensors(bytes: &[u8]) -> Result<(Translator, HashMap<String, String>)> {
    let st =
        SafeTensors::deserialize(bytes).map_err(|e| CoreError::Serialization(e.to_string()))?;
    let (_, header) =
        SafeTensors::read_metadata(bytes).map_err(|e| CoreError::Serialization(e.to_string()))?;
    let meta: HashMap<String, String> = header.metadata().clone().unwrap_or_default();

    let w = st
        .tensor("weight")
        .map_err(|e| CoreError::Serialization(format!("weight: {e}")))?;
    if w.dtype() != Dtype::F32 || w.shape().len() != 2 {
        return Err(CoreError::Serialization(
            "weight must be a rank-2 F32 tensor".into(),
        ));
    }
    let (d1, d2) = (w.shape()[0], w.shape()[1]);
    let weight = le_to_f32s(w.data());

    let read_vec = |name: &str| -> Result<Option<Vec<f32>>> {
        match st.tensor(name) {
            Ok(t) => {
                if t.dtype() != Dtype::F32 || t.shape() != [d2] {
                    return Err(CoreError::Serialization(format!(
                        "{name} must be an F32 tensor of shape [{d2}]"
                    )));
                }
                Ok(Some(le_to_f32s(t.data())))
            }
            Err(_) => Ok(None),
        }
    };
    let bias = read_vec("bias")?.unwrap_or_else(|| vec![0.0; d2]);
    let target_mean = read_vec("target_mean")?.unwrap_or_else(|| vec![0.0; d2]);
    let normalize_output = meta
        .get("normalize_output")
        .map(|v| v == "true")
        .unwrap_or(true);
    let info: TranslatorInfo = meta
        .get("info")
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default();
    let tr = Translator::new(d1, d2, weight, bias, target_mean, normalize_output, info)?;
    Ok((tr, meta))
}
