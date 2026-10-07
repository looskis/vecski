//! In-memory registry of translators plus on-disk persistence (one safetensors
//! file per translator; the fit report rides along in the file's metadata).

use crate::error::ApiError;
use crate::timefmt::now_rfc3339;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use utoipa::ToSchema;
use vecski_core::fit::FitReport;
use vecski_core::{Translator, io};

#[derive(Clone, Debug)]
pub struct Config {
    pub data_dir: PathBuf,
    pub api_keys: Vec<String>,
    pub max_body_bytes: usize,
    pub public_url: Option<String>,
}

/// User-supplied descriptive fields.
#[derive(Clone, Debug, Default, Serialize, Deserialize, ToSchema, utoipa::IntoParams)]
#[into_params(parameter_in = Query)]
pub struct TranslatorMeta {
    /// Optional unique handle usable in place of the id in URLs.
    pub name: Option<String>,
    /// Model that produced the source vectors, e.g. `text-embedding-ada-002`.
    pub source_model: Option<String>,
    /// Model that produced the target vectors, e.g. `text-embedding-3-large`.
    pub target_model: Option<String>,
    pub description: Option<String>,
}

/// Everything stored about a translator except the weights.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Record {
    pub id: String,
    #[serde(flatten)]
    pub meta: TranslatorMeta,
    pub created_at: String,
    pub report: Option<FitReport>,
}

pub struct Entry {
    pub record: Record,
    pub translator: Arc<Translator>,
}

#[derive(Default)]
struct Registry {
    by_id: HashMap<String, Arc<Entry>>,
    by_name: HashMap<String, String>,
}

pub struct AppState {
    pub config: Config,
    registry: RwLock<Registry>,
}

const RECORD_KEY: &str = "vecski.record";

impl AppState {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            registry: RwLock::new(Registry::default()),
        }
    }

    pub fn new_id() -> String {
        format!("tr_{}", uuid::Uuid::now_v7().simple())
    }

    pub fn file_path(&self, id: &str) -> PathBuf {
        self.config.data_dir.join(format!("{id}.safetensors"))
    }

    /// Validate a name: URL-safe, not id-shaped.
    pub fn check_name(name: &str) -> Result<(), ApiError> {
        if name.is_empty() || name.len() > 128 {
            return Err(ApiError::BadRequest("name must be 1-128 characters".into()));
        }
        if !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':'))
        {
            return Err(ApiError::BadRequest(
                "name may only contain ASCII letters, digits, '-', '_', '.', ':'".into(),
            ));
        }
        if name.starts_with("tr_") {
            return Err(ApiError::BadRequest(
                "names may not start with the id prefix 'tr_'".into(),
            ));
        }
        Ok(())
    }

    /// Register a fitted translator, persist it, and return the entry.
    pub fn insert(
        &self,
        meta: TranslatorMeta,
        report: Option<FitReport>,
        translator: Translator,
    ) -> Result<Arc<Entry>, ApiError> {
        if let Some(n) = &meta.name {
            Self::check_name(n)?;
        }
        let record = Record {
            id: Self::new_id(),
            meta,
            created_at: now_rfc3339(),
            report,
        };
        let entry = Arc::new(Entry {
            record,
            translator: Arc::new(translator),
        });
        {
            let mut reg = self.registry.write().expect("registry poisoned");
            if let Some(n) = &entry.record.meta.name {
                if reg.by_name.contains_key(n) {
                    return Err(ApiError::Conflict(format!(
                        "a translator named '{n}' already exists"
                    )));
                }
                reg.by_name.insert(n.clone(), entry.record.id.clone());
            }
            reg.by_id.insert(entry.record.id.clone(), entry.clone());
        }
        if let Err(e) = self.persist(&entry) {
            self.remove(&entry.record.id);
            return Err(e);
        }
        Ok(entry)
    }

    pub fn get(&self, key: &str) -> Result<Arc<Entry>, ApiError> {
        let reg = self.registry.read().expect("registry poisoned");
        let id = reg
            .by_name
            .get(key)
            .cloned()
            .unwrap_or_else(|| key.to_string());
        reg.by_id
            .get(&id)
            .cloned()
            .ok_or_else(|| ApiError::NotFound(format!("no translator with id or name '{key}'")))
    }

    pub fn remove(&self, key: &str) -> Option<Arc<Entry>> {
        let mut reg = self.registry.write().expect("registry poisoned");
        let id = reg
            .by_name
            .get(key)
            .cloned()
            .unwrap_or_else(|| key.to_string());
        let entry = reg.by_id.remove(&id)?;
        if let Some(n) = &entry.record.meta.name {
            reg.by_name.remove(n);
        }
        Some(entry)
    }

    pub fn delete(&self, key: &str) -> Result<Arc<Entry>, ApiError> {
        let entry = self
            .remove(key)
            .ok_or_else(|| ApiError::NotFound(format!("no translator with id or name '{key}'")))?;
        let path = self.file_path(&entry.record.id);
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(ApiError::Internal(format!(
                    "failed to delete {}: {e}",
                    path.display()
                )));
            }
        }
        Ok(entry)
    }

    pub fn list(&self) -> Vec<Arc<Entry>> {
        let reg = self.registry.read().expect("registry poisoned");
        let mut v: Vec<_> = reg.by_id.values().cloned().collect();
        v.sort_by(|a, b| {
            b.record
                .created_at
                .cmp(&a.record.created_at)
                .then(b.record.id.cmp(&a.record.id))
        });
        v
    }

    pub fn len(&self) -> usize {
        self.registry.read().expect("registry poisoned").by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Serialize an entry to safetensors bytes (weights + record in metadata).
    pub fn to_bytes(entry: &Entry) -> Result<Vec<u8>, ApiError> {
        let mut meta = HashMap::new();
        meta.insert(
            RECORD_KEY.to_string(),
            serde_json::to_string(&entry.record).map_err(|e| ApiError::Internal(e.to_string()))?,
        );
        if let Some(n) = &entry.record.meta.name {
            meta.insert("name".into(), n.clone());
        }
        Ok(io::to_safetensors(&entry.translator, &meta)?)
    }

    fn persist(&self, entry: &Entry) -> Result<(), ApiError> {
        std::fs::create_dir_all(&self.config.data_dir)
            .map_err(|e| ApiError::Internal(format!("cannot create data dir: {e}")))?;
        let bytes = Self::to_bytes(entry)?;
        let path = self.file_path(&entry.record.id);
        let tmp = path.with_extension("safetensors.tmp");
        std::fs::write(&tmp, &bytes)
            .map_err(|e| ApiError::Internal(format!("cannot write {}: {e}", tmp.display())))?;
        std::fs::rename(&tmp, &path).map_err(|e| {
            ApiError::Internal(format!("cannot rename into {}: {e}", path.display()))
        })?;
        Ok(())
    }

    /// Load every `*.safetensors` in the data dir. Files without a vecski record
    /// (e.g. hand-made weight files) are registered under their file stem.
    pub fn load_all(&self) -> anyhow::Result<usize> {
        let dir = &self.config.data_dir;
        if !dir.exists() {
            return Ok(0);
        }
        let mut loaded = 0;
        for ent in std::fs::read_dir(dir)? {
            let path = ent?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
                continue;
            }
            match self.load_file(&path) {
                Ok(()) => loaded += 1,
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "skipping translator file")
                }
            }
        }
        Ok(loaded)
    }

    fn load_file(&self, path: &Path) -> anyhow::Result<()> {
        let bytes = std::fs::read(path)?;
        let (translator, meta) = io::from_safetensors(&bytes)?;
        let record = match meta.get(RECORD_KEY) {
            Some(json) => serde_json::from_str::<Record>(json)?,
            None => {
                let stem = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("imported")
                    .to_string();
                Record {
                    id: if stem.starts_with("tr_") {
                        stem.clone()
                    } else {
                        Self::new_id()
                    },
                    meta: TranslatorMeta {
                        name: (!stem.starts_with("tr_")).then_some(stem),
                        ..Default::default()
                    },
                    created_at: now_rfc3339(),
                    report: None,
                }
            }
        };
        let entry = Arc::new(Entry {
            record,
            translator: Arc::new(translator),
        });
        let mut reg = self.registry.write().expect("registry poisoned");
        if let Some(n) = &entry.record.meta.name {
            reg.by_name.insert(n.clone(), entry.record.id.clone());
        }
        reg.by_id.insert(entry.record.id.clone(), entry);
        Ok(())
    }
}
