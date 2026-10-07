use clap::Parser;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tracing_subscriber::EnvFilter;
use vecski_server::api::router;
use vecski_server::{AppState, Config};

/// vecski: a Rust API server that fits and applies embedding-space translators.
#[derive(Parser, Debug)]
#[command(name = "vecski", version, about)]
struct Cli {
    /// Address to bind.
    #[arg(long, env = "VECSKI_BIND", default_value = "0.0.0.0:8080")]
    bind: SocketAddr,

    /// Directory where translators are persisted as safetensors files.
    #[arg(long, env = "VECSKI_DATA_DIR", default_value = "./data")]
    data_dir: PathBuf,

    /// Comma-separated API keys. When set, /v1/* requires one (Bearer or X-API-Key).
    #[arg(
        long,
        env = "VECSKI_API_KEYS",
        value_delimiter = ',',
        hide_env_values = true
    )]
    api_keys: Vec<String>,

    /// Maximum request body size in megabytes (fit uploads can be large).
    #[arg(long, env = "VECSKI_MAX_BODY_MB", default_value_t = 2048)]
    max_body_mb: usize,

    /// Threads for linear algebra (0 = all cores).
    #[arg(long, env = "VECSKI_THREADS", default_value_t = 0)]
    threads: usize,

    /// Public base URL used in docs output, e.g. https://vecski.example.com
    #[arg(long, env = "VECSKI_PUBLIC_URL")]
    public_url: Option<String>,

    /// Emit JSON logs.
    #[arg(long, env = "VECSKI_LOG_JSON", default_value_t = false)]
    log_json: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,tower_http=info"));
    if cli.log_json {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .json()
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }

    if cli.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cli.threads)
            .build_global()
            .ok();
    }
    vecski_core::set_threads(cli.threads);

    let api_keys: Vec<String> = cli
        .api_keys
        .into_iter()
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty())
        .collect();
    let config = Config {
        data_dir: cli.data_dir,
        api_keys,
        max_body_bytes: cli.max_body_mb.saturating_mul(1024 * 1024),
        public_url: cli.public_url.map(|u| u.trim_end_matches('/').to_string()),
    };
    let state = Arc::new(AppState::new(config));
    let loaded = state.load_all()?;
    tracing::info!(
        bind = %cli.bind,
        data_dir = %state.config.data_dir.display(),
        translators = loaded,
        auth = !state.config.api_keys.is_empty(),
        threads = rayon::current_num_threads(),
        "vecski starting"
    );

    let listener = tokio::net::TcpListener::bind(cli.bind).await?;
    axum::serve(listener, router(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await?;
    Ok(())
}
