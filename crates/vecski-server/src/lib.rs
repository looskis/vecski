//! vecski HTTP server library (the binary lives in `main.rs`).

pub mod api;
pub mod auth;
pub mod error;
pub mod state;
pub mod timefmt;

pub use state::{AppState, Config};
