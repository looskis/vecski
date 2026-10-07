# syntax=docker/dockerfile:1
FROM rust:1.99-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
# Set RUSTFLAGS="-C target-cpu=native" only if the image will run on the build host.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release -p vecski-server && cp target/release/vecski /vecski

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /vecski /usr/local/bin/vecski
ENV VECSKI_BIND=0.0.0.0:8080 VECSKI_DATA_DIR=/data
VOLUME /data
EXPOSE 8080
ENTRYPOINT ["vecski"]
