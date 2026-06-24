# syntax=docker/dockerfile:1.7

FROM rust:1-bookworm AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        libasound2-dev \
        libssl-dev \
        libudev-dev \
        libusb-1.0-0-dev \
        libv4l-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build --release -p netherwick-tools --bin netherwick \
    && cp target/release/netherwick /usr/local/bin/netherwick

FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libasound2 \
        libgcc-s1 \
        libssl3 \
        libudev1 \
        libusb-1.0-0 \
        libv4l-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /usr/local/bin/netherwick /usr/local/bin/netherwick
COPY configs /app/configs

ENTRYPOINT ["/usr/local/bin/netherwick"]
