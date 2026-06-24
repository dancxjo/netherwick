set shell := ["bash", "-euxo", "pipefail", "-c"]

create1_port := env_var_or_default("CREATE1_PORT", "/dev/ttyUSB0")
gps_serial_port := env_var_or_default("GPS_SERIAL_PORT", "/dev/ttyACM0")
camera_device := env_var_or_default("CAMERA_DEVICE", "/dev/video0")

default:
    @just --list

setup: setup-system setup-rust setup-kinect
    @echo "netherwick Linux setup complete"
    @echo "next: cargo check && just sim"

setup-system:
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        pkg-config \
        cmake \
        ninja-build \
        git \
        curl \
        just \
        ffmpeg \
        v4l-utils \
        libasound2-dev \
        libudev-dev \
        libusb-1.0-0-dev \
        libv4l-dev

setup-rust:
    if ! command -v cargo >/dev/null 2>&1; then \
        curl https://sh.rustup.rs -sSf | sh -s -- -y; \
    fi

setup-kinect:
    if apt-cache show libfreenect-dev >/dev/null 2>&1; then \
        sudo apt-get install -y libfreenect-dev freenect; \
    else \
        echo "libfreenect-dev not found in apt metadata; run 'just setup-kinect-from-source'"; \
        exit 1; \
    fi

setup-kinect-from-source:
    mkdir -p .vendor
    if [ ! -d .vendor/libfreenect/.git ]; then \
        git clone https://github.com/OpenKinect/libfreenect.git .vendor/libfreenect; \
    fi
    cmake -S .vendor/libfreenect -B .vendor/libfreenect/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_CPP=ON \
        -DBUILD_AUDIO=ON \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_OPENNI2_DRIVER=OFF
    cmake --build .vendor/libfreenect/build -j
    sudo cmake --install .vendor/libfreenect/build

fmt:
    cargo fmt --all

check:
    cargo check --workspace

compose-config:
    docker compose config

compose-build:
    docker compose build netherwick-live

servers:
    docker compose up -d neo4j qdrant

live-server:
    docker compose --profile netherwick up -d neo4j qdrant netherwick-live

server-logs service="netherwick-live":
    docker compose logs -f {{service}}

stop-servers:
    docker compose down

test:
    cargo test --workspace

clippy:
    cargo clippy --workspace --all-targets -- -D warnings

sim:
    cargo run -p netherwick-tools -- sim

go target="virtual":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{target}}" != "virtual" ]; then
        echo "usage: just go virtual"
        exit 2
    fi
    just dev-cert
    PORT="${NETHERWICK_LIVE_PORT:-8787}"
    LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
    if [ -z "$LAN_IP" ]; then LAN_IP="127.0.0.1"; fi
    echo
    echo "Netherwick Dream World is starting."
    echo "Virtual training theater is collecting experience."
    echo "Inline learning defaults to world-outcome. Set NETHERWICK_INLINE_LEARNING_MODE=off for collect-only."
    echo 'Offline training still exists: `cargo run --bin netherwick -- train behavior ...`'
    echo
    echo "Desktop:"
    echo "  https://127.0.0.1:$PORT/view/3d"
    echo
    echo "Headset/LAN:"
    echo "  https://$LAN_IP:$PORT/view/3d"
    echo
    echo "Scene JSON:"
    echo "  https://$LAN_IP:$PORT/view/scene"
    echo
    echo "This serves robot/dream-world sensor data on the LAN. Use only on trusted networks."
    if command -v qrencode >/dev/null 2>&1; then
        qrencode -t ANSIUTF8 "https://$LAN_IP:$PORT/view/3d" || true
    fi
    if [ "${NETHERWICK_OPEN_BROWSER:-0}" = "1" ] && command -v xdg-open >/dev/null 2>&1; then
        xdg-open "https://127.0.0.1:$PORT/view/3d" >/dev/null 2>&1 || true
    fi
    cargo run -p netherwick-tools -- sim \
        --live \
        --live-tls \
        --live-addr "0.0.0.0:$PORT" \
        --live-tls-cert certs/netherwick-dev.crt \
        --live-tls-key certs/netherwick-dev.key \
        --scenario "${NETHERWICK_SCENARIO:-dream}" \
        --steps "${NETHERWICK_SIM_STEPS:-1000000000}" \
        --tick-delay-ms "${NETHERWICK_TICK_DELAY_MS:-100}" \
        --ledger "${NETHERWICK_LEDGER:-data/ledger/virtual-live}"

go-virtual:
    just go virtual

virtual:
    just go virtual

virtual-https:
    just go virtual

train target="virtual":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{target}}" != "virtual" ]; then
        echo "usage: just train virtual"
        exit 2
    fi
    cargo run -p netherwick-tools -- train virtual \
        --ledger "${NETHERWICK_LEDGER:-data/ledger/virtual-live}" \
        --out-dir "${NETHERWICK_MODEL_OUT:-data/models/virtual/latest}" \
        --report-out "${NETHERWICK_REPORT_OUT:-data/reports/virtual/latest.json}" \
        --epochs "${NETHERWICK_EPOCHS:-5}"

train-virtual:
    just train virtual

dev-cert:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p certs
    LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
    if [ -z "$LAN_IP" ]; then
        LAN_SAN=""
        echo "warning: could not detect LAN IP; certificate will cover localhost only"
    else
        LAN_SAN=",IP:$LAN_IP"
    fi
    if [ ! -f certs/netherwick-dev.crt ] || [ ! -f certs/netherwick-dev.key ]; then
        openssl req -x509 -newkey rsa:2048 -nodes \
            -keyout certs/netherwick-dev.key \
            -out certs/netherwick-dev.crt \
            -days 365 \
            -subj "/CN=netherwick.local" \
            -addext "subjectAltName=DNS:localhost,DNS:netherwick.local,IP:127.0.0.1$LAN_SAN"
        echo "generated certs/netherwick-dev.crt and certs/netherwick-dev.key"
    else
        echo "using existing certs/netherwick-dev.crt and certs/netherwick-dev.key"
    fi

virtual-url:
    #!/usr/bin/env bash
    set -euo pipefail
    PORT="${NETHERWICK_LIVE_PORT:-8787}"
    LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
    if [ -z "$LAN_IP" ]; then LAN_IP="127.0.0.1"; fi
    echo "https://$LAN_IP:$PORT/view/3d"

run *args:
    cargo run -p netherwick-tools -- {{args}}

rehearse-models:
    just run sim --steps 200 --ledger data/ledger/sim1
    just run train behavior danger --ledger data/ledger/sim1
    just run train behavior charge --ledger data/ledger/sim1
    just run train behavior future --ledger data/ledger/sim1
    just run evaluate behavior danger --ledger data/ledger/sim1
    just run model-status
    just run sim --steps 200 --danger-checkpoint data/models/danger_v0 --danger-mode shadow-infer
    just run robot --mode read-only --create-port mock --steps 20 --capture data/captures/mock-readonly
    just run replay-capture --capture data/captures/mock-readonly

eval-scenario-smoke:
    just run eval-scenario --scenario empty-room --episodes 2 --steps 10 --out data/reports/scenario/empty-smoke.json
    just run eval-scenario --scenario obstacle-avoidance --episodes 2 --steps 10 --out data/reports/scenario/obstacle-smoke.json
    just run eval-scenario --scenario corner-trap --episodes 1 --steps 40 --out data/reports/scenario/corner-trap-smoke.json
    just run eval-scenario --scenario charger-seeking --episodes 2 --steps 10 --out data/reports/scenario/charge-smoke.json

inspect-ledger:
    cargo run -p netherwick-tools -- inspect-ledger

xtask command="check":
    cargo run -p xtask -- {{command}}

real-create1:
    @echo "Create 1 port: {{create1_port}}"
    @echo "The Create 1 control path is scaffolded at the crate level but the CLI robot mode is not wired yet."

hardware-env:
    cargo run -p netherwick-tools -- hardware-env

clean:
    cargo clean
