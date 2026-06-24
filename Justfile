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

inspect-ledger:
    cargo run -p netherwick-tools -- inspect-ledger

xtask command="check":
    cargo run -p xtask -- {{command}}

real-create1:
    @echo "Create 1 port: {{create1_port}}"
    @echo "The Create 1 control path is scaffolded at the crate level but the CLI robot mode is not wired yet."

hardware-env:
    @echo "CREATE1_PORT={{create1_port}}"
    @echo "GPS_SERIAL_PORT={{gps_serial_port}}"
    @echo "CAMERA_DEVICE={{camera_device}}"

clean:
    cargo clean
