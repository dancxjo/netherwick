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
    echo "Dream controller mode: baseline (no automatic NEAT training in go virtual)."
    echo "Use just evolve for Dream NEAT policy training."
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
    cargo build -p netherwick-tools
    AUTO_DREAM_POLICY="${NETHERWICK_AUTO_DREAM_POLICY:-0}"
    DREAM_POLICY_CHECKPOINT="${NETHERWICK_DREAM_POLICY_CHECKPOINT:-}"
    if [ -z "$DREAM_POLICY_CHECKPOINT" ] && [ "$AUTO_DREAM_POLICY" = "1" ] && [ -f "${NETHERWICK_NEAT_CHECKPOINT_DIR:-data/models/dream-policy/neat}/evolve-best.json" ]; then
        DREAM_POLICY_CHECKPOINT="${NETHERWICK_NEAT_CHECKPOINT_DIR:-data/models/dream-policy/neat}/evolve-best.json"
    fi
    SIM_DREAM_POLICY_ARGS=()
    if [ -n "$DREAM_POLICY_CHECKPOINT" ]; then
        echo "Visible Dream World controller: $DREAM_POLICY_CHECKPOINT"
        SIM_DREAM_POLICY_ARGS=(--dream-policy-checkpoint "$DREAM_POLICY_CHECKPOINT")
    else
        echo "Visible Dream World controller: baseline"
    fi
    target/debug/netherwick sim \
        --live \
        --live-tls \
        --live-addr "0.0.0.0:$PORT" \
        --live-tls-cert certs/netherwick-dev.crt \
        --live-tls-key certs/netherwick-dev.key \
        --scenario "${NETHERWICK_SCENARIO:-dream}" \
        --steps "${NETHERWICK_SIM_STEPS:-1000000000}" \
        --tick-delay-ms "${NETHERWICK_TICK_DELAY_MS:-100}" \
        --ledger "${NETHERWICK_LEDGER:-data/ledger/virtual-live}" \
        "${SIM_DREAM_POLICY_ARGS[@]}" &
    PID="$!"
    trap 'kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true' INT TERM EXIT
    if [ "${NETHERWICK_OPEN_BROWSER:-0}" = "1" ] && command -v xdg-open >/dev/null 2>&1; then
        if command -v curl >/dev/null 2>&1; then
            for _ in $(seq 1 80); do
                if curl -kfsS "https://127.0.0.1:$PORT/view/scene" >/dev/null 2>&1; then
                    xdg-open "https://127.0.0.1:$PORT/view/3d" >/dev/null 2>&1 || true
                    break
                fi
                if ! kill -0 "$PID" >/dev/null 2>&1; then
                    break
                fi
                sleep 0.25
            done
        else
            sleep 2
            xdg-open "https://127.0.0.1:$PORT/view/3d" >/dev/null 2>&1 || true
        fi
    fi
    set +e
    wait "$PID"
    STATUS="$?"
    set -e
    trap - INT TERM EXIT
    exit "$STATUS"

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

evolve clear="false":
    #!/usr/bin/env bash
    set -euo pipefail
    START_LEVEL="${NETHERWICK_NEAT_START_LEVEL:-motion}"
    GENERATIONS="${NETHERWICK_NEAT_GENERATIONS:-12}"
    POPULATION="${NETHERWICK_NEAT_POPULATION:-24}"
    if [ -n "${NETHERWICK_NEAT_SEED:-}" ]; then
        SEED="${NETHERWICK_NEAT_SEED}"
    else
        SEED="$(date +%s)"
    fi
    HIDDEN_DIM="${NETHERWICK_NEAT_HIDDEN_DIM:-10}"
    CHECKPOINT_DIR="${NETHERWICK_NEAT_CHECKPOINT_DIR:-data/models/dream-policy/neat}"
    DATASET_DIR="${NETHERWICK_NEAT_DATASET_DIR:-datasets/dream-policy/v0/episodes}"
    EXPORT_DATASET="${NETHERWICK_NEAT_EXPORT_DATASET:-false}"
    CLEAR_FLAG=()
    if [ "{{clear}}" = "true" ] || [ "{{clear}}" = "--clear" ]; then
        CLEAR_FLAG=(--clear)
    fi

    echo "Dream NEAT evolve: fast mode + detailed CLI progress"
    echo "  start-level:   $START_LEVEL"
    echo "  generations:   $GENERATIONS"
    echo "  population:    $POPULATION"
    echo "  seed:          $SEED"
    echo "  hidden-dim:    $HIDDEN_DIM"
    echo "  checkpoint:    $CHECKPOINT_DIR"
    echo "  dataset-dir:   $DATASET_DIR"
    echo "  export-dataset:$EXPORT_DATASET"

    cargo build --release -p netherwick-tools
    target/release/netherwick dream-train \
        --start-level "$START_LEVEL" \
        --generations "$GENERATIONS" \
        --population "$POPULATION" \
        --seed "$SEED" \
        --hidden-dim "$HIDDEN_DIM" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --dataset-dir "$DATASET_DIR" \
        --export-dataset "$EXPORT_DATASET" \
        --detailed-logs \
        "${CLEAR_FLAG[@]}"

evolve-fast clear="false":
    just evolve clear={{clear}}

evolve-quality clear="false":
    #!/usr/bin/env bash
    set -euo pipefail
    START_LEVEL="${NETHERWICK_NEAT_START_LEVEL:-motion}"
    GENERATIONS="${NETHERWICK_NEAT_QUALITY_GENERATIONS:-36}"
    POPULATION="${NETHERWICK_NEAT_QUALITY_POPULATION:-64}"
    if [ -n "${NETHERWICK_NEAT_SEED:-}" ]; then
        SEED="${NETHERWICK_NEAT_SEED}"
    else
        SEED="$(date +%s)"
    fi
    HIDDEN_DIM="${NETHERWICK_NEAT_QUALITY_HIDDEN_DIM:-14}"
    CHECKPOINT_DIR="${NETHERWICK_NEAT_CHECKPOINT_DIR:-data/models/dream-policy/neat}"
    DATASET_DIR="${NETHERWICK_NEAT_DATASET_DIR:-datasets/dream-policy/v0/episodes}"
    EXPORT_DATASET="${NETHERWICK_NEAT_EXPORT_DATASET:-false}"
    CLEAR_FLAG=()
    if [ "{{clear}}" = "true" ] || [ "{{clear}}" = "--clear" ]; then
        CLEAR_FLAG=(--clear)
    fi

    echo "Dream NEAT evolve-quality: stronger first-draft profile + detailed CLI progress"
    echo "  start-level:   $START_LEVEL"
    echo "  generations:   $GENERATIONS"
    echo "  population:    $POPULATION"
    echo "  seed:          $SEED"
    echo "  hidden-dim:    $HIDDEN_DIM"
    echo "  checkpoint:    $CHECKPOINT_DIR"
    echo "  dataset-dir:   $DATASET_DIR"
    echo "  export-dataset:$EXPORT_DATASET"

    cargo build --release -p netherwick-tools
    target/release/netherwick dream-train \
        --start-level "$START_LEVEL" \
        --generations "$GENERATIONS" \
        --population "$POPULATION" \
        --seed "$SEED" \
        --hidden-dim "$HIDDEN_DIM" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --dataset-dir "$DATASET_DIR" \
        --export-dataset "$EXPORT_DATASET" \
        --detailed-logs \
        "${CLEAR_FLAG[@]}"

evolve-infinite clear="false":
    #!/usr/bin/env bash
    set -euo pipefail
    cyan() { printf '\033[36m%s\033[0m' "$1"; }
    green() { printf '\033[32m%s\033[0m' "$1"; }
    yellow() { printf '\033[33m%s\033[0m' "$1"; }
    print_progress() {
        local current="$1"
        local total="$2"
        local label="$3"
        if [ "$total" -gt 0 ]; then
            local pct
            pct="$(awk -v c="$current" -v t="$total" 'BEGIN { printf "%.1f", (100*c)/t }')"
            printf "\r%s %5s%% (%s/%s)" "$(cyan "$label")" "$pct" "$current" "$total"
        else
            printf "\r%s %s" "$(cyan "$label")" "$current"
        fi
        if [ "$current" -ge "$total" ] && [ "$total" -gt 0 ]; then
            printf "\n"
        fi
    }

    CLEAR_VALUE="{{clear}}"
    CLEAR_VALUE="${CLEAR_VALUE#clear=}"

    DATASET_DIR="${NETHERWICK_NEAT_DATASET_DIR:-datasets/dream-policy/v0/episodes}"
    EXPORT_DATASET="${NETHERWICK_NEAT_EXPORT_DATASET:-false}"
    CHECKPOINT_DIR="${NETHERWICK_NEAT_CHECKPOINT_DIR:-data/models/dream-policy/neat}"
    BENCHMARK_EVERY="${NETHERWICK_EVOLVE_BENCHMARK_EVERY:-10}"
    BENCHMARK_STEPS="${NETHERWICK_EVOLVE_BENCHMARK_STEPS:-160}"
    BENCHMARK_ROOT="${NETHERWICK_EVOLVE_BENCHMARK_ROOT:-data/reports/scenario/evolve}"
    BENCHMARK_LEDGER_ROOT="${NETHERWICK_EVOLVE_BENCHMARK_LEDGER_ROOT:-data/ledger/evolve-benchmark}"

    BENCHMARK_MAX_RUNS="${NETHERWICK_EVOLVE_BENCHMARK_MAX_RUNS:-64}"
    BENCHMARK_MAX_AGE_DAYS="${NETHERWICK_EVOLVE_BENCHMARK_MAX_AGE_DAYS:-21}"

    DATASET_MAX_FILES="${NETHERWICK_DATASET_MAX_FILES:-8000}"
    DATASET_MAX_BYTES="${NETHERWICK_DATASET_MAX_BYTES:-536870912}"
    DATASET_MAX_AGE_DAYS="${NETHERWICK_DATASET_MAX_AGE_DAYS:-10}"

    prune_dataset() {
        mkdir -p "$DATASET_DIR"
        local files_before
        files_before="$(find "$DATASET_DIR" -type f -name 'level-*-seed-*-genome-*.jsonl' | wc -l || true)"
        if [ "$DATASET_MAX_AGE_DAYS" -gt 0 ]; then
            find "$DATASET_DIR" -type f -name 'level-*-seed-*-genome-*.jsonl' -mtime "+$DATASET_MAX_AGE_DAYS" -delete || true
        fi

        local files_now
        files_now="$(find "$DATASET_DIR" -type f -name 'level-*-seed-*-genome-*.jsonl' | wc -l || true)"
        if [ "$DATASET_MAX_FILES" -gt 0 ] && [ "$files_now" -gt "$DATASET_MAX_FILES" ]; then
            local drop
            drop="$((files_now - DATASET_MAX_FILES))"
            find "$DATASET_DIR" -type f -name 'level-*-seed-*-genome-*.jsonl' -printf '%T@ %p\n' \
                | sort -n \
                | head -n "$drop" \
                | cut -d' ' -f2- \
                | xargs -r rm -f
        fi

        local size_now
        size_now="$(du -sb "$DATASET_DIR" | awk '{print $1}')"
        while [ "$DATASET_MAX_BYTES" -gt 0 ] && [ "$size_now" -gt "$DATASET_MAX_BYTES" ]; do
            local oldest
            oldest="$(find "$DATASET_DIR" -type f -name 'level-*-seed-*-genome-*.jsonl' -printf '%T@ %p\n' | sort -n | head -n 1 | cut -d' ' -f2-)"
            if [ -z "$oldest" ]; then
                break
            fi
            rm -f "$oldest"
            size_now="$(du -sb "$DATASET_DIR" | awk '{print $1}')"
        done

        local files_after
        files_after="$(find "$DATASET_DIR" -type f -name 'level-*-seed-*-genome-*.jsonl' | wc -l || true)"
        echo "$(green "dataset") files: $files_before -> $files_after, size: $(du -sh "$DATASET_DIR" | awk '{print $1}')"
    }

    prune_benchmark_artifacts() {
        mkdir -p "$BENCHMARK_ROOT" "$BENCHMARK_LEDGER_ROOT"
        if [ "$BENCHMARK_MAX_AGE_DAYS" -gt 0 ]; then
            find "$BENCHMARK_ROOT" -mindepth 1 -maxdepth 1 -type d -mtime "+$BENCHMARK_MAX_AGE_DAYS" -exec rm -rf {} + || true
            find "$BENCHMARK_LEDGER_ROOT" -mindepth 1 -maxdepth 1 -type d -mtime "+$BENCHMARK_MAX_AGE_DAYS" -exec rm -rf {} + || true
        fi

        local reports_count
        reports_count="$(find "$BENCHMARK_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l || true)"
        if [ "$BENCHMARK_MAX_RUNS" -gt 0 ] && [ "$reports_count" -gt "$BENCHMARK_MAX_RUNS" ]; then
            local drop
            drop="$((reports_count - BENCHMARK_MAX_RUNS))"
            find "$BENCHMARK_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
                | sort -n \
                | head -n "$drop" \
                | cut -d' ' -f2- \
                | xargs -r rm -rf
            find "$BENCHMARK_LEDGER_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
                | sort -n \
                | head -n "$drop" \
                | cut -d' ' -f2- \
                | xargs -r rm -rf
        fi
        echo "$(green "bench-retain") reports: $(find "$BENCHMARK_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l), ledgers: $(find "$BENCHMARK_LEDGER_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    }

    run_benchmarks() {
        local iteration="$1"
        local checkpoint="$CHECKPOINT_DIR/evolve-best.json"
        if [ ! -f "$checkpoint" ]; then
            echo "$(yellow "benchmark") skipped (missing checkpoint: $checkpoint)"
            return 0
        fi

        local stamp
        stamp="$(date -u +%Y%m%dT%H%M%SZ)"
        local out_dir="$BENCHMARK_ROOT/$stamp-iter-$iteration"
        local ledger_dir="$BENCHMARK_LEDGER_ROOT/$stamp-iter-$iteration"
        mkdir -p "$out_dir" "$ledger_dir"

        local names=(obstacle-avoidance corner-trap column-trap)
        local seeds=(701 1701 2701)
        local total="${#names[@]}"
        local idx=0
        while [ "$idx" -lt "$total" ]; do
            local scenario="${names[$idx]}"
            local seed="${seeds[$idx]}"
            local scenario_ledger="$ledger_dir/$scenario"
            local scenario_out="$out_dir/$scenario.json"
            rm -rf "$scenario_ledger"

            target/release/netherwick sim \
                --scenario "$scenario" \
                --steps "$BENCHMARK_STEPS" \
                --tick-delay-ms 0 \
                --seed "$seed" \
                --ledger "$scenario_ledger" \
                --dream-policy-checkpoint "$checkpoint" >/dev/null

            target/release/netherwick virtual-report \
                --ledger "$scenario_ledger" \
                --out "$scenario_out" >/dev/null

            idx="$((idx + 1))"
            print_progress "$idx" "$total" "benchmark"
        done
        echo "$(green "benchmark") reports: $out_dir"
    }

    ITERATION=0
    echo "$(cyan "evolve-infinite") clear=$CLEAR_VALUE benchmark_every=$BENCHMARK_EVERY export_dataset=$EXPORT_DATASET"
    while true; do
        ITERATION="$((ITERATION + 1))"
        echo "$(cyan "iteration") #$ITERATION"
        NETHERWICK_NEAT_EXPORT_DATASET="$EXPORT_DATASET" just evolve-quality clear="$CLEAR_VALUE"
        if [ "$EXPORT_DATASET" = "true" ]; then
            prune_dataset
        else
            echo "$(yellow "dataset") export disabled; skipping dataset retention"
        fi
        if [ "$BENCHMARK_EVERY" -gt 0 ] && [ $((ITERATION % BENCHMARK_EVERY)) -eq 0 ]; then
            run_benchmarks "$ITERATION"
        fi
        prune_benchmark_artifacts
    done

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
