set shell := ["bash", "-euxo", "pipefail", "-c"]
set dotenv-load := true

cockpit_port := env_var_or_default("PETE_COCKPIT_PORT", "auto")
gps_serial_port := env_var_or_default("GPS_SERIAL_PORT", "")
lidar_serial_port := env_var_or_default("LIDAR_SERIAL_PORT", "")
lidar_yaw_deg := env_var_or_default("LIDAR_YAW_DEG", "0")
lidar_pitch_deg := env_var_or_default("LIDAR_PITCH_DEG", "0")
lidar_roll_deg := env_var_or_default("LIDAR_ROLL_DEG", "0")
lidar_height_m := env_var_or_default("LIDAR_HEIGHT_M", "0")
lidar_forward_m := env_var_or_default("LIDAR_FORWARD_M", "0")
lidar_left_m := env_var_or_default("LIDAR_LEFT_M", "0")
camera_device := env_var_or_default("CAMERA_DEVICE", "/dev/video0")
mic_device := env_var_or_default("MIC_DEVICE", "")
imu_device := env_var_or_default("IMU_DEVICE", "")
robot_dashboard := env_var_or_default("PETE_ROBOT_DASHBOARD", "0.0.0.0:3000")
robot_dashboard_tls_cert := env_var_or_default("PETE_ROBOT_DASHBOARD_TLS_CERT", "certs/pete-dev.crt")
robot_dashboard_tls_key := env_var_or_default("PETE_ROBOT_DASHBOARD_TLS_KEY", "certs/pete-dev.key")
kinect_depth := env_var_or_default("PETE_KINECT_DEPTH", "1")
kinect_rgb_target_luma := env_var_or_default("KINECT_RGB_TARGET_LUMA", "0.32")
kinect_rgb_auto_gain_max := env_var_or_default("KINECT_RGB_AUTO_GAIN_MAX", "3.0")
kinect_rgb_gain := env_var_or_default("KINECT_RGB_GAIN", "1.0")
kinect_rgb_gamma := env_var_or_default("KINECT_RGB_GAMMA", "0.80")
kinect_rgb_brightness := env_var_or_default("KINECT_RGB_BRIGHTNESS", "0.0")
tts_output_device := env_var_or_default("PETE_TTS_OUTPUT_DEVICE", "")
cyw43_firmware_ref := env_var_or_default("CYW43_FIRMWARE_REF", "main")
pico_w_bootsel_url := env_var_or_default("PICO_W_BOOTSEL_URL", "http://192.168.4.1/command")
pico_w_mount := env_var_or_default("PICO_W_MOUNT", "")
pico_w_mount_timeout_secs := env_var_or_default("PICO_W_MOUNT_TIMEOUT_SECS", "30")

# Default to the real robot path.
default *args:
    just robot {{args}}

# Install Linux dependencies, Rust toolchain, Docker, Kinect prerequisites, Pico BOOTSEL automount, and local models.
setup: setup-system setup-docker setup-user setup-pico-bootsel setup-rust setup-kinect setup-ort setup-tts setup-whisper
    @echo "pete Linux setup complete"
    @echo "next: cargo check && just sim"

# Build the portable forebrain worker and operator CLI.
forebrain-build:
    cargo build --locked --release -p pete-higher-brain

# Validate a provisioned forebrain without requiring a GPU.
forebrain-validate config="/etc/netherwick/forebrain.toml":
    cargo run -q -p pete-higher-brain -- validate-node --config "{{config}}"

# Provision the hosts in a local, untracked Ansible inventory.
forebrain-provision inventory:
    ansible-playbook -i "{{inventory}}" provisioning/forebrain/site.yml

# Show the idempotent provisioning changes without applying them.
forebrain-provision-check inventory:
    ansible-playbook --check --diff -i "{{inventory}}" provisioning/forebrain/site.yml

# Remove services and code while preserving received data by default.
forebrain-remove inventory:
    ansible-playbook -i "{{inventory}}" provisioning/forebrain/remove.yml

# Install required system packages via apt.
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
        clang \
        libclang-dev \
        ffmpeg \
        i2c-tools \
        v4l-utils \
        libasound2-dev \
        libgomp1 \
        libssl-dev \
        libudev-dev \
        libusb-1.0-0-dev \
        libv4l-dev \
        udisks2

# Install Docker and Docker Compose.
setup-docker:
    if ! command -v docker >/dev/null 2>&1; then \
        curl -fsSL https://get.docker.com | sudo sh; \
    fi

# Configure user permissions for hardware and docker access.
setup-user:
    #!/usr/bin/env bash
    set -e
    for group in docker dialout video audio i2c plugdev; do
        if getent group "$group" >/dev/null 2>&1; then
            sudo usermod -aG "$group" "$(whoami)"
            echo "Added $(whoami) to group: $group"
        else
            echo "Group $group does not exist, skipping."
        fi
    done
    echo "User setup complete. Please reboot or log out and back in for group changes to take effect."

# Install udev/systemd integration that mounts Pico/Pico W BOOTSEL at /media/$USER/RPI-RP2.
setup-pico-bootsel:
    #!/usr/bin/env bash
    set -euo pipefail
    user="${SUDO_USER:-$(whoami)}"
    group="$(id -gn "$user")"
    sudo install -d -m 0755 /usr/local/lib/netherwick
    sudo install -m 0755 scripts/pico-bootsel-mount.sh /usr/local/lib/netherwick/pico-bootsel-mount
    sudo install -m 0644 configs/systemd/netherwick-pico-bootsel-mount@.service /etc/systemd/system/netherwick-pico-bootsel-mount@.service
    sudo install -m 0644 configs/udev/99-netherwick-pico-bootsel.rules /etc/udev/rules.d/99-netherwick-pico-bootsel.rules
    printf 'PICO_BOOTSEL_USER=%s\nPICO_BOOTSEL_GROUP=%s\nPICO_BOOTSEL_MOUNT_BASE=/media\n' "$user" "$group" \
        | sudo tee /etc/default/netherwick-pico-bootsel >/dev/null
    sudo systemctl daemon-reload
    sudo udevadm control --reload-rules
    sudo udevadm trigger --subsystem-match=block --property-match=ID_FS_LABEL=RPI-RP2 || true
    echo "Pico BOOTSEL automount installed for $user:$group at /media/$user/RPI-RP2"

# Install Rust with rustup plus embedded firmware build tools.
setup-rust:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! command -v cargo >/dev/null 2>&1; then
        curl https://sh.rustup.rs -sSf | sh -s -- -y
    fi
    if [ -f "$HOME/.cargo/env" ]; then
        # Make a freshly installed cargo/rustup visible to the rest of this recipe.
        source "$HOME/.cargo/env"
    fi
    rustup target add thumbv6m-none-eabi
    if ! command -v elf2uf2-rs >/dev/null 2>&1; then
        cargo install elf2uf2-rs
    fi

# Fetch and build Piper's self-contained Rust ONNX package.
setup-ort:
    cargo check -p pete-mouth

# Install Kinect packages from apt when available.
setup-kinect:
    if apt-cache show libfreenect-dev >/dev/null 2>&1; then \
        sudo apt-get install -y libfreenect-dev freenect; \
    else \
        echo "libfreenect-dev not found in apt metadata; run 'just setup-kinect-from-source'"; \
        exit 1; \
    fi

# Build and install libfreenect from source.
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

# Download the default Piper voice used by the robot mouth.
setup-tts:
    #!/usr/bin/env bash
    set -euo pipefail
    VOICE_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/tongues/models/piper"
    MODEL="$VOICE_DIR/en_US-ryan-medium.onnx"
    CONFIG="$VOICE_DIR/en_US-ryan-medium.onnx.json"
    mkdir -p "$VOICE_DIR"
    if [ ! -s "$MODEL" ]; then
        curl -fL --retry 3 --retry-delay 2 \
            -o "$MODEL" \
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx"
    fi
    if [ ! -s "$CONFIG" ]; then
        curl -fL --retry 3 --retry-delay 2 \
            -o "$CONFIG" \
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json"
    fi
    echo "Piper voice ready: $MODEL"

# Download the default Whisper model used by command-backed ASR.
setup-whisper:
    #!/usr/bin/env bash
    set -euo pipefail
    MODEL_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/pete/models/whisper"
    MODEL="$MODEL_DIR/ggml-tiny.en.bin"
    mkdir -p "$MODEL_DIR"
    if [ ! -s "$MODEL" ]; then
        curl -fL --retry 3 --retry-delay 2 \
            -o "$MODEL" \
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin"
    fi
    echo "Whisper model ready: $MODEL"

# Fetch local runtime/model assets without running the full system setup.
fetch: setup-tts setup-whisper

# Format all Rust code in the workspace.
fmt:
    cargo fmt --all

# Run cargo check across the whole workspace.
check:
    cargo check --workspace

# Build the brainstem firmware ELF for the configured chip backend.
brainstem-build:
    cd crates/pete-brainstem && cargo build --release

# Convert the brainstem firmware ELF to a Pico UF2 for the RP2040 backend.
brainstem-uf2: brainstem-build
    elf2uf2-rs \
        crates/pete-brainstem/target/thumbv6m-none-eabi/release/pete-brainstem \
        crates/pete-brainstem/target/thumbv6m-none-eabi/release/pete-brainstem.uf2

# Generate the Row-0 brainstem skull enclosure STL/STEP files.
skull:
    #!/usr/bin/env bash
    set -euo pipefail
    cd skeleton/brainstem
    if [ -x .venv/bin/python ]; then
        .venv/bin/python skull.py
    else
        python skull.py
    fi

# Fetch CYW43 firmware blobs required by the Pico W backend.
brainstem-fetch-cyw43:
    #!/usr/bin/env bash
    set -euo pipefail
    dir="crates/pete-brainstem/firmware/cyw43"
    base="https://raw.githubusercontent.com/embassy-rs/embassy/{{cyw43_firmware_ref}}/cyw43-firmware"
    mkdir -p "$dir"
    for file in \
        43439A0.bin \
        43439A0_clm.bin \
        nvram_rp2040.bin \
        LICENSE-permissive-binary-license-1.0.txt; do
        curl -fL --retry 3 --retry-delay 2 -o "$dir/$file" "$base/$file"
    done

# Build the Brainstem Pico W firmware with AP/status support.
brainstem-pico-w-build: brainstem-fetch-cyw43
    cd crates/pete-brainstem && cargo build --release --no-default-features --features pico-w,service-mode,operator-debug

# Convert the Brainstem Pico W firmware ELF to UF2.
brainstem-pico-w-uf2: brainstem-pico-w-build
    elf2uf2-rs \
        crates/pete-brainstem/target/thumbv6m-none-eabi/release/pete-brainstem \
        crates/pete-brainstem/target/thumbv6m-none-eabi/release/pete-brainstem-pico-w.uf2

# Build the Pico W firmware, put the board in BOOTSEL over Wi-Fi, then copy the UF2.
flash: brainstem-pico-w-uf2
    #!/usr/bin/env bash
    set -euo pipefail
    uf2="crates/pete-brainstem/target/thumbv6m-none-eabi/release/pete-brainstem-pico-w.uf2"
    bootsel_url="{{pico_w_bootsel_url}}"
    mount_override="{{pico_w_mount}}"
    timeout_secs="{{pico_w_mount_timeout_secs}}"

    if [ ! -s "$uf2" ]; then
        echo "UF2 not found: $uf2" >&2
        exit 1
    fi

    find_rpi_rp2_mount() {
        local candidate
        if [ -n "$mount_override" ] && [ -d "$mount_override" ] && [ -w "$mount_override" ]; then
            printf '%s\n' "$mount_override"
            return 0
        fi
        for candidate in \
            "/media/$USER/RPI-RP2" \
            "/run/media/$USER/RPI-RP2" \
            "/media/RPI-RP2" \
            "/Volumes/RPI-RP2"; do
            if [ -d "$candidate" ] && [ -w "$candidate" ]; then
                printf '%s\n' "$candidate"
                return 0
            fi
        done
        if command -v lsblk >/dev/null 2>&1; then
            while IFS= read -r candidate; do
                if [ -n "$candidate" ] && [ -d "$candidate" ] && [ -w "$candidate" ]; then
                    printf '%s\n' "$candidate"
                    return 0
                fi
            done < <(lsblk -rpo LABEL,MOUNTPOINT | awk '$1 == "RPI-RP2" { print $2 }')
        fi
        return 1
    }

    find_rpi_rp2_block() {
        if command -v lsblk >/dev/null 2>&1; then
            lsblk -rnpo LABEL,PATH,FSTYPE,MOUNTPOINT | awk '$1 == "RPI-RP2" && $3 == "vfat" && $4 == "" { print $2; exit }'
        fi
    }

    find_rpi_rp2_block_any() {
        if command -v lsblk >/dev/null 2>&1; then
            lsblk -rnpo LABEL,PATH,FSTYPE | awk '$1 == "RPI-RP2" && $3 == "vfat" { print $2; exit }'
        fi
    }

    mount_rpi_rp2() {
        local block existing_mount mount_dir
        block="$(find_rpi_rp2_block)"
        if [ -z "$block" ]; then
            block="$(find_rpi_rp2_block_any)"
        fi
        if [ -z "$block" ]; then
            return 1
        fi

        echo "Mounting RPI-RP2 from $block" >&2
        if command -v udisksctl >/dev/null 2>&1 && udisksctl mount -b "$block" >/dev/null 2>&1; then
            find_rpi_rp2_mount
            return 0
        fi

        mount_dir="/media/$USER/RPI-RP2"
        sudo mkdir -p "$mount_dir"
        existing_mount="$(lsblk -rnpo PATH,MOUNTPOINT "$block" | awk '$1 == "'"$block"'" { print $2; exit }')"
        if [ -n "$existing_mount" ] && [ "$existing_mount" != "$mount_dir" ]; then
            sudo umount "$existing_mount"
        fi
        if mountpoint -q "$mount_dir" && [ ! -w "$mount_dir" ]; then
            sudo umount "$mount_dir"
        fi
        if ! mountpoint -q "$mount_dir"; then
            sudo mount -t vfat -o "uid=$(id -u),gid=$(id -g),umask=022" "$block" "$mount_dir"
        fi
        find_rpi_rp2_mount
    }

    mount_path=""
    if mount_path="$(find_rpi_rp2_mount)"; then
        echo "RPI-RP2 already mounted at $mount_path; skipping BOOTSEL request"
    elif mount_path="$(mount_rpi_rp2)"; then
        echo "RPI-RP2 mounted at $mount_path; skipping BOOTSEL request"
    else
        echo "Requesting authorized BOOTSEL via $bootsel_url"
        host="${bootsel_url#http://}"
        host="${host%%/*}"
        if [[ "$host" != *:* ]]; then
            host="$host:80"
        fi
        if ! PETE_BRAINSTEM_HTTP_HOST="$host" cargo run -q -p pete-cockpit --example service_bootsel \
            && ! PETE_BOOTSEL_USB=1 cargo run -q -p pete-cockpit --example service_bootsel; then
            if [ "${PETE_ALLOW_LEGACY_BOOTSEL:-0}" != "1" ]; then
                echo "Authorized BOOTSEL failed; legacy fallback is disabled." >&2
                echo "Set PETE_ALLOW_LEGACY_BOOTSEL=1 only for explicit development recovery." >&2
                exit 1
            fi
            echo "WARNING: USING UNAUDITED LEGACY BOOTSEL DEVELOPMENT FALLBACK" >&2
            curl -fsS --max-time 3 -H "Content-Type: application/json" \
                -d '{"kind":"bootsel","command_id":1}' "$bootsel_url" >/dev/null
        fi

        echo "Waiting for RPI-RP2 mount"
        deadline=$((SECONDS + timeout_secs))
        while [ "$SECONDS" -lt "$deadline" ]; do
            if mount_path="$(find_rpi_rp2_mount)"; then
                break
            fi
            if mount_path="$(mount_rpi_rp2)"; then
                break
            fi
            sleep 1
        done
    fi

    if [ -z "$mount_path" ]; then
        echo "Timed out waiting for RPI-RP2. Set PICO_W_MOUNT=/path/to/RPI-RP2 if it mounted somewhere unusual." >&2
        exit 1
    fi

    echo "Copying $uf2 to $mount_path"
    cp "$uf2" "$mount_path/"
    sync
    echo "Flash copy complete"

# Render merged docker-compose configuration.
compose-config:
    docker compose config

# Build the pete-live compose image.
compose-build:
    docker compose build pete-live

# Start shared backing services (neo4j and qdrant).
servers:
    docker compose up -d neo4j qdrant

# Ensure shared graph and vector memory services are reachable.
_ensure-memory-servers:
    #!/usr/bin/env bash
    set -euo pipefail

    neo4j_url="${PETE_NEO4J_HTTP_URL:-http://127.0.0.1:${PETE_NEO4J_HTTP_PORT:-7474}}"
    qdrant_url="${PETE_QDRANT_URL:-http://127.0.0.1:${PETE_QDRANT_HTTP_PORT:-6333}}"
    neo4j_url="${neo4j_url%/}"
    qdrant_url="${qdrant_url%/}"

    neo4j_ready() {
        curl -fsS --max-time 2 "$neo4j_url/" >/dev/null 2>&1
    }

    qdrant_ready() {
        curl -fsS --max-time 2 "$qdrant_url/readyz" >/dev/null 2>&1 \
            || curl -fsS --max-time 2 "$qdrant_url/collections" >/dev/null 2>&1
    }

    if neo4j_ready && qdrant_ready; then
        exit 0
    fi

    echo "Graph/vector memory services are not reachable; starting local Neo4j and Qdrant with just servers."
    just servers

    deadline=$((SECONDS + ${PETE_MEMORY_SERVER_WAIT_SECS:-90}))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if neo4j_ready && qdrant_ready; then
            echo "Graph/vector memory services are ready."
            exit 0
        fi
        sleep 2
    done

    echo "Timed out waiting for Neo4j at $neo4j_url and Qdrant at $qdrant_url." >&2
    echo "Check 'docker compose ps neo4j qdrant' and 'just server-logs neo4j' / 'just server-logs qdrant'." >&2
    exit 1

# Start backing services plus the pete-live container.
live-server:
    docker compose --profile pete up -d neo4j qdrant pete-live

# Follow docker compose logs for a selected service.
server-logs service="pete-live":
    docker compose logs -f {{service}}

# Stop and remove compose services.
stop-servers:
    docker compose down

# Run workspace tests.
test:
    cargo test --workspace

# Lint with clippy and fail on warnings.
clippy:
    cargo clippy --workspace --all-targets -- -D warnings

# Run simulator mode through pete-tools.
sim:
    cargo run -p pete-tools -- sim

# Speak through the robot mouth without starting the robot body or sensors.
say text="Hello. My name is Pete.":
    PETE_TTS_OUTPUT_DEVICE="{{tts_output_device}}" cargo run -p pete-tools -- mouth "{{text}}"

# Transcribe a WAV file through the local Whisper ASR path.
transcribe wav:
    cargo run -p pete-tools -- whisper-transcribe "{{wav}}"

# Select the real Cockpit transport. Explicit configuration wins; otherwise
# prefer a responsive USB/UART brainstem and use Wi-Fi only while attached to
# one of the brainstem's pete-* access points.
[private]
_robot-cockpit-backend:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -n "${PETE_COCKPIT_BACKEND:-}" ]; then
        printf '%s\n' "$PETE_COCKPIT_BACKEND"
        exit 0
    fi

    PORT="{{cockpit_port}}"
    if [ "$PORT" = auto ]; then
        shopt -s nullglob
        CANDIDATES=(/dev/serial/by-id/*Pete_Brainstem* /dev/ttyACM* /dev/ttyUSB*)
        PORT="${CANDIDATES[0]:-}"
    fi
    if [ "${PETE_SKIP_COCKPIT_UART:-0}" != 1 ] \
        && [ -n "$PORT" ] \
        && cargo run -q -p pete-cockpit --example contract_check -- uart "$PORT" >/dev/null 2>&1; then
        printf 'uart\n'
        exit 0
    fi

    PETE_SSID=""
    if command -v nmcli >/dev/null 2>&1; then
        PETE_SSID="$(
            nmcli -t -f active,ssid dev wifi 2>/dev/null \
                | sed -n 's/^yes://p' \
                | awk 'tolower($0) ~ /^pete-/ { print; exit }' \
                || true
        )"
    elif command -v iwgetid >/dev/null 2>&1; then
        SSID="$(iwgetid -r 2>/dev/null || true)"
        if [[ "${SSID,,}" == pete-* ]]; then PETE_SSID="$SSID"; fi
    fi
    if [ -n "$PETE_SSID" ] \
        && command -v curl >/dev/null 2>&1 \
        && curl -fsS --connect-timeout 1 --max-time 2 \
            "http://${PETE_BRAINSTEM_HTTP_HOST:-192.168.4.1:80}/status.json" >/dev/null; then
        printf 'wifi\n'
        exit 0
    fi

    # Preserve the existing UART behavior (including its diagnostics) when no
    # verified real transport is available.
    if [ "${PETE_SKIP_COCKPIT_UART:-0}" = 1 ]; then
        printf 'none\n'
    else
        printf 'uart\n'
    fi

# Bring up the real robot read-only by default with hardware auto-detection.
robot *args:
    #!/usr/bin/env bash
    set -euo pipefail
    just --quiet _ensure-memory-servers
    CAMERA_ARGS=()
    MIC_ARGS=()
    IMU_ARGS=()
    GPS_ARGS=()
    LIDAR_ARGS=()
    KINECT_ARGS=()
    if [ -n "{{camera_device}}" ]; then
        CAMERA_ARGS+=(--camera "{{camera_device}}")
    fi
    if [ -n "{{mic_device}}" ]; then
        MIC_ARGS+=(--mic "{{mic_device}}")
    fi
    if [ -n "{{imu_device}}" ]; then
        IMU_ARGS+=(--imu "{{imu_device}}")
    fi
    if [ -n "{{gps_serial_port}}" ]; then
        GPS_ARGS+=(--gps "{{gps_serial_port}}")
    fi
    if [ -n "{{lidar_serial_port}}" ]; then
        LIDAR_ARGS+=(
            --lidar "{{lidar_serial_port}}"
            --lidar-yaw-deg "{{lidar_yaw_deg}}"
            --lidar-pitch-deg "{{lidar_pitch_deg}}"
            --lidar-roll-deg "{{lidar_roll_deg}}"
            --lidar-height-m "{{lidar_height_m}}"
            --lidar-forward-m "{{lidar_forward_m}}"
            --lidar-left-m "{{lidar_left_m}}"
        )
    fi
    if [ "{{kinect_depth}}" = "1" ] || [ "{{kinect_depth}}" = "true" ] || [ "{{kinect_depth}}" = "on" ]; then
        KINECT_ARGS+=(
            --kinect-depth
            --kinect-rgb-target-luma "{{kinect_rgb_target_luma}}"
            --kinect-rgb-auto-gain-max "{{kinect_rgb_auto_gain_max}}"
            --kinect-rgb-gain "{{kinect_rgb_gain}}"
            --kinect-rgb-gamma "{{kinect_rgb_gamma}}"
            --kinect-rgb-brightness "{{kinect_rgb_brightness}}"
        )
    fi
    if [ ! -f "{{robot_dashboard_tls_cert}}" ] || [ ! -f "{{robot_dashboard_tls_key}}" ]; then
        mkdir -p "$(dirname "{{robot_dashboard_tls_cert}}")" "$(dirname "{{robot_dashboard_tls_key}}")"
        LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
        if [ -z "$LAN_IP" ]; then
            LAN_SAN=""
        else
            LAN_SAN=",IP:$LAN_IP"
        fi
        openssl req -x509 -newkey rsa:2048 -nodes \
            -keyout "{{robot_dashboard_tls_key}}" \
            -out "{{robot_dashboard_tls_cert}}" \
            -days 365 \
            -subj "/CN=pete.local" \
            -addext "subjectAltName=DNS:localhost,DNS:pete.local,IP:127.0.0.1$LAN_SAN"
    fi
    COCKPIT_BACKEND="$(just --quiet _robot-cockpit-backend)"
    if [ -z "${PETE_COCKPIT_BACKEND:-}" ] && [ "$COCKPIT_BACKEND" = wifi ]; then
        echo "Cockpit USB/UART unavailable; using Pete brainstem Wi-Fi at ${PETE_BRAINSTEM_HTTP_HOST:-192.168.4.1:80}."
    fi
    PETE_TTS_OUTPUT_DEVICE="{{tts_output_device}}" cargo run -p pete-tools -- robot \
        --mode "${PETE_ROBOT_MODE:-read-only}" \
        --cockpit "$COCKPIT_BACKEND" \
        --create-port "{{cockpit_port}}" \
        --ledger "${PETE_ROBOT_LEDGER:-data/ledger/real/robot}" \
        "${CAMERA_ARGS[@]}" \
        "${KINECT_ARGS[@]}" \
        "${MIC_ARGS[@]}" \
        "${IMU_ARGS[@]}" \
        "${GPS_ARGS[@]}" \
        "${LIDAR_ARGS[@]}" \
        --dashboard "{{robot_dashboard}}" \
        --dashboard-tls \
        --dashboard-tls-cert "{{robot_dashboard_tls_cert}}" \
        --dashboard-tls-key "{{robot_dashboard_tls_key}}" \
        {{args}}

# Take and maintain motherbrain possession using the pinned physical brainstem.
possess *args:
    #!/usr/bin/env bash
    set -euo pipefail
    : "${PETE_BRAINSTEM_DEVICE_ID:?set PETE_BRAINSTEM_DEVICE_ID in .env}"

    set_env_var() {
        local key="$1" value="$2" escaped
        escaped="$(printf '%s' "$value" | sed 's/[\&/]/\\&/g')"
        if [ -f .env ] && grep -q "^${key}=" .env; then
            sed -i "s/^${key}=.*/${key}=${escaped}/" .env
        else
            printf '\n%s=%s\n' "$key" "$value" >> .env
        fi
    }

    detect_single_brainstem_port() {
        shopt -s nullglob
        local candidates=(/dev/serial/by-id/*Pete_Brainstem*)
        shopt -u nullglob
        if [ "${#candidates[@]}" -eq 1 ]; then
            printf '%s\n' "${candidates[0]}"
        fi
    }

    refresh_brainstem_pin_from_usb() {
        local detected_port bootstrap_log live_device_id live_boot_id
        detected_port="$(detect_single_brainstem_port || true)"
        if [ -z "$detected_port" ]; then
            return 1
        fi
        bootstrap_log="$(mktemp)"
        if ! PETE_BRAINSTEM_DEVICE_ID= cargo run -q -p pete-cockpit --example motherbrain_bootstrap -- --identity-only >"$bootstrap_log" 2>&1; then
            cat "$bootstrap_log" >&2
            rm -f "$bootstrap_log"
            return 1
        fi
        live_device_id="$(sed -n 's/^brainstem identity: //p' "$bootstrap_log" | tail -n 1)"
        live_boot_id="$(sed -n 's/^brainstem boot: //p' "$bootstrap_log" | tail -n 1)"
        cat "$bootstrap_log"
        rm -f "$bootstrap_log"
        if [ -z "$live_device_id" ] || [ -z "$live_boot_id" ]; then
            echo "USB bootstrap did not report a usable brainstem identity." >&2
            return 1
        fi
        if [ "$live_device_id" != "$PETE_BRAINSTEM_DEVICE_ID" ] \
            && [ "${PETE_ACCEPT_BRAINSTEM_REPLACEMENT:-0}" != "1" ]; then
            echo "Detected brainstem $live_device_id at $detected_port, but .env pins $PETE_BRAINSTEM_DEVICE_ID." >&2
            echo "To accept this replacement over the wired USB link, rerun:" >&2
            echo "  PETE_ACCEPT_BRAINSTEM_REPLACEMENT=1 just possess" >&2
            return 1
        fi
        set_env_var PETE_BRAINSTEM_DEVICE_ID "$live_device_id"
        set_env_var PETE_BRAINSTEM_BOOT_ID "$live_boot_id"
        set_env_var PETE_COCKPIT_PORT "$detected_port"
        export PETE_BRAINSTEM_DEVICE_ID="$live_device_id"
        export PETE_BRAINSTEM_BOOT_ID="$live_boot_id"
        export PETE_COCKPIT_PORT="$detected_port"
        BOOT_ID="$live_boot_id"
        echo "Updated .env brainstem pin from wired USB: device=$live_device_id boot=$live_boot_id port=$detected_port"
    }

    if [ -n "${PETE_COCKPIT_PORT:-}" ] && [ "${PETE_COCKPIT_PORT:-}" != auto ] && [ ! -e "$PETE_COCKPIT_PORT" ]; then
        DETECTED_PORT="$(detect_single_brainstem_port || true)"
        if [ -n "$DETECTED_PORT" ]; then
            echo "Configured PETE_COCKPIT_PORT is missing: $PETE_COCKPIT_PORT"
            echo "Detected one wired brainstem candidate: $DETECTED_PORT"
            refresh_brainstem_pin_from_usb
        fi
    fi

    BACKEND_WAS_EXPLICIT=0
    if [ -n "${PETE_COCKPIT_BACKEND:-}" ]; then BACKEND_WAS_EXPLICIT=1; fi
    COCKPIT_BACKEND="$(just --quiet _robot-cockpit-backend)"
    export PETE_COCKPIT_BACKEND="$COCKPIT_BACKEND"
    BOOT_ID="${PETE_BRAINSTEM_BOOT_ID:-unknown}"
    run_possession() {
        echo "Taking brainstem possession over ${PETE_COCKPIT_BACKEND:-wifi} at ${PETE_BRAINSTEM_HTTP_HOST:-192.168.4.1:80}"
        echo "device=$PETE_BRAINSTEM_DEVICE_ID boot=$BOOT_ID"
        echo "limits: 50 mm/s linear, 500 mrad/s angular; exit performs STOP then exorcize"
        PETE_ROBOT_MODE=possession-slow \
        PETE_COCKPIT_BACKEND="${PETE_COCKPIT_BACKEND:-wifi}" \
        PETE_ROBOT_LEDGER="${PETE_POSSESSION_LEDGER:-data/ledger/real/possession}" \
        CAMERA_DEVICE="" MIC_DEVICE="" IMU_DEVICE="" GPS_SERIAL_PORT="" PETE_KINECT_DEPTH=0 \
        just robot \
            --brainstem-device-id "$PETE_BRAINSTEM_DEVICE_ID" \
            --brainstem-boot-id "$BOOT_ID" \
            --max-linear-mm-s 50 \
            --max-angular-mrad-s 500 \
            --autonomous-motion \
            --imu none --gps none \
            --llm-provider disabled \
            --capture "${PETE_POSSESSION_CAPTURE:-data/captures/real/possession}" {{args}}
    }

    LOG="$(mktemp)"
    trap 'rm -f "$LOG"' EXIT
    set +e
    run_possession 2>&1 | tee "$LOG"
    STATUS=${PIPESTATUS[0]}
    set -e
    if [ "$STATUS" -eq 0 ]; then exit 0; fi

    if [ "$BACKEND_WAS_EXPLICIT" -eq 0 ] && [ "$COCKPIT_BACKEND" = uart ]; then
        WIFI_BACKEND="$(PETE_COCKPIT_BACKEND= PETE_SKIP_COCKPIT_UART=1 just --quiet _robot-cockpit-backend)"
        if [ "$WIFI_BACKEND" = wifi ]; then
            COCKPIT_BACKEND=wifi
            export PETE_COCKPIT_BACKEND="$COCKPIT_BACKEND"
            echo "Brainstem USB/UART failed; retrying possession over Pete Wi-Fi at ${PETE_BRAINSTEM_HTTP_HOST:-192.168.4.1:80}."
            : > "$LOG"
            set +e
            run_possession 2>&1 | tee "$LOG"
            STATUS=${PIPESTATUS[0]}
            set -e
            if [ "$STATUS" -eq 0 ]; then exit 0; fi
        fi
    fi

    if [ "$COCKPIT_BACKEND" = wifi ] \
        && grep -q 'reason_code: InvalidIdentity' "$LOG"; then
        echo "Wi-Fi identity continuity is not established; bootstrapping the pinned brainstem over USB."
        cargo run -q -p pete-cockpit --example motherbrain_bootstrap -- --identity-only
        : > "$LOG"
        set +e
        run_possession 2>&1 | tee "$LOG"
        STATUS=${PIPESTATUS[0]}
        set -e
        if [ "$STATUS" -eq 0 ]; then exit 0; fi
    fi

    LIVE_BOOT_ID="$(sed -n 's/^Error: brainstem boot identity mismatch: expected .* received \([^[:space:]]*\)$/\1/p' "$LOG" | tail -n 1)"
    if [ -z "$LIVE_BOOT_ID" ]; then exit "$STATUS"; fi

    set_env_var PETE_BRAINSTEM_BOOT_ID "$LIVE_BOOT_ID"
    BOOT_ID="$LIVE_BOOT_ID"
    export PETE_BRAINSTEM_BOOT_ID="$BOOT_ID"
    echo "Accepted current boot identity for pinned device $PETE_BRAINSTEM_DEVICE_ID; updated .env and retrying."
    run_possession

# Launch the virtual dream world with HTTPS live view.
go target="virtual":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{target}}" != "virtual" ]; then
        echo "usage: just go virtual"
        exit 2
    fi
    just dev-cert
    PORT="${PETE_LIVE_PORT:-8787}"
    LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
    if [ -z "$LAN_IP" ]; then LAN_IP="127.0.0.1"; fi
    echo
    echo "Pete Dream World is starting."
    echo "Virtual training theater is collecting experience."
    echo "Inline learning defaults to world-outcome. Set PETE_INLINE_LEARNING_MODE=off for collect-only."
    echo "Dream controller mode: mechanical Reign passthrough."
    echo "Models shadow the mechanics; no NEAT/evolution controller is used for live control."
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
    cargo build -p pete-tools
    SIM_DREAM_POLICY_ARGS=()
    ACTION_SELECTOR="${PETE_ACTION_SELECTOR:-baseline}"
    if [ -n "${PETE_DREAM_POLICY_CHECKPOINT:-}" ] || [ "${PETE_AUTO_DREAM_POLICY:-0}" = "1" ]; then
        echo "Visible Dream World controller: mechanical Reign passthrough (Dream NEAT ignored)"
    fi
    echo "Visible Dream World action selector: $ACTION_SELECTOR"
    target/debug/pete sim \
        --live \
        --live-tls \
        --live-addr "0.0.0.0:$PORT" \
        --live-tls-cert certs/pete-dev.crt \
        --live-tls-key certs/pete-dev.key \
        --action-selector "$ACTION_SELECTOR" \
        --scenario "${PETE_SCENARIO:-dream}" \
        --steps "${PETE_SIM_STEPS:-1000000000}" \
        --tick-delay-ms "${PETE_TICK_DELAY_MS:-100}" \
        --ledger "${PETE_LEDGER:-data/ledger/virtual-live}" \
        "${SIM_DREAM_POLICY_ARGS[@]}" &
    PID="$!"
    trap 'kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true' INT TERM EXIT
    if [ "${PETE_OPEN_BROWSER:-0}" = "1" ] && command -v xdg-open >/dev/null 2>&1; then
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

# Alias for `just go virtual`.
go-virtual:
    just go virtual

# Alias for `just go virtual`.
virtual:
    just go virtual

# Alias for `just go virtual` (legacy naming).
virtual-https:
    just go virtual

# Train models from virtual ledger data.
train target="virtual":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{target}}" != "virtual" ]; then
        echo "usage: just train virtual"
        exit 2
    fi
    cargo run -p pete-tools -- train virtual \
        --ledger "${PETE_LEDGER:-data/ledger/virtual-live}" \
        --out-dir "${PETE_MODEL_OUT:-data/models/virtual/latest}" \
        --report-out "${PETE_REPORT_OUT:-data/reports/virtual/latest.json}" \
        --epochs "${PETE_EPOCHS:-5}"

# Alias for `just train virtual`.
train-virtual:
    just train virtual

# Run Dream NEAT evolution in fast detailed mode.
evolve clear="false":
    #!/usr/bin/env bash
    set -euo pipefail
    START_LEVEL="${PETE_NEAT_START_LEVEL:-motion}"
    GENERATIONS="${PETE_NEAT_GENERATIONS:-12}"
    POPULATION="${PETE_NEAT_POPULATION:-24}"
    if [ -n "${PETE_NEAT_SEED:-}" ]; then
        SEED="${PETE_NEAT_SEED}"
    else
        SEED="$(date +%s)"
    fi
    HIDDEN_DIM="${PETE_NEAT_HIDDEN_DIM:-10}"
    CHECKPOINT_DIR="${PETE_NEAT_CHECKPOINT_DIR:-data/models/dream-policy/neat}"
    DATASET_DIR="${PETE_NEAT_DATASET_DIR:-datasets/dream-policy/v0/episodes}"
    EXPORT_DATASET="${PETE_NEAT_EXPORT_DATASET:-false}"
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

    cargo build --release -p pete-tools
    target/release/pete dream-train \
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

# Alias for `just evolve`.
evolve-fast clear="false":
    just evolve clear={{clear}}

# Run a higher-quality Dream NEAT evolution profile.
evolve-quality clear="false":
    #!/usr/bin/env bash
    set -euo pipefail
    START_LEVEL="${PETE_NEAT_START_LEVEL:-motion}"
    GENERATIONS="${PETE_NEAT_QUALITY_GENERATIONS:-36}"
    POPULATION="${PETE_NEAT_QUALITY_POPULATION:-64}"
    if [ -n "${PETE_NEAT_SEED:-}" ]; then
        SEED="${PETE_NEAT_SEED}"
    else
        SEED="$(date +%s)"
    fi
    HIDDEN_DIM="${PETE_NEAT_QUALITY_HIDDEN_DIM:-14}"
    CHECKPOINT_DIR="${PETE_NEAT_CHECKPOINT_DIR:-data/models/dream-policy/neat}"
    DATASET_DIR="${PETE_NEAT_DATASET_DIR:-datasets/dream-policy/v0/episodes}"
    EXPORT_DATASET="${PETE_NEAT_EXPORT_DATASET:-false}"
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

    cargo build --release -p pete-tools
    target/release/pete dream-train \
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

# Continuously run evolve-quality with retention and periodic benchmarks.
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

    DATASET_DIR="${PETE_NEAT_DATASET_DIR:-datasets/dream-policy/v0/episodes}"
    EXPORT_DATASET="${PETE_NEAT_EXPORT_DATASET:-false}"
    CHECKPOINT_DIR="${PETE_NEAT_CHECKPOINT_DIR:-data/models/dream-policy/neat}"
    BENCHMARK_EVERY="${PETE_EVOLVE_BENCHMARK_EVERY:-10}"
    BENCHMARK_STEPS="${PETE_EVOLVE_BENCHMARK_STEPS:-160}"
    BENCHMARK_ROOT="${PETE_EVOLVE_BENCHMARK_ROOT:-data/reports/scenario/evolve}"
    BENCHMARK_LEDGER_ROOT="${PETE_EVOLVE_BENCHMARK_LEDGER_ROOT:-data/ledger/evolve-benchmark}"

    BENCHMARK_MAX_RUNS="${PETE_EVOLVE_BENCHMARK_MAX_RUNS:-64}"
    BENCHMARK_MAX_AGE_DAYS="${PETE_EVOLVE_BENCHMARK_MAX_AGE_DAYS:-21}"

    DATASET_MAX_FILES="${PETE_DATASET_MAX_FILES:-8000}"
    DATASET_MAX_BYTES="${PETE_DATASET_MAX_BYTES:-536870912}"
    DATASET_MAX_AGE_DAYS="${PETE_DATASET_MAX_AGE_DAYS:-10}"

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

            target/release/pete sim \
                --scenario "$scenario" \
                --steps "$BENCHMARK_STEPS" \
                --tick-delay-ms 0 \
                --seed "$seed" \
                --ledger "$scenario_ledger" \
                --dream-policy-checkpoint "$checkpoint" >/dev/null

            target/release/pete virtual-report \
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
        PETE_NEAT_EXPORT_DATASET="$EXPORT_DATASET" just evolve-quality clear="$CLEAR_VALUE"
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

# Create or reuse local development TLS certificates.
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
    if [ ! -f certs/pete-dev.crt ] || [ ! -f certs/pete-dev.key ]; then
        openssl req -x509 -newkey rsa:2048 -nodes \
            -keyout certs/pete-dev.key \
            -out certs/pete-dev.crt \
            -days 365 \
            -subj "/CN=pete.local" \
            -addext "subjectAltName=DNS:localhost,DNS:pete.local,IP:127.0.0.1$LAN_SAN"
        echo "generated certs/pete-dev.crt and certs/pete-dev.key"
    else
        echo "using existing certs/pete-dev.crt and certs/pete-dev.key"
    fi

# Print the LAN URL for the virtual 3D HTTPS view.
virtual-url:
    #!/usr/bin/env bash
    set -euo pipefail
    PORT="${PETE_LIVE_PORT:-8787}"
    LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
    if [ -z "$LAN_IP" ]; then LAN_IP="127.0.0.1"; fi
    echo "https://$LAN_IP:$PORT/view/3d"

# Pass arbitrary arguments through to pete-tools.
run *args:
    cargo run -p pete-tools -- {{args}}

# Run an end-to-end model rehearsal sequence.
rehearse-models:
    just run sim --steps 200 --ledger data/ledger/sim1
    just run train behavior danger --ledger data/ledger/sim1
    just run train behavior charge --ledger data/ledger/sim1
    just run train behavior future --ledger data/ledger/sim1
    just run evaluate behavior danger --ledger data/ledger/sim1
    just run model-status
    just run sim --steps 200 --danger-checkpoint data/models/danger_v0 --danger-mode shadow-infer
    just run robot --mode read-only --cockpit sim --steps 20 --capture data/captures/mock-readonly
    just run replay-capture --capture data/captures/mock-readonly

# Run lightweight scenario evaluations for quick validation.
eval-scenario-smoke:
    just run eval-scenario --scenario empty-room --episodes 2 --steps 10 --out data/reports/scenario/empty-smoke.json
    just run eval-scenario --scenario obstacle-avoidance --episodes 2 --steps 10 --out data/reports/scenario/obstacle-smoke.json
    just run eval-scenario --scenario corner-trap --episodes 1 --steps 40 --out data/reports/scenario/corner-trap-smoke.json
    just run eval-scenario --scenario charger-seeking --episodes 2 --steps 10 --out data/reports/scenario/charge-smoke.json

# Inspect ledger contents with the tools CLI.
inspect-ledger:
    cargo run -p pete-tools -- inspect-ledger

# Run workspace automation commands from xtask.
xtask command="check":
    cargo run -p xtask -- {{command}}

# Show cockpit wiring status and selected serial port.
real-cockpit:
    @echo "Cockpit backend: ${PETE_COCKPIT_BACKEND:-uart}"
    @echo "Cockpit port: {{cockpit_port}}"

# Print detected hardware-related environment status.
hardware-env:
    cargo run -p pete-tools -- hardware-env

# Remove build artifacts.
clean:
    cargo clean
