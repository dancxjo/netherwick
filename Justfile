set dotenv-load := true

# Compatibility aliases only. Workflow implementations live in `cargo xtask`.
default *args:
    cargo run -q -p xtask -- robot {{args}}

setup:
    cargo run -q -p xtask -- setup
setup-system:
    cargo run -q -p xtask -- setup-system
setup-docker:
    cargo run -q -p xtask -- setup-docker
setup-user:
    cargo run -q -p xtask -- setup-user
setup-pico-bootsel:
    cargo run -q -p xtask -- setup-pico-bootsel
setup-rust:
    cargo run -q -p xtask -- setup-rust
setup-ort:
    cargo run -q -p xtask -- setup-ort
setup-kinect:
    cargo run -q -p xtask -- setup-kinect
setup-kinect-from-source:
    cargo run -q -p xtask -- setup-kinect-from-source
setup-tts:
    cargo run -q -p xtask -- setup-tts
setup-whisper:
    cargo run -q -p xtask -- setup-whisper
fetch:
    cargo run -q -p xtask -- fetch

forebrain-build:
    cargo run -q -p xtask -- forebrain-build
forebrain-validate config="/etc/netherwick/forebrain.toml":
    cargo run -q -p xtask -- forebrain-validate "{{config}}"
forebrain-provision inventory:
    cargo run -q -p xtask -- forebrain-provision "{{inventory}}"
forebrain-provision-check inventory:
    cargo run -q -p xtask -- forebrain-provision-check "{{inventory}}"
forebrain-remove inventory:
    cargo run -q -p xtask -- forebrain-remove "{{inventory}}"

fmt:
    cargo run -q -p xtask -- fmt
check:
    cargo run -q -p xtask -- check
test:
    cargo run -q -p xtask -- test
clippy:
    cargo run -q -p xtask -- clippy
clean:
    cargo run -q -p xtask -- clean

brainstem-build:
    cargo run -q -p xtask -- brainstem-build
brainstem-uf2:
    cargo run -q -p xtask -- brainstem-uf2
brainstem-fetch-cyw43:
    cargo run -q -p xtask -- brainstem-fetch-cyw43
brainstem-pico-w-build:
    cargo run -q -p xtask -- brainstem-pico-w-build
brainstem-pico-w-uf2:
    cargo run -q -p xtask -- brainstem-pico-w-uf2
brainstem-rpi5-build:
    cargo build --manifest-path crates/pete-brainstem/Cargo.toml --no-default-features --features rpi5 --release --bin pete-brainstem
brainstem-rpi5:
    cargo run --manifest-path crates/pete-brainstem/Cargo.toml --no-default-features --features rpi5 --bin pete-brainstem
brainstem-rpi5-check address="127.0.0.1:8787":
    cargo run -q -p pete-cockpit --example contract_check -- udp "{{address}}"
flash:
    cargo run -q -p xtask -- flash
skull:
    cargo run -q -p xtask -- skull

compose-config:
    cargo run -q -p xtask -- compose-config
compose-build:
    cargo run -q -p xtask -- compose-build
servers:
    cargo run -q -p xtask -- servers
live-server:
    cargo run -q -p xtask -- live-server
server-logs service="pete-live":
    cargo run -q -p xtask -- server-logs "{{service}}"
stop-servers:
    cargo run -q -p xtask -- stop-servers

sim:
    cargo run -q -p xtask -- sim
say text="Hello. My name is Pete.":
    cargo run -q -p xtask -- say "{{text}}"
transcribe wav:
    cargo run -q -p xtask -- transcribe "{{wav}}"
robot *args:
    cargo run -q -p xtask -- robot {{args}}
possess *args:
    cargo run -q -p xtask -- possess --mode regular {{args}}
possess-rpi5 *args:
    PETE_COCKPIT_BACKEND=local cargo run -q -p xtask -- possess --mode regular {{args}}
physical-qa *args:
    cargo run -q -p xtask -- physical-qa {{args}}
go target="virtual":
    cargo run -q -p xtask -- go "{{target}}"
go-virtual:
    cargo run -q -p xtask -- go virtual
virtual:
    cargo run -q -p xtask -- go virtual
virtual-https:
    cargo run -q -p xtask -- go virtual
train *args:
    cargo run -q -p xtask -- train {{args}}
train-virtual:
    cargo run -q -p xtask -- train virtual
evolve clear="false":
    cargo run -q -p xtask -- evolve "{{clear}}"
evolve-fast clear="false":
    cargo run -q -p xtask -- evolve "{{clear}}"
evolve-quality clear="false":
    cargo run -q -p xtask -- evolve-quality "{{clear}}"
evolve-infinite clear="false":
    cargo run -q -p xtask -- evolve-infinite "{{clear}}"
dev-cert:
    cargo run -q -p xtask -- dev-cert
virtual-url:
    cargo run -q -p xtask -- virtual-url

run *args:
    cargo run -q -p xtask -- run {{args}}
rehearse-models:
    cargo run -q -p xtask -- rehearse-models
eval-scenario-smoke:
    cargo run -q -p xtask -- eval-scenario-smoke
inspect-ledger:
    cargo run -q -p xtask -- inspect-ledger
hardware-env:
    cargo run -q -p xtask -- hardware-env
real-cockpit:
    cargo run -q -p xtask -- real-cockpit
cockpit *args:
    cargo run -q -p xtask -- cockpit {{args}}
xtask command="check":
    cargo run -q -p xtask -- {{command}}
codex-sync:
    cargo run -q -p xtask -- codex-sync
sup:
    cargo run -q -p xtask -- codex-sync
