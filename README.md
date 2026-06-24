# netherwick

`netherwick` is a Rust workspace for Pete Netherwick, an embodied self-training robot architecture.

Pete gathers raw sensors, memory recalls, internal drives, model predictions, surprise, and LLM guidance into `Now`. `Now` is compressed into an `ExperienceLatent`, used to imagine futures, choose actions, and train from consequences.

Pete acts through high-level action primitives. A hard-coded autonomic layer keeps the body safe. The LLM may command, reflect, critique, and teach, but cannot bypass safety.

Every hard-coded behavior is replaceable. It can run directly, shadow-train a model, compare with a model, promote a model, or fall back to safe hand-written logic.

Pete Netherwick is an embodied predictive organism: a robot body with reflexes, an experience ledger, compact learned present, imagined futures, memory returning as sensation, swappable learned behaviors, and an LLM consciousness that commands and teaches while safety protects the body.

## Setup

On Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install -y just
just setup
```

`just setup` installs the Linux build dependencies plus Kinect 1 userspace support through `libfreenect` when distro packages are available.

Useful commands:

```bash
just check
just test
just sim
just go virtual
just eval-scenario-smoke
just run model-status
just run compare-scenario-reports --baseline data/reports/scenario/obstacle-baseline-smoke.json --candidate data/reports/scenario/obstacle-baseline-smoke.json
just inspect-ledger
just hardware-env
```

Scenario reports can be generated with `just run eval-scenario --scenario empty-room --episodes 2 --steps 10 --out data/reports/scenario/empty-smoke.json`. See [docs/scenario-evaluation.md](docs/scenario-evaluation.md) for baseline-vs-checkpoint comparison notes and [docs/model-registry.md](docs/model-registry.md) for checkpoint registration and promotion gates.

## Hardware Bring-Up

Pete's Raspberry Pi 5 hardware path starts capture-first: inspect devices, run bounded read-only body/sensor ticks, record Worldlab captures, and inspect the result. Autonomous motor movement is not enabled.

```bash
cargo run --bin netherwick -- hardware-env
cargo run --bin netherwick -- robot --mode read-only --duration-seconds 30 --ledger data/ledger/real/read-only-smoke
cargo run --bin netherwick -- capture-real --duration-seconds 60 --out data/captures/real/rpi5-smoke
cargo run --bin netherwick -- inspect-capture data/captures/real/rpi5-smoke
```

See [docs/rpi5-bringup.md](docs/rpi5-bringup.md) for packages, permissions, device expectations, success criteria, and failure behavior.

## Docker services

Copy `.env.example` to `.env` if you want to override ports or passwords. Then:

```bash
just servers      # Neo4j + Qdrant
just live-server  # Neo4j + Qdrant + live sim server
```

The live view is at `http://localhost:8787/view`, with a 3D/WebXR sensorium at `http://localhost:8787/view/3d` and compact scene JSON at `http://localhost:8787/view/scene`. In immersive VR, supported controllers can feed short-lived Reign commands into the virtual session. Neo4j Browser is at `http://localhost:7474` with the default `.env.example` credentials.

For a one-command HTTPS virtual training theater, run:

```bash
just go virtual
```

It starts a long-running virtual sim, serves HTTPS on `0.0.0.0`, and prints desktop/headset URLs for `/view/3d` plus `/view/scene`. See [docs/go-virtual.md](docs/go-virtual.md) for certificate, headset, scenario, and LAN security notes. See [docs/webxr-viewer.md](docs/webxr-viewer.md) for desktop controls, WebXR caveats, capture replay hooks, and sensor-view privacy notes.
