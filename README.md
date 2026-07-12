# pete

`pete` is a Rust workspace for Pete, an embodied self-training robot architecture.

Pete now has a working real-world perception path: Kinect RGB and depth data can be fused into aligned, Minecrafty but properly colored 3D voxels that correspond to real objects in real space. The current milestone is important because Pete is no longer only passing sensor packets around. It is beginning to hold a visible world model that a human can inspect, debug, and eventually enter through the WebXR viewer.

The project still keeps the larger PETE architecture in view. Pete gathers raw sensors, memory recalls, internal drives, model predictions, surprise, and LLM guidance into `Now`. `Now` is compressed into an `ExperienceLatent`, used to imagine futures, choose actions, and train from consequences.

Pete acts through high-level action primitives. A hard-coded autonomic layer keeps the body safe. The LLM may command, reflect, critique, and teach, but cannot bypass safety.

The LLM loop is now active as a trainer, critic, and planner. It predicts counterfactual outcomes, critiques training data, and suggests motion intents. Movement itself is still not responding downstream, so the current debugging target is the command-to-base path: safety vetoes, robot mode, stale base connection, controller regression, or body-state refusal.

Every hard-coded behavior is replaceable. It can run directly, shadow-train a model, compare with a model, promote a model, or fall back to safe hand-written logic.

Pete is an embodied predictive organism: a robot body with reflexes, an experience ledger, compact learned present, imagined futures, memory returning as sensation, swappable learned behaviors, an LLM consciousness that commands and teaches, and safety that protects the body.

## Current milestone

The live perception stack has matured from loose point clouds and transient hypotheses into an inspectable 3D voxel world:

- Kinect depth and RGB are being aligned into colored voxels.
- The voxel output is coarse, blocky, and intentionally debuggable.
- Visible structures in the voxel scene correspond to real things in the room.
- The 3D/WebXR viewer is the main inspection surface for the current world model.
- The 2D map path needs restoration after drifting out of alignment and then failing.
- Movement is temporarily under investigation: commands may be blocked by a safety veto, dropped by a controller path, or blocked by robot mode/state.

The next good artifact is a capture set: screenshots, video capture, and a short golden run containing RGB, depth, IMU, odometry, command traces, safety decisions, and map/voxel outputs from the same session.

## Architecture sketch

```text
sensors
  -> synchronized RGB/depth/IMU/body events
  -> point cloud / voxel projection
  -> Features
  -> 3D live view and WebXR inspection
  -> 2D map / occupancy surface
  -> cross-modal constellations
  -> object, place, affordance, and action hypotheses
  -> Now / ExperienceLatent
  -> prediction, memory, action, and training loops
```

The present engineering emphasis is not beauty. It is spatial trust. A crude voxel world that stays aligned is more useful than a polished render that cannot be believed.

See [docs/013-feature-registry.md](docs/013-feature-registry.md) for the universal observation layer: everything Pete observes should become a Feature before new learning systems consume it.

## Constellations

A **constellation** is a repeatable pattern of experience, not merely a visual cluster. The first obvious constellations come from nearby colored voxels, planes, corners, and depth edges, but the abstraction should generalize across all modalities:

- geometry: points, voxels, surfaces, occupancy, relative position,
- color and image evidence,
- motion and odometry,
- robot body state,
- audio and speech events,
- text labels and image descriptions,
- memory recalls,
- prediction error and surprise,
- LLM counterfactuals, critiques, and suggested actions.

The search target is not yet "chair" or "kitchen." The search target is "I have seen this arrangement before." Once a constellation survives time, viewpoint changes, lighting changes, motion, and critique, it can be promoted into an object, place, affordance, or training specimen.

See [docs/constellations.md](docs/constellations.md) for the generalized pattern-search model.

## Active debugging tracks

### 3D voxel world

The voxel path is currently the strongest signal. Keep it capture-first and regression-friendly:

```bash
just live-server
# open the 3D/WebXR view printed by the server, or visit /view/3d
```

When the world looks right, save screenshots and a short video. When it looks wrong, preserve the input capture rather than only the rendered failure.

### 2D map restoration

The 2D map previously drifted out of alignment and then stopped working. Treat it as a derived product of the same spatial truth used by the voxel view:

1. Verify that the coordinate frame used for voxel projection is the same frame the 2D map consumes.
2. Confirm that map updates are still being emitted.
3. Check whether the map renderer is alive but receiving empty or invalid data.
4. Compare a known-good capture against the current map path.

### Movement restoration

Movement is expected to remain conservative. Debug it from command intent outward:

1. Did a movement intent get generated?
2. Did the controller receive it?
3. Did the safety layer veto it?
4. Is the base in the correct mode to accept motion?
5. Did the robot report a fault, dock state, cliff signal, bumper signal, or stale serial/body connection?

Do not bypass safety just to make the wheels turn. The goal is to make the vetoes inspectable so the system can explain why the body refuses to move.

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

## Hardware bring-up

Pete's Raspberry Pi 5 hardware path starts capture-first: inspect devices, run bounded read-only body/sensor ticks, record Worldlab captures, and inspect the result. Autonomous motor movement is not enabled by default.

```bash
cargo run --bin pete -- hardware-env
cargo run --bin pete -- robot --mode read-only --duration-seconds 30 --ledger data/ledger/real/read-only-smoke
cargo run --bin pete -- capture-real --duration-seconds 60 --out data/captures/real/rpi5-smoke
cargo run --bin pete -- inspect-capture data/captures/real/rpi5-smoke
```

After read-only validation, the guarded production possession command is
explicit and wheels-off-floor first:

```bash
just possess
```

Set `PETE_BRAINSTEM_DEVICE_ID` in `.env` first; optionally pin
`PETE_COCKPIT_PORT=/dev/serial/by-id/DEVICE`. The recipe learns the current
boot ID from the pinned device, saves it as `PETE_BRAINSTEM_BOOT_ID`, and
retries automatically after a brainstem reboot. After a cold boot, a rejected
Wi-Fi identity is automatically established over the pinned USB brainstem
before retrying. It expands to the guarded command below:

```bash
cargo run -p pete-tools -- robot --mode possession-slow \
  --cockpit uart --create-port /dev/serial/by-id/DEVICE \
  --brainstem-device-id BRAINSTEM_ID \
  --brainstem-boot-id BOOT_ID \
  --max-linear-mm-s 50 --max-angular-mrad-s 500 \
  --duration-seconds 30 \
  --ledger data/ledger/real/possession-wheels-off-floor \
  --capture data/captures/real/possession-wheels-off-floor
```

The motherbrain control lease is possession; there is no separate arm layer.
The runner begins with STOP, never exposes Create OI, and does not fall back to
another device or transport. Orderly shutdown requires acknowledged STOP then
exorcize (the brainstem DISARM wire operation) and final stopped/unpossessed
status. Serial loss is handled by the short
command, heartbeat, and lease deadlines; no power toggle is attempted.

See [docs/rpi5-bringup.md](docs/rpi5-bringup.md) for packages, permissions, device expectations, success criteria, and failure behavior.

## Docker services

Copy `.env.example` to `.env` if you want to override ports or passwords. Then:

```bash
just servers      # Neo4j + Qdrant
just live-server  # Neo4j + Qdrant + live sim server
```

The live view is at `http://localhost:8787/view`, with a 3D/WebXR sensorium at `http://localhost:8787/view/3d` and compact scene JSON at `http://localhost:8787/view/scene`. In immersive VR, supported controllers can feed short-lived Reign commands into the virtual session. Neo4j Browser is at `http://localhost:7474` with the default `.env.example` credentials.

For a one-command HTTPS Dream World training run, run:

```bash
just go virtual
```

It starts a long-running Dream World, serves HTTPS on `0.0.0.0`, and prints desktop/headset URLs for `/view/3d` plus `/view/scene`. See [docs/go-virtual.md](docs/go-virtual.md) for certificate, headset, scenario, and LAN security notes. See [docs/webxr-viewer.md](docs/webxr-viewer.md) for desktop controls, WebXR caveats, capture replay hooks, and sensor-view privacy notes.
