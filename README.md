# pete

`pete` is a Rust workspace for Pete, an embodied, self-training machine with a
robot body, local reflexes, inspectable perception, durable experience, learned
behaviors, and higher-brain training infrastructure.

Pete is no longer only a simulation architecture or a collection of sensor
experiments. The repository now contains a working vertical slice from physical
body control through perception and experience exchange:

- a Pico W brainstem owns the iRobot Create Open Interface, immediate safety,
  bounded motion, possession, body supervision, and ordered physical events;
- a common Cockpit contract exposes the same body capabilities over simulator,
  UART, UDP, HTTP, and WebSocket transports;
- the motherbrain can establish the physical brainstem identity, acquire a
  guarded control lease, stream sensors, issue short-lived motion, and stop on
  safety events or missing history;
- Kinect RGB/depth and optional HLS-LFCD2 lidar can contribute to a shared,
  inspectable 3D voxel world;
- an experience ledger, replay/evaluation paths, model registry, and promotion
  gates support learning without replacing safe hand-written behavior blindly;
- the higher-brain framework can provision forebrains, export immutable
  experience bundles, run durable jobs, return model candidates, and activate
  or roll them back atomically;
- the X1202 UPS integration has telemetry, charge-control, service, and
  motherbrain-reset plumbing, with final electrical and dock-readiness
  validation dependent on the physical board.

The project still follows one central rule: intelligence may request physical
action, but it may not bypass the body-local controller that owns the
consequences.

## Where we are now

### Body control

The Create control path is implemented rather than hypothetical.

- The brainstem acquires and supervises Create OI, maintains the required body
  mode, decodes sensor frames, and owns stop behavior.
- Possession is an explicit, identity-bound control lease. There is no separate
  higher-level arm state that can evade the brainstem's authority.
- Motion commands are bounded by TTL, heartbeat, lease, body state, and safety
  policy.
- Stop and emergency stop preempt ordinary work.
- Bump, cliff, wheel-drop, tilt, impact, UART, stale-command, and missed-event
  conditions are represented in the safety and event paths.
- The body remains supervised when motherbrain possession is released.
- A dedicated forebrain UART lane and the Pico W network services enter through
  bounded command handling rather than direct hardware access.

Real-hardware work is now validation and refinement: repeatable cold starts,
longer possession runs, sensor calibration, dock-state interpretation, UPS
integration, and capture of golden sessions. It is no longer a search for a
basic command-to-wheel route.

See [crates/pete-brainstem/README.md](crates/pete-brainstem/README.md) for wiring,
firmware, transport, protocol, and bring-up details.

### The brainstem is broader than Pete's first body

The current driver targets the iRobot Create, but the architectural role is a
deterministic controller for embodied machinery. A brainstem owns local
hardware, immediate sensing, actuator authority, bounded execution, safety
invariants, and an ordered account of what physically occurred.

That same model can describe a pinball playfield, a toaster oven, an electric
kettle, an industrial fixture, or another robot. The higher process chooses
goals; the brainstem retains final authority over relays, coils, heaters,
motors, and safe transitions.

The normative body-model document defines the layers, command lifecycle,
body-defined safety semantics, worked examples, current Create-shaped seams,
and the path toward a real second body:

[docs/019-brainstem-body-model.md](docs/019-brainstem-body-model.md)

### Perception and world model

The strongest current perception artifact is an inspectable 3D voxel world.

- Kinect depth and RGB can be aligned into coarse colored voxels that correspond
  to real structures in the room.
- The 3D and WebXR views are the primary inspection surfaces.
- HLS-LFCD2 lidar support can feed planar scans into the 2D map and the same
  Kinect/lidar 3D cloud using configured sensor extrinsics.
- The intentionally blocky representation favors spatial trust and debugging
  over visual polish.
- The 2D occupancy path still needs systematic comparison against the shared
  coordinate frame and known-good captures.

The next important evidence is a golden run that preserves synchronized RGB,
depth, lidar, IMU, odometry, body events, command traces, safety decisions, and
rendered map/voxel outputs from the same session.

### Learning and experience

Pete gathers raw sensors, body state, memory recalls, internal drives, model
predictions, surprise, and higher-level guidance into `Now`. `Now` can be
compressed into an `ExperienceLatent`, used to imagine futures, choose actions,
and train from consequences.

Every hand-written behavior is intended to be replaceable through an explicit
process: run directly, shadow a model, compare outcomes, promote a candidate,
and retain a safe fallback. Learned artifacts do not receive direct brainstem
or actuator authority.

A **constellation** is a repeatable pattern of experience rather than merely a
visual cluster. The search begins with arrangements that recur across time,
viewpoint, modality, action, and consequence, then promotes stable patterns
into objects, places, affordances, or training specimens.

See:

- [docs/013-feature-registry.md](docs/013-feature-registry.md) for the universal
  observation layer;
- [docs/constellations.md](docs/constellations.md) for generalized pattern
  discovery;
- [docs/scenario-evaluation.md](docs/scenario-evaluation.md) for repeatable
  evaluation;
- [docs/model-registry.md](docs/model-registry.md) for checkpoint registration
  and promotion gates.

### Higher-brain split

Pete now has deliberately separate bodily control and bulk-data planes.

- The Pico W brainstem exclusively owns bodily safety, possession, bounded
  commands, reflexes, and recovery authority.
- The Pi 5 motherbrain remains canonical for Pete's live experience, graph,
  vector stores, inference state, and bounded online learning.
- Forebrains are enrolled compute nodes that receive immutable exports and run
  authorized jobs without acquiring motion authority.
- Experience bundles are checksummed, resumable, and immutable.
- Jobs are durable and restart-aware.
- Returned candidates are treated as untrusted until validated, staged, and
  explicitly activated.
- Activation and rollback use atomic model pointers.

See [docs/018-higher-brain-framework.md](docs/018-higher-brain-framework.md) for
the protocol, provisioning, authorization, transfer, jobs, candidate lifecycle,
and simulated end-to-end exercise.

## Architecture sketch

```text
physical body and sensors
    -> brainstem safety/runtime lane
       -> body state, reflexes, bounded actions, ordered events
    -> synchronized RGB/depth/lidar/IMU/body observations
       -> Features
       -> point cloud, voxel world, and 2D occupancy
       -> cross-modal constellations
       -> object, place, affordance, and action hypotheses
       -> Now / ExperienceLatent
       -> prediction, memory, action, evaluation, and training
       -> immutable experience bundles
       -> forebrain jobs and candidate models
       -> validation, staging, activation, and rollback
```

The architecture separates three questions that are easy to blur together:

1. **What is physically happening?** The brainstem and sensors provide the
   authoritative local transcript.
2. **What does it mean, and what should happen next?** The motherbrain performs
   perception, memory, planning, and live inference.
3. **What can be learned from accumulated experience?** Forebrains perform
   heavier replay, analysis, training, and evaluation on immutable exports.

## Near-term work

The next milestones are integration milestones rather than another wholesale
architectural restart:

1. Validate the X1202 UPS telemetry, charging control, external-power signals,
   and Pi RUN reset path on the physical board.
2. Derive trustworthy dock and charging readiness from the real Create and UPS
   status paths.
3. Record a synchronized golden physical run and preserve it as a regression
   fixture.
4. Calibrate Kinect, lidar, IMU, and odometry frames against the same physical
   scene.
5. Restore and validate the 2D map as a projection of shared spatial truth.
6. Exercise possession, heartbeat expiry, event loss, reconnection, stop, and
   exorcize across repeated cold boots.
7. Move remaining Create controller tuning constants into the body contract.
8. Generalize motion-centric runtime terms into body actions and implement a
   deliberately non-mobile second body or safe low-voltage fixture.
9. Exercise higher-brain transfer interruption, candidate return, activation,
   and rollback across the actual Pi and forebrain links.

## Setup

On Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install -y just
just setup
```

`just setup` installs the Linux build dependencies plus Kinect 1 userspace
support through `libfreenect` when distro packages are available.

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

Scenario reports can be generated with:

```bash
just run eval-scenario --scenario empty-room --episodes 2 --steps 10 \
  --out data/reports/scenario/empty-smoke.json
```

## Hardware bring-up

Pete's Raspberry Pi 5 hardware path starts capture-first: inspect devices, run
bounded read-only body and sensor ticks, record Worldlab captures, and inspect
the result. Autonomous motor movement is not enabled by default.

```bash
cargo run --bin pete -- hardware-env
cargo run --bin pete -- robot --mode read-only --duration-seconds 30 \
  --ledger data/ledger/real/read-only-smoke
cargo run --bin pete -- capture-real --duration-seconds 60 \
  --out data/captures/real/rpi5-smoke
cargo run --bin pete -- inspect-capture data/captures/real/rpi5-smoke
```

After read-only validation, guarded production possession is explicit and
wheels-off-floor first:

```bash
just possess
```

Set `PETE_BRAINSTEM_DEVICE_ID` in `.env` first; optionally pin
`PETE_COCKPIT_PORT=/dev/serial/by-id/DEVICE`. The recipe learns the current boot
ID from the pinned device, saves it as `PETE_BRAINSTEM_BOOT_ID`, and retries
after a brainstem reboot. After a cold boot, a rejected Wi-Fi identity can be
established over the pinned USB brainstem before retrying.

The guarded command is equivalent to:

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

An HLS-LFCD2 or LDS-01 can run alongside either path. Set its serial path and
mount pose in `.env`; `just robot` and `just possess` then feed its 360-degree
scans into Pete's planar map and shared Kinect/lidar 3D voxel cloud. A pitched
scan can accumulate into 3D as odometry changes during forward motion or a
spin.

The runner begins with STOP, never exposes Create OI, and does not fall back to
another device or transport. Orderly shutdown requires acknowledged STOP,
exorcize, and final stopped/unpossessed status. Exorcize releases motherbrain
control without abandoning brainstem supervision of the Create. Serial loss is
handled by short command, heartbeat, and lease deadlines; no blind power toggle
is attempted.

See [docs/rpi5-bringup.md](docs/rpi5-bringup.md) for packages, permissions,
device expectations, success criteria, and failure behavior.

## Docker services

Copy `.env.example` to `.env` to override ports or passwords, then run:

```bash
just servers      # Neo4j + Qdrant
just live-server  # Neo4j + Qdrant + live sim server
```

The live view is at `http://localhost:8787/view`, with a 3D/WebXR sensorium at
`http://localhost:8787/view/3d` and compact scene JSON at
`http://localhost:8787/view/scene`. In immersive VR, supported controllers can
feed short-lived Reign commands into the virtual session. Neo4j Browser is at
`http://localhost:7474` with the default `.env.example` credentials.

For a one-command HTTPS Dream World training run:

```bash
just go virtual
```

It starts a long-running Dream World, serves HTTPS on `0.0.0.0`, and prints
desktop/headset URLs for `/view/3d` plus `/view/scene`. See
[docs/go-virtual.md](docs/go-virtual.md) for certificate, headset, scenario, and
LAN security notes. See [docs/webxr-viewer.md](docs/webxr-viewer.md) for desktop
controls, WebXR caveats, capture replay hooks, and sensor-view privacy notes.