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
- The live 2D panel now projects obstacle cells directly from the same
  calibrated odometry-world voxels shown in 3D and reports alignment, geometry,
  and navigation trust separately. A synchronized physical golden capture is
  still needed to clear the real-sensor geometry and corrected-SLAM gates.

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
  commands, and immediate reflex authority. The motherbrain possessor owns
  reusable skills and higher-level recovery sequencing; every such command is
  still subordinate to brainstem safety and reflex preemption.
- The Pi 5 motherbrain remains canonical for Pete's live experience, graph,
  vector stores, inference state, and bounded online learning.
- Deterministic motherbrain policy is runtime-loaded Lua. Its embedded
  scheduler keeps one foreground intention, transparently suspends ordinary
  functions at bodily operations, and lets disjoint organs overlap through
  [`together`](docs/028-lua-motherbrain-skills.md).
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
`pete` is a Rust workspace for Pete, an embodied self-training robot architecture.

Pete now has working real-world perception and guarded physical possession. Kinect RGB and depth data can be fused into aligned, Minecrafty but properly colored 3D voxels, while `just possess` establishes the motherbrain lease, streams body telemetry, and drives through bounded safety gates. The active milestone is behavior validation: proving that normal runs compose perception, action selection, autonomic vetoes, brainstem reflexes, recovery, and learning traces consistently.

The project still keeps the larger PETE architecture in view. Pete gathers raw sensors, memory recalls, internal drives, model predictions, surprise, and LLM-derived evidence into `Now`. `Now` is compressed into an `ExperienceLatent`, used to imagine futures, choose actions, and train from consequences.

Pete acts through high-level action primitives. A hard-coded autonomic layer keeps the body safe. The LLM may reflect, critique, teach, and produce counterfactuals, but its output returns only as evidence. It cannot select goals, become a Cockpit proposal, acquire motion authority, or issue motor commands.

The LLM loop is active as a trainer, critic, and advisory planner. It predicts counterfactual outcomes, critiques training data, and records possible tests or motions as typed evidence. Local goals, skills, Reign, and safety own action selection and physical motion through possession. The remaining work is to validate behavior and safety outcomes under real contact, charging, cliff, wheel-drop, heartbeat-loss, and transport-loss conditions.

Every hard-coded behavior is replaceable. It can run directly, shadow-train a model, compare with a model, promote a model, or fall back to safe hand-written logic.

The first small evolved nervous system is the replaceable `locomotion` behavior.
Run `just train --neat locomotion` to evolve it through a staged, WorldLab-visible
curriculum and promote it when the transfer audit beats the active baseline. See
[docs/neat-locomotion.md](docs/neat-locomotion.md).

Pete is an embodied predictive organism: a robot body with reflexes, an experience ledger, compact learned present, imagined futures, memory returning as sensation, swappable learned behaviors, LLM cognition that contributes evidence and teaching, and safety that protects the body.

Pete's higher-brain split now has a reproducible forebrain, a strictly separate
bulk-data plane, immutable experience bundles, durable training jobs, and
atomic candidate activation/rollback. See
[docs/018-higher-brain-framework.md](docs/018-higher-brain-framework.md) for the
architecture, provisioning, enrollment, and simulated end-to-end workflow.

## Current milestone: behavior validation

Possession and the live perception stack are operational. The current milestone is to validate complete behavior episodes and preserve their evidence:

- Kinect depth and RGB are being aligned into colored voxels.
- The voxel output is coarse, blocky, and intentionally debuggable.
- Visible structures in the voxel scene correspond to real things in the room.
- The 3D/WebXR viewer is the main inspection surface for the current world model.
- The 2D map is restored as a projection of the calibrated 3D voxel world;
  physical capture comparison and corrected-SLAM trust remain explicit gates.
- `just possess` acquires and maintains the motherbrain lease with bounded motion and explicit STOP/exorcize shutdown.
- Normal-run tests cover random walk, bump stop, and conductor recovery.
- Create packet age and completeness now drive body freshness; reconnect cannot reopen motion on cached telemetry.
- The next physical work is the safety checklist below, not restoration of the command-to-base path.

The next good artifact is a capture set: screenshots, video capture, and a short golden run containing RGB, depth, IMU, odometry, command traces, safety decisions, and map/voxel outputs from the same session.

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
5. Validate the restored 2D shared-world projection against a synchronized
   physical golden capture.
6. Exercise possession, heartbeat expiry, event loss, reconnection, stop, and
   exorcize across repeated cold boots.
7. Move remaining Create controller tuning constants into the body contract.
8. Generalize motion-centric runtime terms into body actions and implement a
   deliberately non-mobile second body or safe low-voltage fixture.
9. Exercise higher-brain transfer interruption, candidate return, activation,
   and rollback across the actual Pi and forebrain links.
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
- LLM counterfactuals, critiques, and advisory action hypotheses preserved as evidence.

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

### 2D map validation

The live 2D panel reports the canonical runtime map used by behavior, while its
3D-world overlay is derived from the calibrated odometry-world voxel cloud used
by the 3D view. Its status line distinguishes
three claims: whether 2D and 3D share a frame, whether current sensor geometry
has enough stable evidence to trust, and whether corrected SLAM is ready for
navigation. Pose-graph loop corrections rebuild 2D occupancy from retained
submaps. They do not yet rebuild accumulated 3D voxels, so that overlay becomes
explicitly untrusted after a nontrivial graph correction. The remaining closure
gates are a synchronized multi-frame stationary-rotation capture and a physical
return-to-start route that passes the report's graph-error, wall-overlap, and
corrected-endpoint checks.

### Behavior validation

Movement is deliberately conservative and now operational under possession. Validate each behavior from command intent outward:

1. Did a movement intent get generated?
2. Did the controller receive it?
3. Did the safety layer veto it?
4. Is the base in the correct mode to accept motion?
5. Did the robot report a fault, dock state, cliff signal, bumper signal, or stale serial/body connection?

Do not bypass safety to make a scenario pass. Preserve the chosen action, autonomic decision, final hardware gate, brainstem event sequence, and observed body outcome so discrepancies become regression tests.

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

This first-stage command deliberately runs body-only: Kinect, V4L camera,
microphone, GPS, local IMU, and the LLM provider are disabled. After that path
is healthy, run the same guarded possession envelope with the configured higher
senses restored:

```bash
just possess-sensorium
```

The sensorium profile uses the normal robot sensor configuration and discovery
path. Configure `CAMERA_DEVICE`, `MIC_DEVICE`, `IMU_DEVICE`, and
`GPS_SERIAL_PORT` in `.env`; `PETE_KINECT_DEPTH=0` selects the V4L camera path
instead of the default Kinect RGB/depth path. Ollama is enabled by default, or
pass `--llm-config PATH` / `--llm-provider disabled` explicitly. Both profiles
retain the same identity pinning, possession lease, motion limits, safety and
autonomic vetoes, reconnect behavior, STOP, and exorcize shutdown.

Set `PETE_BRAINSTEM_DEVICE_ID` in `.env` first; optionally pin
`PETE_COCKPIT_PORT=/dev/serial/by-id/DEVICE`. The recipe learns the current boot
ID from the pinned device, saves it as `PETE_BRAINSTEM_BOOT_ID`, and retries
after a brainstem reboot. After a cold boot, a rejected Wi-Fi identity can be
established over the pinned USB brainstem before retrying.

The guarded command is equivalent to:

```bash
cargo run -p pete-tools -- robot --mode regular \
  --cockpit uart --create-port /dev/serial/by-id/DEVICE \
  --brainstem-device-id BRAINSTEM_ID \
  --brainstem-boot-id BOOT_ID \
  --max-linear-mm-s 50 --max-angular-mrad-s 500 \
  --tick-ms 20 \
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
The guarded physical bumper-recovery smoke test uses the same possession path:

```bash
just possess --recovery-smoke --wheels-off-floor
```

It requires a physical brainstem and explicit wheels-off-floor acknowledgement,
then waits for an operator-held bumper and verifies contact → stop → clear →
reverse → turn → probe → inspect before STOP and exorcize. After the bumper is
released, the smoke clears its bump latch and any e-stop it observed during that
same incident; a pre-existing e-stop remains latched for the operator. This command is
documented but has not been run as part of this software-only readiness pass.

Run the pending physical validation as a guided, evidence-recording session:

```bash
just physical-qa
```

The interactive runner captures the brainstem firmware identity, lets the
operator select cases, explains each safe setup and acceptance gate, launches
the guarded bumper helper, and records pass/fail/blocked evidence in
`data/reports/physical-qa/`. Preview every case without hardware with
`just physical-qa --plan`.

See [docs/rpi5-bringup.md](docs/rpi5-bringup.md) for packages, permissions, device expectations, success criteria, and failure behavior.

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
