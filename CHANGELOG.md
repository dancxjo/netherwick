# Changelog

All notable changes are grouped by date.

## Unreleased

### Added

- Ready: Author canonical loss-intolerant calibration transitions at the Kinect,
  IMU, locomotion, and per-stream latency estimator boundaries, preserving
  prior/new state, epochs, per-degree observability and uncertainty, evidence
  and artifact provenance, affected consumers, reasons, and clock domains;
  replay now reconstructs identical epoch/trust transitions and the server no
  longer infers calibration events from snapshots. Lidar remains optional.

- Ready: Add a new versioned `pete-events::BrainEvent` observability contract with
  typed event identity, causality, quality, trust, authority, loss policy, and
  payload/reference boundaries; include migration support from legacy v0 input and a
  v1 schema validator.
- Ready: Add conversion/adaptor pathways from existing runtime records (`Sensation`,
  `Impression`, `Experience`, legacy `Event`, Reign outcomes, and now snapshots) into canonical `BrainEvent` envelopes for replay-capture-explain interoperability.
- Ready: Add strict deduplication and schema regression coverage for BrainEvent
  envelopes, including a v0 fixture migration test and canonical replay compatibility
  checks.
- Ready: Add the `docs/brain-event-contract.md` contract document covering identity,
  causality, time/epoch semantics, loss policy, payload boundaries, and schema
  evolution.
- Add bounded offline object detection for Kinect/camera RGB with a backend-
  neutral interface, latest-first Pi 5 worker, calibration-aware depth
  association, short-term tracking, provenance-bearing descendant crops,
  capture/live diagnostics, and deterministic fixture/capture evaluation.

- Add exact-input learned-locomotion shadow records with baseline/candidate
  provenance, inference timing, explicit candidate failures, downstream safety
  evidence, and fail-closed promotion gates requiring valid held-out simulation
  plus physical metrics, hardcoded fallback, atomic activation, and rollback.

- Add a truthful pogo-only X1202/Create power evidence model with per-signal
  ages/confidence, explicit absent-current semantics, trend-qualified charging
  inference, independent consolidation gates, mid-cycle pause on power loss,
  and dashboard/WorldLab capture exposure through the real runtime path.

- Extend replay validation with a full-turn stationary-rotation evaluator
  (unwrapped heading, IMU agreement, axle translation, calibration remount and
  covariance gates, optional lidar) and return-to-start reports with physical
  reference sidecars, raw/corrected endpoint and heading metrics, direction,
  calibration epochs, Kinect-only status, and an explicit navigation-trust
  decision backed by deterministic synthetic fixtures.

- Add advisory bounded locomotion calibration with conservative nominal
  fallback, global/left/right/forward/reverse distance estimates, independent
  CW/CCW rotation scale and effective wheelbase, condition epochs, uncertainty,
  rejection and held-out validation, retained replay episodes, and capture
  provenance without changing motor or brainstem safety authority.

- Add per-stream end-to-end latency calibration with raw timestamp/clock/event
  provenance, median/p95/jitter/uncertainty distributions, correlated rotation
  evidence, drift and clock epochs, held-out validation, replayable capture
  state, optional-lidar reporting, and timing-aware fusion trust degradation.

- Add adaptive local-IMU calibration with odometry-confirmed stationary windows,
  gyro bias/noise and temperature tracking, gravity-derived mounting, explicit
  yaw observability and rotation-scale evidence, remount epochs, trust-gated
  arbitration/fusion, capture provenance, and deterministic motion/remount tests.

- Add the shared adaptive Kinect mount-calibration foundation: configured,
  estimating, trusted, degraded, and invalidated states; six-degree transform
  covariance and observability; evidence-source interfaces; residual-based
  shift detection and epochs; floor-plane estimation; capture/replay
  diagnostics; live geometry trust gating; and deterministic remount tests,
  with lidar remaining optional and brainstem safety unchanged.

- Add a useful forebrain `dataset_construction` lifecycle for checksummed
  `RealRobot` WorldLab captures, including complete physical-capture export,
  resumable-transfer and worker-restart recovery controls, deterministic
  frame/modality datasets and quality metrics, schema-gated candidate staging,
  explicit activation, incompatible-candidate rejection, and rollback coverage
  without granting learned artifacts brainstem authority.

- Add `just possess-sensorium` and `just possess-sensorium-rpi5` as explicit
  second-stage physical checks that retain possession safeguards while restoring
  configured higher sensors and Ollama cognition; keep existing possession
  commands deliberately body-only.

- Add the complete brainstem inertial handoff: parity timing/trust fields in
  compact and JSON status, stationary firmware gyro-bias estimation, bounded
  host clock mapping with reboot/reconnect epochs, truthful roll/pitch-only MPU
  conversion, automatic trust-gated per-source arbitration, isolated fusion
  histories, Kinect exposure alignment, diagnostics, and bring-up guidance,
  while preserving independent brainstem safety and raw Create odometry.

- Add Brainstem-owned Solresol auditory annunciation with typed hazard,
  control, power, health, recovery, docking, and service cues; bounded
  urgent/informational scheduling; transition deduplication; playback
  observability; transport-compatible `pete-cockpit audio silent|audible`
  control; and a status-synchronized silent-mode toggle in the embedded browser
  pilot panel, neither of which changes safety behavior.
- Add a deterministic seven-case social acceptance exam spanning simulator
  perception, canonical social/temporal/epistemic state, sleep interruption,
  asynchronous cognition delay, and forebrain failure, with a `just
  social-exam` operator command and optional JSON report.
- Add `independent_watchdog` brainstem capability negotiation and runtime safety
  metadata, then enforce reduced-motion gating in physical possession unless
  `--wheels-off-floor` or explicit `--acknowledge-no-independent-watchdog` is
  provided for direct-RPi floor operation.
- Add explicit Ollama resource-bounds propagation (`num_ctx`, `num_predict`,
  `num_thread`) from `LlmConfig` into Ollama generate/chat requests and capture it
  in the possession runtime test coverage.
- Add capture manifest/asset health reporting and raw stream diagnostics:
  background capture-writer queueing with drop accounting, per-stream metadata with
  capture/producer timing, write checksums, and CLI inspection output for counts,
  unavailable/late/partial/dropped/failed statuses.
- Add capture stream expansion to include optional `camera`, `lidar`, `imu`,
  `transcript`, and `calibration` assets plus `camera`/`calibration` provenance,
  while keeping `capture schema v2` compatibility through manifest and frame metadata.
- Add capture startup diagnostics for physical possession, including optional stream
  readiness gating via `--require-camera`, `--require-kinect`, and
  `--require-llm` plus `--sensor-readiness-timeout-ms`/`PETE_SENSOR_READINESS_TIMEOUT_MS`
  to fail fast when required streams are unavailable.

### Fixed

- Ready: Ensure real-robot possession always attempts STOP/exorcize and capture finalization through a unified exit path even when the control loop exits with an error, and report combined shutdown/capture/control failures instead of silently dropping finalization work.
- Ready: Extend physical possession capture coverage with a regression test that uses runtime frame timing, writes raw RGB/depth/audio assets, and validates captured asset timestamps, IDs, and file outputs in snapshot metadata.
- Ready: Preserve RGB/depth/audio capture asset export in real-robot snapshot recording while enriching exported asset metadata with capture timing, frame identifiers, device timestamps, and source/coordinate provenance.
- Connect the UDP cockpit socket to its configured brainstem peer so valid-looking
  datagrams from unrelated local senders cannot be accepted as brainstem replies.
- Ready: Remove stale `export_snapshot_assets` from `pete-tools` capture import usage while preserving behavior.
- Ready: Make possession reconnect cancellation-aware during both connection
  attempts and exponential backoff, keep the dashboard in an explicit
  stopped/reconnecting state, and allow SIGINT/SIGTERM to close capture and
  ledger output when the brainstem remains unavailable.
- Ready: Update real-robot runtime documentation for reconnection shutdown
  behavior so transport-loss possession exits explain the fail-closed stop path
  and control-state signaling.
- Require a fresh complete Create OI packet reporting an active electrical
  charging state before Lua docking skills claim charging, keeping Home Base
  contact, dock IR, charge-indicator GPIO state, and OI waiting/fault states
  from satisfying the charging postcondition.
- Keep charging completion in docking dependent on VerifyCharging freshness and
  charging-state code, and return source/state metadata so consumers can
  distinguish create-derived evidence.
- Ready: Keep live possession snapshots compact by stripping large sensor payload
  blobs from JSON records, exporting raw assets with capture metadata, and
  rehydrating stripped fields when opening capture records.
- Ready: Harden capture durability checks by validating dropped-frame telemetry from
  bounded background writer queues and ensuring dropped frames/assets are tracked
  and surfaced in manifest writer health.
- Timestamp physical capture frames with the canonical fused runtime-frame time
  while retaining body and sensor producer timestamps as provenance, preventing
  stale Create packets from collapsing asynchronous sensor evidence onto duplicate
  capture times.
- Integrate legacy cumulative physical odometry as stateful SE(2) deltas,
  require measured scan/submap registration for live loop constraints, publish
  the runtime map as the dashboard's canonical map, rebuild retained 3D depth
  observations through corrected graph poses, and gate physical SLAM on fresh synchronized
  multi-frame Kinect/IMU evidence plus return-to-start quality metrics.
- Pair Kinect RGB and depth atomically on the mapped device clock, interpolate
  body pose and filtered calibrated IMU orientation to depth exposure time,
  reproject color through measured RGB-D intrinsics/extrinsics, add full camera
  translation and LFCD2 beam motion de-skew, and reject future, stale,
  unsynchronized, or nominal-only physical calibration evidence.
- Integrate Brainstem Create distance/angle deltas into coherent planar pose,
  carry that pose through JSON and compact status into runtime body sensing, and
  enable the Pico W hardware watchdog from the runtime safety lane with a
  body-configured two-second timeout.
- Restore the live 2D map as an explicit projection of the calibrated 3D
  odometry-world voxel cloud, expose alignment, geometry, and navigation trust
  separately, and cover depth-only projection through a rotated world-frame
  regression fixture.
- Make audio silent-mode changes session-authorized so cockpit preference
  toggles cannot replace a motherbrain control lease or interrupt active motion.
- Honor a cognition provider's declared availability before scheduling work, so
  a disabled forebrain is immediately represented as unavailable while local
  control continues without generating background requests.
- Preserve provider-suggested cognition decisions as provenance-bearing,
  non-executable advisory telemetry through runtime and scene snapshots when
  they are discarded at the authority boundary.
- Keep `rich_language` available while a healthy cognition request is in flight,
  and report request occupancy separately as `busy` instead of treating every
  pending request as a service outage; enforce a runtime-owned cooldown after
  every cognition outcome so fast providers cannot continuously generate work.
- Separate higher-cognition outcomes from executable local actions by adding an
  `advisory_action` payload to cognition proposals, keeping provider-suggested
  suggestions as non-authoritative telemetry while preserving
  `proposed_action` as the only executable path.
- Read the r23 Create charging indicator from GP20/physical pin 26 in both Pico
  backends, and move the optional external status output to GP17 so firmware
  never drives the TXS channel 8 charging signal.
- Make r23 Create sleep/wake requests state-aware at the pulse site: repeated
  sleeps cannot toggle a known-OFF robot back on, known-ON wake timeouts remain
  probe-only, UNKNOWN wake permits at most one best-effort pulse, and failed
  probes no longer queue an automatic power cycle.
- Remove the unused Create baud-control configuration, commands, runtime
  actions, capability field, operator control, compatibility parser, and dead
  driver path; r23 power control now has only the state-aware GP18 toggle.
- Restrict brainstem contact withdrawal to a fresh bumper edge during unsafe
  forward output; held-at-boot and stationary contact now latch and stop
  without starting authority-independent reverse motion.
- Keep slow possession running when its motion preflight observes an existing
  safety latch by returning the typed latch reason to the recovery path.
- Run `just possess` on a 20 ms target period and count sensing/runtime work
  inside that period, eliminating the extra post-tick delay that slowed
  bounded velocity refreshes and could create stop gaps.
- Make brainstem contact withdrawal react to an asserted bumper state without
  requiring a new edge or forward wheel output.
- Add calibrated 3D world point-cloud projection metadata to the live map API:
  emit a new `world_projection` payload with source attribution, coordinate
  frame, trust gates, reasons, and per-cell stability/confidence/age data.
- Add a live-map trust banner on the 2D trace view that displays projection
  alignment, geometry trust, navigation gating, and projected-cell counts
  while rendering projected cells with separate stable/unstable styling.
- Extend live-map endpoint and browser contract tests to validate projected-cell
  trust metadata, depth-based projection behavior, and the new `map-trust`
  rendering path in the 2D map page.
- Derive `Clone`, `Debug`, `PartialEq`, `Serialize`, and `Deserialize` for
  `EmbodiedDemo` so demo embeddings can be copied, compared, and serialized
  consistently through experience tooling.
- Ready: Normalize Brainstem test module formatting in
  `crates/pete-brainstem/src/{arch/pico_w_tests.rs,display_tests.rs,runtime_tests.rs,status_tests.rs}`
  by removing stray leading indentation and restoring canonical top-level item
  alignment.
- Propagate optional live control intent (`control_state`, `control_detail`) through
  scene session contracts and UI dashboards so live mode and 3D views display the
  current control posture alongside mode, scenario, and training labels.
- Ready: Persist IMU stream sample diagnostics (`sample`, age, candidate metadata)
  in telemetry and improve possession recovery safety defaults so local direct-RPi
  sessions default to reduced motion surface unless explicitly opted out.

### Auto-sync (2026-07-15)

- Refresh the lockfile against the current workspace and external Tongues
  dependency graph, restoring the speech-training packages and plotting
  dependencies required by the current path manifests.
- Record the `pete-cockpit` CLI's `clap` dependency in the workspace lockfile.
- Record the current `pete-memory`, ORT, `burn-core`, `rgb`, and `xtask`
  dependency metadata.
- Deduplicate and normalize `Cargo.lock` dependency metadata by removing stale
  `colored`, `console`, `indicatif`, `drawille`, `number_prefix`, and
  `textplots` package entries, then relaxing crate dependency pins to shared
  workspace versions where applicable.

### Added

- Add a runtime-owned asynchronous cognition supervisor that submits bounded
  immutable-snapshot requests in the background, polls without delaying control,
  and rejects cancelled, expired, or obsolete responses; route live scene
  enrichment through the shared boundary.
- Add an optional Pico W SSD1306 status OLED on the shared MPU I2C bus, with
  double-height normal state/control-authority words, full-screen fault alerts,
  stale-safe self-sustaining battery telemetry, persistent charging state, a
  secondary SSID/IP/AP/DHCP-lease page, a liveness pixel, explicit Create
  bring-up and runtime errors, complete safety-latch alert coverage, exact
  framebuffer snapshots, bounded display writes, and failure isolation from
  boot, control, safety, and IMU sampling.
- Add an RPi 5 Brainstem backend that owns the Create 1 side-port serial cable,
  reuses the bounded Brainstem runtime and Cockpit session/lease contract over
  loopback, advertises the cable's reduced hardware capabilities, sends a final
  STOP on service shutdown, and supports local possession plus systemd operation.
- Add encounter-scoped social acknowledgments and a persistent
  `greet_person` goal that proposes `motherbrain.greet` for a recognized,
  newly present person; greeting progress and completion retain the person,
  encounter, skill execution, source hash, and provenance.
- Add an embedded, sandboxed Lua 5.4 motherbrain skill runtime with atomic
  runtime loading and reload, one foreground call tree, transparent coroutine
  suspension at bodily operations, deterministic implicit organ ownership,
  fail-fast concurrent `together`, typed preemption/outcomes, bounded execution,
  canonical `Now` queries, goal progress, and experience provenance.
- Add generation-bound `escape_motion` segments for automatic bump/cliff
  recovery, with hazard-specific envelopes, 250 ms TTLs, absolute-hazard
  preemption, observed-motion reporting, and no freestanding safety override;
  reserve broad `careful_mode` for attended operator-debug authority.
- Add a transport-neutral `just cockpit` operator CLI with auto-discovered USB
  CDC, explicit UART and HTTP backends, structured status/capability/event
  output, bounded guarded drive, and IR-guided dock alignment.
- Carry the Create's omnidirectional IR character from brainstem sensor packets
  into compact cockpit status, motherbrain body state, queryable features,
  experience vectors, and LLM-visible senses while preserving older serialized
  body/status data.
- Add bounded Pico W AP-local ICMPv4 echo replies with IPv4/ICMP validation,
  four-per-second rate limiting, status diagnostics, parser coverage, and a
  ping bring-up check; ICMP remains isolated from routing and robot hardware.
- Embed reproducible Brainstem firmware build identity from Git or explicit CI
  overrides, expose it across status/capability/operator surfaces, and record
  it in physical capture manifests.
- Add a host-aware control-path failover state machine and deterministic
  failure-injection matrix that preserve motherbrain control across forebrain
  link loss, change motherbrain transport without changing its role, require a
  fresh authoritative no-controller observation plus atomic acquisition for
  takeover, and coordinate safe handback without overlapping controllers.
- Define brainstem host transit over the existing deterministic AP and
  session-bound `motherbrain.pete.internal` registration, with an explicit
  recovery-service allowlist; association and reachability grant no motor
  authority and the path excludes bulk data and direct motion commands.
- Add replayable per-goal progress reports that preserve behavior/skill
  expectations with explicit metric, baseline, horizon, and tolerance,
  optional observations, bounded failure state, strategy transitions, help
  escalation, abandonment, human-readable reasons, and scenario aggregates.
- Add a provenance-backed semantic relation graph to the canonical world model,
  grounding charger, obstacle, behavior, skill, goal, drive, and outcome
  meanings with contextual confidence and contradiction tracking.
- Expose semantic evidence observations and graph queries, including
  charger explanations and causal-grounding safeguards that downgrade
  unsupported causal claims to predictions.
- Surface charger semantic explanations, supporting relation IDs, and grounded
  energy-meaning confidence in goal evaluations and affordances.
- Project canonical semantic relations into durable graph-memory entities and
  edges, preserving grounding metadata and confidence for recall.
- Ground charger approach, docking, and withdrawal predictions in observed
  runtime action outcomes, retaining their evidence provenance across ticks.
- Add a safety-gated sleep lifecycle that quiesces deliberative control, plans
  bounded consolidation, replay, training, and evaluation work, defers
  unavailable accelerator work, and wakes for operator, safety, power, or
  body-link events without automatically promoting candidates.

### Changed

- Refactor `crates/pete-tools/src/main_tests.rs` into domain-specific modules
  under `crates/pete-tools/src/main_tests/` and re-export them through
  targeted `include!` statements.

- Retire the action-producing `event_face_detected` behavior. Face events now
  remain perception evidence: canonical social `Now` opens and closes
  encounters, the conductor arbitrates acknowledgment, and `greet.lua`
  coordinates orientation and speech before recording success.
- Move deterministic motherbrain sequencing, including IR docking and
  post-reflex bump/cliff recovery, into ordinary runtime-loaded Lua functions;
  retain numerical controllers, possession, 250 ms primitive renewal, and
  hazard-envelope validation in Rust, and use the identical scheduler and
  scripts in simulation and on Pete.
- Interpret Home Base red, green, and force-field IR as directional charger
  evidence in Pete's sensorium, bias charger-seeking without implying contact,
  and let `AlignWithDock` follow that gradient with 250 ms primitives until
  charging or Home Base contact; lost IR fails stopped.
- Restrict the public Brainstem contract to bounded motor primitives,
  body-native operations, immutable reflexes, and explicit services; run
  deterministic navigation and recovery skills in the motherbrain through
  short TTL renewals, reject retired convenience and safety-policy wire verbs
  in firmware and simulation, remove their firmware runtime implementations,
  queue expanders, browser controls, and shared policy constants, validate the
  advertised verb inventory at build time, and document the remaining
  hardware-in-loop acceptance gates.
- Document the grounded semantic graph and safety-gated sleep lifecycle,
  including their authority boundaries, artifact contracts, and replacement
  points.
- Move repository automation into the typed Rust `xtask` command layer while
  retaining the public `just` recipes as thin compatibility aliases. Preserve
  brainstem identity continuity, authorized BOOTSEL flashing, automatic NEAT
  continuation, and clean worktree synchronization, and add the portable
  repository-root `wuzzup` shorthand for `just sup`.
- Removed retired behavior configuration spellings (`mode`, `hardcoded_on_error`, and `stop_on_error`) and compatibility type aliases; configurations must use `regime` and the canonical fallback values.

### Fixed

- Replace timid tick-count contact recovery with odometry-gated escape phases:
  reverse 80-160 mm at the possession speed limit, turn until 1.57 radians are
  observed, probe 50 mm, alternate and escalate bounded retries, then stop when
  body feedback proves no mechanically useful progress; include phase progress
  in the live possession trace.
- Keep complete brainstem body evidence available when an optional Kinect or
  other sensor poll fails, publish named per-sensor health, rate-limit repeated
  failure reports, stop failed libfreenect workers, and back off hardware
  retries so device diagnostics do not bury the possession trace.
- Open `just cockpit` motion commands as operator-control sessions, accept
  ordinary negative velocity arguments, and report Home Base arrival before a
  preserved terminal safety latch so successful physical docking is not
  mislabeled as a failed run.
- Treat Create Home Base contact as dock geometry in browser and motherbrain
  status instead of promoting the dock's bumper-plus-cliff pattern into a
  multi-sensor safety incident; raw Cockpit sensor bits remain available for
  diagnostics, and wheel-drop evidence remains authoritative.
- Reconcile a bump/cliff latch raised before the private Home Base packet
  arrives, preventing the dock's packet-order race from leaving a stale cliff
  latch while preserving wheel-drop and every stronger safety latch.
- Keep Home Base contact normalized throughout the bounded dock-departure
  reverse, instead of letting alternating packet-0 and packet-34 observations
  repeatedly trip and clear the dock's cliff bits while stopping motion.
- Prevent the browser's 900 ms motion heartbeat from cancelling the fixed
  body-local dock departure, give its 200 mm/s by 1.5-second envelope enough
  breakaway torque and travel to clear the Home Base, prevent periodic OI Full
  supervision from zeroing active motor output, expose wheel-overcurrent in
  browser and compact diagnostics, and provide an Undock control without
  requiring a held drive button; cancel an unstarted departure when fresh Home
  Base telemetry clears so it cannot surprise a later off-dock command, and
  expire stale Create-link evidence before it can authorize motion.
- Keep authorized `just flash` BOOTSEL negotiation compatible with older
  brainstems that advertise retired convenience verbs, while preserving the
  primitive-only contract for newly built firmware.
- Make contact withdrawal an immutable brainstem safety reflex: only a fresh
  bumper edge during forward output can start the bounded reverse, held or
  stationary contact cannot initiate motion, active physical hazards cannot be
  cleared, and diagnostic safety-policy commands cannot weaken firmware rules.
- Accept checksum-valid Create group-0 sensor frames even when battery
  telemetry is unusual, preserving their independently validated safety data
  instead of repeatedly surfacing opaque `error 4` messages; spell out
  brainstem error event names in the Pico W operator frontend.
- Accept older brainstem handshake frames that predate the nested software
  version field so an authorized USB BOOTSEL upgrade can still proceed.
- Keep Create 1 dock departure inside the brainstem: after Full mode ends
  charging, privately poll its Home Base source packet to hold the first
  nonzero motion request, perform a bounded reverse off the dock, then execute
  the original request without exposing a charging latch that callers must
  clear, including after a brainstem-only restart.
- Consume sleep inputs per successfully completed work kind, keeping deferred,
  failed, and cancelled training inputs eligible when resources return, and
  declare canonical world-model schema 3 in sleep provenance.
- Credit semantic action outcomes only when the autonomous goal primitive was
  actually executed unchanged through safety, and measure progress against the
  same canonical world-model target identity rather than raw observations.
- Treat stale or missing target progress as unknown instead of failure, let
  Explore change between multiple strategies without dropping its goal, and
  make repeated charger-search failure prefer help before bounded abandonment.
- Keep autonomic safety preemption separate from the interrupted possessor
  skill's intended progress.
- Preserve context-distinct semantic relations through graph-memory
  deduplication and Neo4j persistence by carrying `SemanticRelationId` as the
  stable edge identity instead of collapsing edges by triple alone, and
  transactionally backfill stable identities onto legacy `RELATED` projections
  without deleting historical graph relationships.
- Give each possessor skill execution a stable id, count attempts once per
  execution rather than per motor refresh, expose dispatches separately, and
  consume each terminal failure only once before retrying.
- Preserve lifecycle telemetry when velocity and heartbeat commands coalesce:
  identical velocity refreshes renew one streaming command without restarting
  the motor or transferring lifecycle ownership, and every replaced accepted
  command receives a terminal event.
- Close accepted pending command lifecycles when Stop or E-stop preempts them,
  retain 128 brainstem audit events, and page bounded event responses so routine
  velocity renewals cannot overflow Pico transport buffers.
- Enforce sleep power and thermal budgets on every work item: candidate
  training/evaluation require a stable powered dock, while a rising thermal
  limit interrupts sleep before subsequent maintenance can run.
- Make sleep-session admission edge-triggered and consume completed input
  evidence exactly once, preventing continuous fatigue, a held operator request,
  or already-consolidated work from immediately starting duplicate sessions.
- Add regression coverage and lifecycle documentation for consumed deferred
  references, fatigue recovery, newly observed failures, and held/released
  operator sleep requests.
- Keep every E-stop latched through bump/contact recovery unless an operator
  explicitly clears it; possessor recovery no longer infers E-stop provenance
  from event timing or clears an E-stop while releasing the bump latch.
- Keep Pico-W cockpit operator-control refresh aligned with the granted lease,
  prevent concurrent browser handshakes, and retry control acquisition after
  transient session or lease failures.
- Use `portable-atomic` for brainstem registry and session counters in the default RP2040 build as well as Pico-W builds.
- Pin the Pico-W firmware's `fixed` dependency to the Rust-1.92-compatible 1.30 release so the documented embedded build remains reproducible.
- Add a bounded `careful_mode` runtime verb for active possessors: it clears advisory latches for the command window, suspends automatic safety gating while a short TTL is active, then re-latches hazards and authority conditions on expiry while exposing remaining-time telemetry.

### Ready

- Add typed temporal clock domains, timed predictions, and deterministic charging, conversation, recovery, exploration, and task episodes to the canonical world model.
- Add persistent uncertain people, relationship, presence, identity, and interaction beliefs shared by goals and durable memory without granting social trust any motor authority.
- Add explicit epistemic questions and information-gathering affordances whose progress is measured by actual uncertainty reduction, including charger-seeking inquiry that preserves goal commitment.
- Persist temporal, social, and epistemic snapshots in memory and expose social and episode structure as queryable durable graph records.
- Make a bump-triggered contact withdrawal a bounded brainstem-local reflex: it continues across authority changes, records start and terminal safety events, and remains preemptible by stronger safety conditions.
- Expose the contact-withdrawal lifecycle through typed cockpit events and mirror its bounded, authority-independent behavior in the simulator.
- Route selected goal affordances through typed, bounded possessor skills, with target-based turn, approach, docking, search, and retreat requests that report completion, timeout, unavailable-target, authority-loss, and safety-preemption outcomes.
- Feed possessor-skill progress and terminal outcomes back to the originating goal, including explicit progress expectations and failure pressure without discarding the active goal.
- Add `careful_mode` to the brainstem verb inventory and compatibility checks, including firmware parser/runtime/status support for both line-based and JSON control payloads and compact status fields for active/remaining time.
- Keep `codex` sync automation behavior stable by pinning the sync model to `gpt-5.3-codex-spark` and forcing `high` model reasoning effort for summary generation.
- Let `just sup` complete ready work by explicitly authorizing its nested Codex run to use Git for the requested commit, fast-forward pull, and push workflow.
- Make `just possess` default to `read-only` robot mode via CLI `--mode regular`, then normalize regular/read-only mode aliases so operator scripts can use the regular launcher profile without changing possession behavior.
- Add `--mode` parsing in `xtask possess` and propagate a normalized robot mode into `PETE_ROBOT_MODE`, preserving other possession arguments and retries across backend fallback and identity-acceptance flows.
- Add a user-facing possession mode rename in `pete-tools`: introduce explicit `regular` mode alongside `read-only` and `possession-slow`, and route `regular` to the existing physical slow possession behavior (`RobotMode::Slow`) while preserving guardrails.
- Update `--recovery-smoke`, `--orientation-probe`, and serial/session validation paths to refer to `regular` possession language, and keep all regular/slow possession diagnostics aligned to physical brainstem requirements.
- Refresh dependency-lock metadata in `Cargo.lock` for speech/visualization crates (`tongues-*`, `textplots`, `drawille`, version-resolved `indicatif`/`console`/`colored`) so speech pipeline, lockfile, and package metadata stay consistent after the mode/workflow updates.
- Add a second lockfile maintenance pass that removes stale `tongues-*`, `drawille`, `textplots`, `number_prefix`, and duplicate older `console`/`indicatif`/`colored` entries while normalizing dependency edges to current workspace versions.
- Update the workspace lockfile with a dependency graph refresh that adds missing `textplots`/`rgb`/`drawille`/`tongues-*` package metadata and reconciles `indicatif`/`console`/`colored` version constraints used by updated consumer crates.
- Refresh `Cargo.lock` graph deduplication again to remove stale transitive duplicates (`colored`/`console`/`indicatif` version pins, `number_prefix`, `textplots`, `drawille`) and align dependency edges to current minor-compatible versions.

### Ongoing

- Migration of face, object, and voice features to `VectorArtifact`-only schema fields. Downstream runtime, training, and event consumers still require coordinated updates, so this is not yet a completed release change.

### Cognitive provider health follow-up (2026-07-15)

- Keep optional scene providers degraded while a request is in flight, mark them available only after a successful response, and record deadline failures as degraded health.
- Serialize cognitive role, locality, and resource-class fields with their stable snake-case identifiers in self-model service beliefs.

- Added a `sup` target in `Justfile` as a short alias for `./scripts/codex-sync.sh`.
- Refactored `scripts/codex-sync.sh` to cache both short and porcelain git status once per run.
- In clean-sync mode, `codex-sync.sh` now prints branch/status context and syncs from origin with `git pull --ff-only`.
- In clean-sync mode, local ahead status now triggers an automatic `git push` after successful pull.
- Added temporary-file based output capture in `codex-sync.sh` for robust summary propagation from nested `codex` execution.
- Updated the embedded codex instructions in `scripts/codex-sync.sh` to enforce an expanded repo-sync workflow (status inspection, changelog update, conflict-aware pull/push, and final summary format).
- Refined `codex-sync` workflow instructions to preserve in-progress/Ongoing edits (e.g., TODO, FIXME, WIP, debug scaffolding) and commit only clearly ready, substantial chunks.

### Auto-sync (2026-07-16)

- Added a new `pete-cognition` crate with common cognitive provider contracts, request/response types, and router abstractions.
- Wired `pete-cognition` into the workspace and into `pete-llm`/`pete-now` through path dependencies.
- Reworked live image enrichment in `pete-llm` around an asynchronous `LiveImageCognition` pipeline with request routing and registry snapshots.
- Implemented an ollama-backed `CognitiveProvider` for scene-description requests and embeddings, producing structured scene/caption vectors.
- Extended runtime `Now` cognition bookkeeping to include provider registry snapshots, last scene response, and vision enrichment error metadata.
- Extended cognitive service belief state to persist provider metadata and capabilities in `pete-now` world summaries.

### Cognitive provider refinement (2026-07-15)

- Reject completed scene-description responses when their source frame has been replaced, so stale enrichment cannot overwrite newer perception.
- Project cognitive provider health and capabilities into the canonical self-model service state while keeping organism identity independent of provider restarts.
- Added role-neutral accelerator capability descriptors and documentation for cognitive roles, bounded requests, and graceful local fallback.
- Added focused coverage for unavailable or slow optional cognition and provider disconnect/restart behavior.

## 2026-07-15

### Added

- Introduced canonical self-model integration for goal selection:
  `Now.world.self_model` now carries organism/body/capability/authority/motivation/action-state identity in a typed model.
- Added capability-aware action gating in the goal evaluator:
  behaviors now require specific capabilities (`actuator:drive`, `actuator:speaker`, `sensor:vision`) and are rejected with explicit reasons when unavailable.
- Added a capability registry in `pete-now` with availability (`available`, `degraded`, `unavailable`, `unknown`) and typed IDs.
- Added canonical self-model serialization/tests and documented the new model contract.

### Changed

- The system now uses `goal` as the default action selector in:
  - `Justfile` virtual-live startup
  - `pete-tools` CLI simulation/eval defaults (`sim`, `eval-scenario`)
  - live server status default mode
- LLM prompts now include a rendered canonical self-model context so prompts reflect current capability/state boundaries.
- Runtime `Now` updates now include richer self-model context:
  registered goals/behaviors, registered drive summaries, commitment age/progress, uncertainty/failure pressure,
  and active control provenance.
- Goal architecture now emits and tests a stronger capability-aware failure path for action rejection.

### Fixed

- Missing capabilities now correctly propagate to both planning and language reasoning paths, preventing unsafe/invalid action assumptions.
- Service capability handling now handles higher-brain availability cleanly (e.g., richer cognition disabled state is reflected as unavailable when appropriate).

### Documentation

- Added `docs/021-self-model.md` describing the self-model schema and update/consumer rules.
- Updated architecture docs (`001`, `019`, `020`) to describe self-model as canonical PETE state and the new default behavior.

## 2026-07-15 (prior)

- Persisted `Now.world` as canonical belief state.
- Introduced goal-driven arbitration with a persistent world model pipeline.

## 2026-07-14

- Improved NEAT species persistence and selection workflows.
- Added training/evaluation scenario support and model promotion gates in locomotion training.
