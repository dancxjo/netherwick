# Netherwick Worldlab

`netherwick-worldlab` records streams of `WorldSnapshot` values into reusable capture sessions and replays those sessions back through the normal runtime pipeline.

The v0 loop is:

```text
VirtualWorld or sensors -> WorldSnapshot stream -> capture session -> replay -> Now stream -> MinimalRuntime.tick -> ledger
```

Replay mode observes the recorded world as-is. Counterfactual replay is now available for sim captures with reconstructable scenario metadata; see [counterfactual-replay.md](counterfactual-replay.md).

## Capture Layout

Each capture is a directory:

```text
data/captures/<capture-id>/
  manifest.json
  frames.jsonl
  events.jsonl
  assets/
    rgb/
    depth/
    audio/
    pointcloud/
```

`manifest.json` stores the capture id, source, schema version, tick duration, frame count, optional machine info, command args, device availability, streams present/missing, capture start/end times, warnings, and asset layout. `frames.jsonl` contains one JSON record per captured frame with `index`, `t_ms`, the serialized `WorldSnapshot`, any recorded events, optional frame asset references, and optional stream metadata. Asset paths are relative to the capture root.

## Commands

Record a simulated session:

```bash
cargo run -p netherwick-tools -- capture-sim \
  --out data/captures/sim-test \
  --steps 100 \
  --seed 7
```

Replay a capture into the runtime and write a normal ledger:

```bash
cargo run -p netherwick-tools -- replay-capture \
  --capture data/captures/sim-test \
  --ledger data/ledger/replay-test
```

Replay output uses the existing `JsonlLedger` conventions, so ledger frames and transitions can feed the same inspection and training paths as live simulation runs.

Produce an offline pose graph report directly from a capture without correcting live pose:

```bash
cargo run -p netherwick-tools -- pose-graph-report \
  --capture data/captures/sim-test \
  --out data/reports/pose-graph/sim-test.json
```

The report includes odometry edges, gated loop-closure candidate edges from conservative place recognition, confidence buckets, and rejected low-confidence candidates. Ledger replay feeds loop candidates from canonical `PlaceRecognitionInput` with Experience/Instant provenance; direct capture replay uses available scene vectors as a capture-only fallback.

Live map integration now uses the same conservative candidate shape:

```text
Now
  -> PlaceRecognitionInput
  -> PlaceMemory.recognize_places(...)
  -> PlaceMemory.recognize_entity_constellations(...)
  -> Vec<LoopClosureCandidateInput>
  -> LocalMap.integrate_observation_with_loop_candidates(...)
  -> anchored pose graph optimization
  -> occupancy rebuild from corrected submaps
```

`LocalMap` only promotes candidates when a live pose node is created. Accepted candidates become active `PoseEdgeSource::LoopClosureCandidate` edges; rejected candidates remain inactive edges with `rejection_reason` values for replay/debugging. The live gates are intentionally conservative: confidence must clear the loop threshold, the target must not be the current/source frame, the target pose must be close enough to the current pose to avoid teleportation, and current range geometry must overlap existing occupied cells at the proposed prior node. `MapSummary` surfaces total, accepted, and rejected loop-closure counts.

The runtime now asks the configured `Recall` implementation for conservative loop candidates before storing the current frame, converts those memory candidates into `LoopClosureCandidateInput`, and feeds them into `LocalMap` for the same tick. Representation reports also run `LocalMap` through this candidate-aware path, while the standalone `pose-graph-report` command remains an offline diagnostic view of the same place/entity evidence.

Build a replay-first representation health report from capture or ledger input:

```bash
cargo run --bin netherwick -- representation-report \
  --capture data/captures/real/rpi5-smoke \
  --out data/reports/representation/rpi5-smoke.json
```

```bash
cargo run --bin netherwick -- representation-report \
  --ledger data/ledger/virtual-live \
  --out data/reports/representation/latest.json
```

Record a bounded real read-only session:

```bash
cargo run -p netherwick-tools -- capture-real \
  --duration-seconds 60 \
  --out data/captures/real/rpi5-smoke
```

Record a mocked session with RGB, depth, and audio assets:

```bash
cargo run --bin netherwick -- capture-real \
  --duration-seconds 3 \
  --mock \
  --out data/captures/real/mock-assets-smoke \
  --export-rgb \
  --export-depth \
  --export-audio
```

Export point-cloud v0 assets from captured depth:

```bash
cargo run --bin netherwick -- capture-assets \
  --capture data/captures/real/mock-assets-smoke \
  --pointcloud \
  --stride 4
```

Inspect a capture:

```bash
cargo run -p netherwick-tools -- inspect-capture \
  data/captures/real/rpi5-smoke
```

See [worldlab-assets.md](worldlab-assets.md) for the asset formats and calibration assumptions.

## Future Path

Planned layers include real Kinect raw-frame hooks, calibrated camera intrinsics, a game-engine renderer, semantic editing, and counterfactual replay. Real hardware should still be treated as gated: do not rely on live SLAM corrections from a real capture until the geometry report says `sensor_truth.ready_for_real_slam = true`.
