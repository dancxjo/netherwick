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

The report includes odometry edges, gated loop-closure candidate edges from conservative place recognition, confidence buckets, and rejected low-confidence candidates.

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

Planned layers include real Kinect raw-frame hooks, calibrated camera intrinsics, a game-engine renderer, semantic editing, and counterfactual replay.
