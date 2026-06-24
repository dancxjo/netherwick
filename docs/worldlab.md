# Netherwick Worldlab

`netherwick-worldlab` records streams of `WorldSnapshot` values into reusable capture sessions and replays those sessions back through the normal runtime pipeline.

The v0 loop is:

```text
VirtualWorld or sensors -> WorldSnapshot stream -> capture session -> replay -> Now stream -> MinimalRuntime.tick -> ledger
```

Replay mode observes the recorded world as-is. It does not apply chosen robot actions back into the capture, edit objects, or run counterfactual branches. Counterfactual replay is a later layer.

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

`manifest.json` stores the capture id, source, schema version, tick duration, frame count, optional machine info, command args, device availability, streams present/missing, capture start/end times, warnings, and asset layout. `frames.jsonl` contains one JSON record per captured frame with `index`, `t_ms`, the serialized `WorldSnapshot`, and any recorded events. The asset directories are reserved for future RGB, depth, audio, and point-cloud files; v0 embeds compact sense data directly in JSON.

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

Record a bounded real read-only session:

```bash
cargo run -p netherwick-tools -- capture-real \
  --duration-seconds 60 \
  --out data/captures/real/rpi5-smoke
```

Inspect a capture:

```bash
cargo run -p netherwick-tools -- inspect-capture \
  data/captures/real/rpi5-smoke
```

## Future Path

Planned layers include raw RGB/depth/audio asset export, point-cloud reconstruction, a game-engine renderer, semantic editing, and counterfactual replay.
