# Worldlab Capture Assets

Worldlab captures keep the replayable `WorldSnapshot` stream in `frames.jsonl` and store heavier media under `assets/`. Frame asset paths are relative to the capture root so sessions can be moved as directories.

```text
data/captures/<capture-id>/
  manifest.json
  frames.jsonl
  events.jsonl
  assets/
    rgb/
      000000.png
    depth/
      000000.depth16.png
    audio/
      000000.wav
    pointcloud/
      000000.ply
```

Frame records may include:

```json
{
  "index": 0,
  "t_ms": 1234,
  "snapshot": {},
  "events": [],
  "assets": {
    "rgb": "assets/rgb/000000.png",
    "depth": "assets/depth/000000.depth16.png",
    "audio": "assets/audio/000000.wav",
    "pointcloud": "assets/pointcloud/000000.ply"
  },
  "stream_metadata": {
    "rgb": {"width": 2, "height": 2, "format": "rgb8_png"},
    "depth": {"width": 3, "height": 1, "format": "gray16_png", "units": "millimeters", "scale": 0.001},
    "audio": {"sample_rate_hz": 16000, "channels": 1, "format": "pcm16_wav"},
    "pointcloud": {"format": "ply_ascii", "stride": 4, "calibration": "uncalibrated"}
  }
}
```

Every capture frame includes `assets` and `stream_metadata`; regenerate captures
created before those fields existed.

## Formats

RGB assets are PNG files written as RGB8. Real camera frames are exported when a raw `EyeFrame` is available in a supported format. Mock captures produce deterministic tiny RGB frames so the pipeline can be tested without hardware. Producer time, source, device time, and RGB-D frame identity accompany the asset when available.

Depth assets are 16-bit grayscale PNG files containing millimeters. Declared
image dimensions are required, so a real Kinect frame remains a full 640x480
depth specimen rather than being guessed from a compact feature vector.

Audio assets are WAV PCM16 chunks from `PcmAudioFrame`, with sample rate,
channel count, and producer time recorded in frame metadata.

`robot --capture` exports all three raw asset types automatically, including
captures started by `just possess-sensorium`. The standalone `capture-real`
command uses its `--export-rgb`, `--export-depth`, and `--export-audio`
switches.

Per-frame point-cloud v0 assets are ASCII PLY files generated from depth. The conversion uses a max-depth filter and a stride/downsample factor. Until calibrated intrinsics are supplied, Worldlab uses approximate placeholder intrinsics and writes an `uncalibrated point cloud` warning into the manifest.

The accumulated world PLY also consumes calibrated `RangeSense` scans. A pitched HLS-LFCD2 therefore shares the same odometry-aligned voxel cloud with Kinect depth; capture poses from forward motion or a slow spin supply the third dimension. Run `capture-assets` with `--world-pointcloud`, or use `capture-real --export-pointcloud`, to write `assets/pointcloud/world-accumulated.ply`.

## Commands

```bash
cargo run --bin pete -- capture-real \
  --duration-seconds 3 \
  --mock \
  --out data/captures/real/mock-assets-smoke \
  --export-rgb \
  --export-depth \
  --export-audio

cargo run --bin pete -- capture-assets \
  --capture data/captures/real/mock-assets-smoke \
  --pointcloud \
  --stride 4

cargo run --bin pete -- inspect-capture \
  data/captures/real/mock-assets-smoke
```

`inspect-capture` reports asset counts, representative dimensions or sample rates, and calibration warnings. Replay ignores external assets for now, so captures remain useful even when asset files are absent or when older captures have no asset references.

These assets are the bridge from captured observations to later reconstruction: RGB preserves appearance, depth preserves scene geometry, audio preserves timed sound chunks, and PLY point clouds provide a simple import target for game-engine or visualization experiments.
