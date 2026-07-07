# Kinect / OpenKinect Path

Pete keeps Kinect support optional so the default simulator and Linux sensor build stay light.

## Feature

```sh
cargo check -p pete-sensors --features kinect-freenect
```

The `kinect-freenect` feature exposes `FreenectKinectProvider`, which uses libfreenect sync reads and emits:

- `SensePacket::Kinect(KinectSense)` for depth/color-derived features
- `SensePacket::EyeFrame(EyeFrame)` when RGB frames are available

Kinect audio should be added later. OpenKinect/libfreenect audio support can require firmware handling, and vision should not be blocked on that.

The existing V4L camera, CPAL microphone, and serial GPS provider are behind:

```sh
cargo check -p pete-sensors --features linux-hardware
```

## Linux Build Notes

OpenKinect/libfreenect is the upstream userspace driver for Microsoft Kinect.

Ubuntu/Debian/Mint manual build dependencies:

```sh
sudo apt-get install git cmake build-essential libusb-1.0-0-dev
```

Examples also need:

```sh
sudo apt-get install freeglut3-dev libxmu-dev libxi-dev
```

Install the libfreenect udev rules for non-root device access. The exact rules file depends on the libfreenect package or checkout you use; after installing it, reload udev rules and replug the Kinect.

## Dark RGB Frames

Kinect RGB frames are read through libfreenect, not `/dev/video*`, so V4L exposure controls usually do not apply. Pete applies a software RGB correction before the frame enters the dashboard or capture ledger.

The default `just robot` path enables auto-gain with a moderate gamma lift:

```sh
KINECT_RGB_TARGET_LUMA=0.38 KINECT_RGB_AUTO_GAIN_MAX=4.0 KINECT_RGB_GAMMA=0.70 just robot
```

Useful knobs:

- `KINECT_RGB_TARGET_LUMA`: desired mean luma, `0.0..1.0`; raise this if the Kinect image is too dark.
- `KINECT_RGB_AUTO_GAIN_MAX`: cap for automatic gain; raise cautiously if the room is very dim.
- `KINECT_RGB_GAIN`: fixed multiplier applied before auto gain.
- `KINECT_RGB_GAMMA`: values below `1.0` lift shadows; values above `1.0` darken.
- `KINECT_RGB_BRIGHTNESS`: additive offset in normalized RGB units, usually keep near `0.0`.

Pass `--kinect-rgb-raw` to `pete-tools robot` or `capture-real` to disable software correction and inspect the raw libfreenect RGB frame.

## Replay Recordings

`KinectReplayProvider` reads repo-native recordings shaped like:

```text
data/recordings/kinect-session/
  rgb/
  depth/
  timestamps.jsonl
```

Each JSONL row can point at raw RGB bytes and depth values:

```json
{"t_ms": 1, "rgb_path": "rgb/frame-000001.raw", "depth_path": "depth/frame-000001.json"}
```

Depth files may be JSON arrays of meters or whitespace-separated floats. RGB bytes are converted into compact `EyeSense` features; full image decoding can be layered in later if a recording format needs width/height metadata.

## Calibration

Point-cloud rendering must be driven by calibration data, not by viewer guesses. Live scene generation uses sensor calibration metadata alongside arena/object metadata, and `/view/scene` returns the active calibration for auditing:

- `compact_depth_beam_count`: number of compact range samples, for example `32` in the simulator.
- `compact_depth_fov_rad`: measured horizontal fan width for compact depth samples.
- `depth_scale`: multiplier from raw depth units into meters.
- `point_y_m`: sensor height used when compact range samples are drawn as 3D points.

For real hardware, run a calibration capture with a flat target at known distances, for example `0.5m`, `1.0m`, and `2.0m`, centered and then near the left/right edge of the usable view. Fit `depth_scale` from measured depth versus tape-measured distance, and fit `compact_depth_fov_rad` from the side target angle where the edge samples first line up. Save those values with the sensor profile and attach them to live scene metadata; replay/capture views should use the calibration stored with the capture.

Until the freenect path emits full Kinect intrinsics, compact depth should be treated as a calibrated range fan. Full depth images should later use camera intrinsics (`fx`, `fy`, `cx`, `cy`) plus robot-frame extrinsics instead of the compact fan fields.
