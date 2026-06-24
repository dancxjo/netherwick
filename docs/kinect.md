# Kinect / OpenKinect Path

Netherwick keeps Kinect support optional so the default simulator and Linux sensor build stay light.

## Feature

```sh
cargo check -p netherwick-sensors --features kinect-freenect
```

The `kinect-freenect` feature exposes `FreenectKinectProvider`. It is currently a skeleton for a libfreenect FFI wrapper or a subprocess-backed first pass, and should emit:

- `SensePacket::Kinect(KinectSense)` for depth/color-derived features
- `SensePacket::Eye(EyeSense)` when RGB frames are available

Kinect audio should be added later. OpenKinect/libfreenect audio support can require firmware handling, and vision should not be blocked on that.

The existing V4L camera, CPAL microphone, and serial GPS provider are behind:

```sh
cargo check -p netherwick-sensors --features linux-hardware
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
