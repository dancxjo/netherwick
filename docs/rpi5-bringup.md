# Raspberry Pi 5 Capture-First Bring-Up

This path is for Pete's first Raspberry Pi 5 hardware milestone: observe and record real device state while keeping the wheels quiet.

Motor movement is not enabled by this task. `robot --mode read-only` and `capture-real` read body and sensor state, write ledgers/captures, and record read-only motor suppression. They must not command the Create motors.

## Raspberry Pi OS Assumptions

Use a current 64-bit Raspberry Pi OS or Debian/Ubuntu-derived image with Rust installed. The commands assume a normal shell user, writable project checkout, and access to USB, I2C, video, and audio devices through Linux groups.

Install the workspace prerequisites:

```bash
sudo apt-get update
sudo apt-get install -y just build-essential pkg-config cmake ninja-build git curl ffmpeg i2c-tools v4l-utils libasound2-dev libudev-dev libusb-1.0-0-dev libv4l-dev
```

For Kinect 1/libfreenect:

```bash
sudo apt-get install -y libfreenect-dev freenect
```

If distro packages are missing, use:

```bash
just setup-kinect-from-source
```

## Permissions

Add the user to the common hardware groups, then log out and back in:

```bash
sudo usermod -aG dialout,i2c,video,audio "$USER"
```

`dialout` is normally required for `/dev/ttyUSB*` or `/dev/ttyACM*` brainstem, USB GPS, and HLS-LFCD2 lidar devices. `i2c` is normally required for `/dev/i2c-*` IMU access. `video` is normally required for `/dev/video*` camera access. `audio` may be required for microphone capture.

Enable the Pi I2C bus only for an explicitly Motherbrain-local diagnostic MPU-6050. The normal
brainstem-connected IMU does not require `/dev/i2c-1` on Motherbrain:

```bash
sudo raspi-config nonint do_i2c 0
sudo reboot
```

## Device Expectations

The brainstem Cockpit UART should appear as one of:

```text
/dev/ttyUSB0
/dev/ttyACM0
/dev/serial/by-id/<adapter>
```

The expected Cockpit UART baud rate is `115200`.

`robot` and `capture-real` default to `--cockpit uart` and the first likely serial candidate from `hardware-env`. Pass `--cockpit uart --create-port /dev/ttyUSB0` when you want to pin the adapter, `--cockpit sim` for simulated Cockpit smoke tests, or `--mock` for `capture-real` no-hardware smoke tests.

u-blox7 GPS receivers are read over USB serial at 9600 baud using NMEA. Pete auto-starts GPS on real runs by preferring `/dev/serial/by-id/*u-blox*`, `/dev/serial/by-id/*gps*`, or `/dev/serial/by-id/*gnss*`, then falling back to an unused `/dev/ttyACM*` device. Use `--gps /dev/serial/by-id/<u-blox-device>` to pin it, or `--gps none` to disable GPS capture.

HLS-LFCD2 / ROBOTIS LDS-01 lidars are read directly at 230400 baud; ROS is not required. Pete auto-starts a stable serial path whose name contains `hls-lfcd`, `lfcd2`, `usb2lds`, `lidar`, or `lds-01`, and reserves that device so Cockpit auto-selection cannot claim it. Generic FTDI adapters often have no recognizable name, so pin those explicitly:

```bash
LIDAR_SERIAL_PORT=/dev/serial/by-id/<usb2lds-device> just robot --require-lidar
```

The same environment variables carry through `just possess`. Describe the physical mount in `.env` (illustrative values—replace them with measurements):

```dotenv
LIDAR_HEIGHT_M=0.42
LIDAR_FORWARD_M=0.18
LIDAR_LEFT_M=0
LIDAR_PITCH_DEG=20
LIDAR_ROLL_DEG=0
LIDAR_YAW_DEG=0
```

Coordinates are robot-relative: forward, left, and up are positive. Positive pitch points the front of the scan plane down. The provider emits 360 one-degree `RangeSense` beams. Calibrated endpoints join Kinect points in the shared odometry-aligned voxel cloud; successive poses during forward motion or a spin build the 3D surface. Floor returns remain in 3D but are excluded from the planar obstacle map. Use `--lidar none` to disable the sensor.

Because LFCD2 publishes full sweeps rather than per-beam poses, rotate slowly for cleaner geometry. Point-cloud quality is bounded by lidar mount calibration, odometry/IMU accuracy, and motion during each scan.

Kinect availability is detected best-effort through `freenect-*` tools or `pkg-config libfreenect`. A missing Kinect is a warning for capture-first bring-up, not a failure when other streams are useful.

Camera devices are expected under `/dev/video*`. Microphones should be visible to ALSA, for example through `arecord -l`.

The normal MPU-6050 is attached to brainstem Pico I2C1 (GP2/GP3) and is discovered through Cockpit status; Pete does not assume `/dev/i2c-1` exists on Motherbrain. A separately wired local diagnostic MPU must be explicit, for example `--imu-source local-i2c --imu /dev/i2c-1@0x69`. Use `--imu-source none` to disable fusion IMU without weakening brainstem tilt/impact safety. See [Brainstem IMU handoff](brainstem-imu-handoff.md).

For that separately Motherbrain-local wiring only, this should show address `68`:

```bash
i2cdetect -y 1
```

## Safe First Commands

Inspect the hardware environment:

```bash
cargo run --bin pete -- hardware-env
cargo run --bin pete -- hardware-env --json
```

Bring up the default real hardware stack in slow mode with Kinect depth when available:

```bash
just robot
```

Run a bounded read-only robot smoke:

```bash
cargo run --bin pete -- robot \
  --mode read-only \
  --duration-seconds 30 \
  --ledger data/ledger/real/read-only-smoke
```

Record a real capture session:

```bash
cargo run --bin pete -- capture-real \
  --duration-seconds 60 \
  --out data/captures/real/rpi5-smoke
```

Inspect the capture:

```bash
cargo run --bin pete -- inspect-capture \
  data/captures/real/rpi5-smoke
```

No-hardware smoke test:

```bash
cargo run --bin pete -- capture-real \
  --duration-seconds 3 \
  --mock \
  --out data/captures/real/mock-smoke
```

## What Success Looks Like

`hardware-env` reports OS, architecture, CPU, memory, likely serial devices, GPS and HLS-LFCD2 lidar candidates/defaults, I2C devices, default MPU-6050 bus/pins, cameras, audio inputs, Kinect/libfreenect availability, permissions hints, writable data directories, and whether the host looks like Raspberry Pi hardware.

`capture-real` writes:

```text
manifest.json
frames.jsonl
events.jsonl
assets/rgb/
assets/camera/
assets/depth/
assets/lidar/
assets/imu/
assets/audio/
assets/transcript/
assets/calibration/
assets/pointcloud/
```

The manifest includes machine info, command args, device availability,
present/missing streams, start/end times, warnings, and the asset layout.
`robot --capture` writes lossless Kinect RGB and camera PNG, depth16 PNG, lidar
JSON, selected/candidate IMU JSON, PCM16 WAV, transcript JSON, calibration JSON,
and calibrated derived PLY files when those source streams are present. A
bounded background queue keeps encoding and disk I/O off the possession loop;
its dropped frame/asset counters are finalized in the manifest.

`inspect-capture` should show frame count, duration, streams present/missing,
first/last timestamps, event count, and per-asset counts, producer-time ranges,
missing intervals, bytes, unavailable/late/partial/dropped totals, checksum
failures, and warnings.

## What Failure Looks Like

Missing camera, mic, Kinect, GPS, or IMU should produce warnings while still capturing any available cockpit or sensor stream.

Cockpit UART open failures are clear errors unless `--mock` or `--cockpit sim` is used.

If no useful cockpit or sensor stream can be captured, `capture-real` exits with a clear error instead of writing a misleading success.

Any mode that would command motors is refused. Movement-capable bring-up must be a separate future task with explicit safety gates and tests.
