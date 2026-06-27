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

`dialout` is normally required for `/dev/ttyUSB*` or `/dev/ttyACM*` Create serial and USB GPS devices. `i2c` is normally required for `/dev/i2c-*` IMU access. `video` is normally required for `/dev/video*` camera access. `audio` may be required for microphone capture.

Enable the Pi I2C bus before plugging in the MPU-6050:

```bash
sudo raspi-config nonint do_i2c 0
sudo reboot
```

## Device Expectations

Create serial should appear as one of:

```text
/dev/ttyUSB0
/dev/ttyACM0
/dev/serial/by-id/<adapter>
```

The expected Create baud rate is `57600`.

`robot` and `capture-real` default `--create-port auto`, which uses the first likely serial candidate from `hardware-env`. Pass `--create-port /dev/ttyUSB0` when you want to pin the adapter, `--create-port mock` for `robot` no-hardware smoke tests, or `--mock` for `capture-real` no-hardware smoke tests.

u-blox7 GPS receivers are read over USB serial at 9600 baud using NMEA. Netherwick auto-starts GPS on real runs by preferring `/dev/serial/by-id/*u-blox*`, `/dev/serial/by-id/*gps*`, or `/dev/serial/by-id/*gnss*`, then falling back to an unused `/dev/ttyACM*` device. Use `--gps /dev/serial/by-id/<u-blox-device>` to pin it, or `--gps none` to disable GPS capture.

Kinect availability is detected best-effort through `freenect-*` tools or `pkg-config libfreenect`. A missing Kinect is a warning for capture-first bring-up, not a failure when other streams are useful.

Camera devices are expected under `/dev/video*`. Microphones should be visible to ALSA, for example through `arecord -l`.

MPU-6050 IMUs use I2C bus 1 by default on Raspberry Pi header pins: VCC to 3.3V physical pin 1, GND to pin 6, SDA to GPIO 2 physical pin 3, and SCL to GPIO 3 physical pin 5. Netherwick defaults `robot` and `capture-real` to `/dev/i2c-1` at address `0x68`, so no `--imu` flag is needed for the normal wiring. Use `--imu none` to disable IMU capture, or `--imu /dev/i2c-1@0x69` if AD0 is pulled high.

After wiring, this should show address `68`:

```bash
i2cdetect -y 1
```

## Safe First Commands

Inspect the hardware environment:

```bash
cargo run --bin netherwick -- hardware-env
cargo run --bin netherwick -- hardware-env --json
```

Bring up the default real hardware stack in read-only mode:

```bash
just robot
```

Run a bounded read-only robot smoke:

```bash
cargo run --bin netherwick -- robot \
  --mode read-only \
  --duration-seconds 30 \
  --ledger data/ledger/real/read-only-smoke
```

Record a real capture session:

```bash
cargo run --bin netherwick -- capture-real \
  --duration-seconds 60 \
  --out data/captures/real/rpi5-smoke
```

Inspect the capture:

```bash
cargo run --bin netherwick -- inspect-capture \
  data/captures/real/rpi5-smoke
```

No-hardware smoke test:

```bash
cargo run --bin netherwick -- capture-real \
  --duration-seconds 3 \
  --mock \
  --out data/captures/real/mock-smoke
```

## What Success Looks Like

`hardware-env` reports OS, architecture, CPU, memory, likely serial devices, GPS serial candidates/default, I2C devices, default MPU-6050 bus/pins, cameras, audio inputs, Kinect/libfreenect availability, permissions hints, writable data directories, and whether the host looks like Raspberry Pi hardware.

`capture-real` writes:

```text
manifest.json
frames.jsonl
events.jsonl
assets/rgb/
assets/depth/
assets/audio/
assets/pointcloud/
```

The manifest includes machine info, command args, device availability, present/missing streams, start/end times, warnings, and the reserved asset layout. Compact body/sensor features are embedded in `frames.jsonl`; raw RGB/depth/audio export is reserved by path but not yet written.

`inspect-capture` should show frame count, duration, streams present/missing, first/last timestamps, event count, asset counts, and warnings.

## What Failure Looks Like

Missing camera, mic, Kinect, GPS, or IMU should produce warnings while still capturing any available body or sensor stream.

Create serial open failures are clear errors unless `--mock` or `--create-port mock` is used.

If no useful body or sensor stream can be captured, `capture-real` exits with a clear error instead of writing a misleading success.

Any mode that would command motors is refused. Movement-capable bring-up must be a separate future task with explicit safety gates and tests.
