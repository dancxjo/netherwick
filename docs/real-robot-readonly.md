# Real Robot Read-Only Bring-Up

Read-only robot mode lets Netherwick ingest real Create 1 body state and optional sensor data without allowing motion. It is intended for hardware bring-up, ledger collection, capture/replay data, dashboard inspection, and Reign teaching data before any autonomous driving mode exists.

Read-only mode must not drive motors.

## Commands

Mock body, no hardware:

```bash
cargo run -p netherwick-tools -- robot \
  --mode read-only \
  --create-port mock \
  --ledger data/ledger/robot-readonly \
  --steps 10
```

Default Raspberry Pi hardware:

```bash
just robot
```

Create 1 body over serial:

```bash
cargo run -p netherwick-tools -- robot \
  --mode read-only \
  --create-port /dev/ttyUSB0 \
  --ledger data/ledger/robot-readonly
```

Capture and dashboard:

```bash
cargo run -p netherwick-tools -- robot \
  --mode read-only \
  --create-port mock \
  --capture data/captures/robot-readonly-001 \
  --dashboard 127.0.0.1:3000 \
  --steps 25
```

The dashboard exposes `/now` and `/view` for the latest live snapshot.

## Safety Guarantee

The real robot runner reads body/sensor state, builds `Now`, runs the normal runtime tick, records the chosen action, and suppresses motor application. Ledger frames are annotated with:

```text
source = real_robot_read_only
mode = read_only
motor_applied = false
ReadOnlyActionSuppressed
```

The `Now` extension `read_only_motor_gate` records `motor_applied: false`, `final_motor: Stop`, and `safety_reason: ReadOnlyMode`.

## Hardware Notes

The Create 1 serial path uses the `netherwick-create1` serial feature from the tools binary. On Linux, the user running the command usually needs access to the serial device:

```bash
sudo usermod -aG dialout "$USER"
```

Log out and back in after changing group membership. A missing device or permission error fails clearly.

## Optional Sensors

The CLI accepts `--camera`, `--mic`, `--imu`, and `--gps`. Camera and microphone are optional by default, so absent devices do not block read-only body bring-up. IMU defaults to the Raspberry Pi bus `/dev/i2c-1`; pass `--imu none` to disable it. GPS auto-starts on real runs when Netherwick finds a likely u-blox/GPS USB serial device; pass `--gps none` to disable it. Passing `--require-camera`, `--require-mic`, `--require-imu`, or `--require-gps` makes the command fail if that provider is unavailable.

Current minimum support is robust no-data handling plus mock/body capture. Rich camera, microphone, IMU, and GPS producers can be wired behind hardware features without changing the read-only runner.

MPU-6050 IMUs are supported on Linux I2C buses when `netherwick-tools` is built with the existing `linux-hardware` sensor feature. On a Raspberry Pi, wire VCC to 3.3V physical pin 1, GND to pin 6, SDA to GPIO 2 physical pin 3, and SCL to GPIO 3 physical pin 5. Enable I2C, add the user to the `i2c` group, and reboot:

```bash
sudo raspi-config nonint do_i2c 0
sudo usermod -aG i2c "$USER"
sudo reboot
```

The default bus is `/dev/i2c-1` and the default MPU-6050 address is `0x68`, so this reads real IMU data with normal Pi wiring:

```bash
cargo run -p netherwick-tools -- robot
```

If AD0 is high, include the address in the device string:

```bash
cargo run -p netherwick-tools -- robot --imu /dev/i2c-1@0x69
```

u-blox7 GPS receivers are read over USB serial at 9600 baud using NMEA. Auto-detection prefers stable `/dev/serial/by-id/*u-blox*`, `*gps*`, or `*gnss*` paths and avoids the selected Create serial port. Pin a receiver explicitly when needed:

```bash
cargo run -p netherwick-tools -- robot --gps /dev/serial/by-id/<u-blox-device>
```

## Capture Output

Passing `--capture <path>` creates a Worldlab capture with:

```text
manifest.json
frames.jsonl
```

The capture source is `RealRobot` and can be replayed with:

```bash
cargo run -p netherwick-tools -- replay-capture \
  --capture data/captures/robot-readonly-001 \
  --ledger data/ledger/replay-robot-readonly
```

## What It Does Not Do

Read-only mode does not implement autonomous real robot motion, slow driving, docking, Kinect/libfreenect integration, online model promotion, full ASR/TTS, or face recognition training.

The next movement-capable milestone is a separate slow mode with explicit motor gating and hardware acceptance tests.
