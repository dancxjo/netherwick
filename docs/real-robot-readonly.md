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

The CLI accepts `--camera`, `--mic`, `--imu`, and `--gps`. These are optional by default, so absent devices do not block read-only body bring-up. Passing `--require-camera`, `--require-mic`, `--require-imu`, or `--require-gps` makes the command fail if that provider is unavailable.

Current minimum support is robust no-data handling plus mock/body capture. Rich camera, microphone, IMU, and GPS producers can be wired behind hardware features without changing the read-only runner.

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
