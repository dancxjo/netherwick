# Real Robot Read-Only and Possession Bring-Up

Read-only robot mode lets Pete ingest Cockpit status/events and optional sensor data without allowing motion. It is intended for hardware bring-up, ledger collection, capture/replay data, dashboard inspection, and Reign teaching data before any autonomous driving mode exists.

Read-only mode must not drive motors.

## Commands

Simulated Cockpit, no hardware:

```bash
cargo run -p pete-tools -- robot \
  --mode read-only \
  --cockpit sim \
  --ledger data/ledger/robot-readonly \
  --steps 10
```

Default Raspberry Pi hardware:

```bash
just robot
```

Brainstem Cockpit over UART:

```bash
cargo run -p pete-tools -- robot \
  --mode read-only \
  --cockpit uart --create-port /dev/serial/by-id/DEVICE \
  --ledger data/ledger/robot-readonly
```

Capture and dashboard:

```bash
cargo run -p pete-tools -- robot \
  --mode read-only \
  --cockpit sim \
  --capture data/captures/robot-readonly-001 \
  --dashboard 127.0.0.1:3000 \
  --steps 25
```

The dashboard exposes `/now` and `/view` for the latest live snapshot.

## Safety Guarantee

The real robot runner reads cockpit-derived status/sensor state, builds `Now`, runs the normal runtime tick, records the chosen action, and suppresses motion commands. Ledger frames are annotated with:

```text
source = real_robot_read_only
mode = read_only
motor_applied = false
ReadOnlyActionSuppressed
```

The `Now` extension `read_only_motor_gate` records `motor_applied: false`, `final_motor: Stop`, and `safety_reason: ReadOnlyMode`.

## Production possession mode

Possession is the live motherbrain control lease; there is no additional
motherbrain arm layer. This mode is never the default and must be selected
explicitly. The motherbrain receives only the brainstem's body/safety
interface. Create OI access and mode selection remain private to the
brainstem.

With the wheels off the floor:

```bash
just possess
```

The recipe requires `PETE_BRAINSTEM_DEVICE_ID` in `.env`;
`PETE_COCKPIT_PORT` may pin the `/dev/serial/by-id/DEVICE` path. On a boot-ID
mismatch, it accepts the newly observed boot only after the device-ID check has
succeeded, updates `PETE_BRAINSTEM_BOOT_ID` in `.env`, and retries once. Its
Wi-Fi path also performs the required USB identity bootstrap automatically
after a cold-boot identity rejection. Its expanded command is:

```bash
cargo run -p pete-tools -- robot \
  --mode regular \
  --cockpit uart \
  --create-port /dev/serial/by-id/DEVICE \
  --brainstem-device-id BRAINSTEM_ID \
  --brainstem-boot-id BOOT_ID \
  --max-linear-mm-s 50 \
  --max-angular-mrad-s 500 \
  --autonomous-motion \
  --tick-ms 20 \
  --duration-seconds 30 \
  --ledger data/ledger/real/possession-wheels-off-floor \
  --capture data/captures/real/possession-wheels-off-floor
```

The explicit mode selection and `--autonomous-motion` flag authorize executive
actions to reach the physical wheel bridge. Direct WebRemote/Gamepad commands
continue to override the executive. Startup performs
a fresh handshake, verifies identity and the live contract/safety snapshot,
publishes the complete validated brainstem capability contract into robot
initialization context for the motherbrain, acquires the motherbrain lease, and
sends STOP. Runtime motion is limited to
50 mm/s linear and 500 mrad/s angular with a 300 ms command TTL and a 750 ms
heartbeat stop. Missing hardware, an unstable device path, identity mismatch,
or acquisition failure aborts; possession mode never downgrades.

`just possess` targets a 20 ms (50 Hz) control period by default. Tick work is
included in that period instead of being followed by an additional full delay;
if sensing or cognition overruns 20 ms, the next tick starts immediately.
Override the target explicitly with `--tick-ms` or set
`PETE_POSSESSION_TICK_MS`.

The brainstem is the motherbrain's body interface. Every real-robot tick polls
the brainstem's cursor-bounded event stream and publishes it as
`brainstem.events`; the validated interface contract is published as
`brainstem.interface`. A missed event-history window is an error rather than a
silent skip. The underlying Create OI remains private to brainstem firmware.
The requested sensor stream is not treated as proof of perpetual freshness:
status reports the decoded packet counter, packet ID, and device-relative age.
Only complete packet 0 refreshes `BodySense::last_update_ms`; old, missing, or
partial packets therefore reach the normal stale-sensor STOP policy.

Normal exit, SIGINT, and SIGTERM require acknowledged STOP, acknowledged
exorcize, and final status. Exorcize releases motherbrain control but leaves
Create OI power and Full-mode supervision with the brainstem. Final status must
show no active motion; OI `armed` may remain true because Full mode is retained.
Transport loss relies on the
short command, heartbeat, and lease expiries and never triggers a power toggle.

After transport loss, the runner closes its local motor gate and retries the
same stable USB path with exponential backoff (250 ms through 5 seconds by
default). Every attempt performs a new handshake and acquires a new lease; the
old session and lease are never reused. A replacement begins with STOP,
re-requests packet-0 streaming, and remains outside the runner until the packet
counter advances and a complete body packet is no more than 500 ms old. The
configured device and boot IDs must both match. A reboot/boot-ID change stops retrying and requires the operator to
restart with the newly observed `--brainstem-boot-id`, making acceptance
explicit. Override only the bounded timing with
`--reconnect-initial-backoff-ms` and `--reconnect-max-backoff-ms`.

## Physical recovery smoke and pending validation

With the drive wheels physically clear of the floor, the live bumper recovery
path can be exercised through the standard possession entrypoint. Once bumper
telemetry has cleared, the smoke releases the bump latch and any e-stop it
observed from that same bump incident; it deliberately leaves a pre-existing
e-stop latched for the operator:

```bash
just possess --recovery-smoke --wheels-off-floor
```

The command refuses simulated or read-only operation. It requests fresh Create
telemetry, keeps bounded motion active, asks the operator to hold either bumper,
and requires live contact, `SafetyTripped`, `MotionStopped`, bumper release,
`SafetyCleared`, reverse, turn-away, probe, and inspect before final STOP and
exorcize. It is intentionally documented here without being run unattended.

Collect the remaining physical evidence through the interactive QA runner:

```bash
just physical-qa
```

It guides charging interlock, both bumpers, each individual cliff sensor,
wheel drop, heartbeat loss, and transport loss/reconnect. Human observations
remain explicit pass/fail/blocked decisions, while the existing guarded bumper
helper is launched in-process. Every session captures the firmware identity
and writes a reviewable checklist under `data/reports/physical-qa/`. Use
`just physical-qa --plan` to print all setup instructions and acceptance gates
without contacting hardware.

## Hardware Notes

Real hardware uses the brainstem Cockpit UART transport. Create-specific serial details live below the brainstem firmware. On Linux, the user running the command usually needs access to the serial device:

```bash
sudo usermod -aG dialout "$USER"
```

Log out and back in after changing group membership. A missing cockpit UART device or permission error fails clearly.

## Optional Sensors

The CLI accepts `--camera`, `--mic`, `--asr-command`, `--imu`, `--imu-source`, `--gps`, and `--lidar`. Camera and microphone are optional by default, so absent devices do not block read-only cockpit bring-up. Brainstem IMU telemetry is discovered and selected automatically; the runtime does not assume `/dev/i2c-1` exists. A local I2C IMU registers only after hardware discovery or an explicit `--imu /dev/i2c-N` setting. `--imu-source none` disables fusion IMU while brainstem safety remains active. GPS auto-starts on real runs when Pete finds a likely u-blox/GPS USB serial device; pass `--gps none` to disable it. HLS-LFCD2 / LDS-01 lidar auto-starts when its stable serial name is recognizable; otherwise pin it with `--lidar /dev/serial/by-id/DEVICE` or `LIDAR_SERIAL_PORT`. Passing `--require-camera`, `--require-mic`, `--require-imu`, `--require-gps`, or `--require-lidar` makes an explicitly configured provider fail if unavailable. See [Brainstem IMU handoff](brainstem-imu-handoff.md) for trust, timing, mounting, and verification details.

`--asr-command` enables the command-backed ASR tool for microphone input. Pete chunks voiced PCM, writes each finalized chunk to a temporary WAV file, appends that path to the configured command, and reads the transcript from stdout. The same value can be supplied with `PETE_ASR_COMMAND`. ASR output is delivered as `EarSense.asr` and follows the normal sensation/vector path.

Current minimum support is robust no-data handling plus simulated/brainstem cockpit capture. Rich camera, microphone, IMU, and GPS producers can be wired behind hardware features without changing the read-only runner.

MPU-6050 IMUs are supported on Linux I2C buses when `pete-tools` is built with the existing `linux-hardware` sensor feature. On a Raspberry Pi, wire VCC to 3.3V physical pin 1, GND to pin 6, SDA to GPIO 2 physical pin 3, and SCL to GPIO 3 physical pin 5. Enable I2C, add the user to the `i2c` group, and reboot:

```bash
sudo raspi-config nonint do_i2c 0
sudo usermod -aG i2c "$USER"
sudo reboot
```

The normal robot reads the MPU-6050 through brainstem telemetry, so this needs no local I2C device:

```bash
cargo run -p pete-tools -- robot
```

For a separately wired diagnostic/future Motherbrain-local MPU only, include the device and address explicitly:

```bash
cargo run -p pete-tools -- robot --imu /dev/i2c-1@0x69
```

u-blox7 GPS receivers are read over USB serial at 9600 baud using NMEA. Auto-detection prefers stable `/dev/serial/by-id/*u-blox*`, `*gps*`, or `*gnss*` paths and avoids the selected Create serial port. Pin a receiver explicitly when needed:

```bash
cargo run -p pete-tools -- robot --gps /dev/serial/by-id/<u-blox-device>
```

HLS-LFCD2 / ROBOTIS LDS-01 produces native 360-degree scans at 230400 baud. Pete sends the legacy start command and decodes the 42-byte segments into 360 `RangeSense` beams. Pin and require it during bring-up with:

```bash
cargo run -p pete-tools -- robot \
  --lidar /dev/serial/by-id/<usb2lds-device> \
  --require-lidar
```

Use `--lidar-height-m`, `--lidar-forward-m`, `--lidar-left-m`, and the `--lidar-{roll,pitch,yaw}-deg` options to describe the mount. Positive pitch tilts the forward scan toward the floor. The same names in uppercase (`LIDAR_HEIGHT_M`, `LIDAR_PITCH_DEG`, and so on) work through `.env` and `just robot`.

Lidar endpoints and Kinect depth points enter the same odometry-aligned voxel cloud. A forward trajectory or slow in-place spin therefore accumulates the tilted scan plane into a 3D world cloud. Repeated cached scans are deduplicated so one physical scan is not projected through several later poses, and floor-height lidar returns are kept out of the 2D obstacle map.

To capture and export the combined cloud:

```bash
cargo run -p pete-tools -- capture-real \
  --duration-seconds 20 \
  --lidar /dev/serial/by-id/<usb2lds-device> \
  --export-pointcloud \
  --out data/captures/real/lidar-spin
```

This writes `assets/pointcloud/world-accumulated.ply`. Rotate slowly: the LFCD2 stream and body odometry are associated per completed sweep, so rapid motion produces scan distortion.

## Capture Output

Passing `--capture <path>` creates a Worldlab capture with:

```text
manifest.json
frames.jsonl
```

The capture source is `RealRobot` and can be replayed with:

```bash
cargo run -p pete-tools -- replay-capture \
  --capture data/captures/robot-readonly-001 \
  --ledger data/ledger/replay-robot-readonly
```

## What It Does Not Do

Read-only mode never requests authority or emits motor mutations. Possession
mode does not expose Create OI, toggle body power during recovery, weaken
brainstem safety, or enable unattended roaming.
