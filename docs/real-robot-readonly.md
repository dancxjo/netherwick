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

## Production possession/slow mode

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
  --mode possession-slow \
  --cockpit uart \
  --create-port /dev/serial/by-id/DEVICE \
  --brainstem-device-id BRAINSTEM_ID \
  --brainstem-boot-id BOOT_ID \
  --max-linear-mm-s 50 \
  --max-angular-mrad-s 500 \
  --autonomous-motion \
  --tick-ms 100 \
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

The brainstem is the motherbrain's body interface. Every real-robot tick polls
the brainstem's cursor-bounded event stream and publishes it as
`brainstem.events`; the validated interface contract is published as
`brainstem.interface`. A missed event-history window is an error rather than a
silent skip. The underlying Create OI remains private to brainstem firmware.

Normal exit, SIGINT, and SIGTERM require acknowledged STOP, acknowledged exorcize (translated to
the brainstem's DISARM wire command), and final status
showing no active motion and `armed == false`. Transport loss relies on the
short command, heartbeat, and lease expiries and never triggers a power toggle.

After transport loss, the runner closes its local motor gate and retries the
same stable USB path with exponential backoff (250 ms through 5 seconds by
default). Every attempt performs a new handshake and acquires a new lease; the
old session and lease are never reused. The configured device and boot IDs must
both match. A reboot/boot-ID change stops retrying and requires the operator to
restart with the newly observed `--brainstem-boot-id`, making acceptance
explicit. Override only the bounded timing with
`--reconnect-initial-backoff-ms` and `--reconnect-max-backoff-ms`.

## Hardware Notes

Real hardware uses the brainstem Cockpit UART transport. Create-specific serial details live below the brainstem firmware. On Linux, the user running the command usually needs access to the serial device:

```bash
sudo usermod -aG dialout "$USER"
```

Log out and back in after changing group membership. A missing cockpit UART device or permission error fails clearly.

## Optional Sensors

The CLI accepts `--camera`, `--mic`, `--asr-command`, `--imu`, and `--gps`. Camera and microphone are optional by default, so absent devices do not block read-only cockpit bring-up. IMU defaults to the Raspberry Pi bus `/dev/i2c-1`; pass `--imu none` to disable it. GPS auto-starts on real runs when Pete finds a likely u-blox/GPS USB serial device; pass `--gps none` to disable it. Passing `--require-camera`, `--require-mic`, `--require-imu`, or `--require-gps` makes the command fail if that provider is unavailable.

`--asr-command` enables the command-backed ASR tool for microphone input. Pete chunks voiced PCM, writes each finalized chunk to a temporary WAV file, appends that path to the configured command, and reads the transcript from stdout. The same value can be supplied with `PETE_ASR_COMMAND`. ASR output is delivered as `EarSense.asr` and follows the normal sensation/vector path.

Current minimum support is robust no-data handling plus simulated/brainstem cockpit capture. Rich camera, microphone, IMU, and GPS producers can be wired behind hardware features without changing the read-only runner.

MPU-6050 IMUs are supported on Linux I2C buses when `pete-tools` is built with the existing `linux-hardware` sensor feature. On a Raspberry Pi, wire VCC to 3.3V physical pin 1, GND to pin 6, SDA to GPIO 2 physical pin 3, and SCL to GPIO 3 physical pin 5. Enable I2C, add the user to the `i2c` group, and reboot:

```bash
sudo raspi-config nonint do_i2c 0
sudo usermod -aG i2c "$USER"
sudo reboot
```

The default bus is `/dev/i2c-1` and the default MPU-6050 address is `0x68`, so this reads real IMU data with normal Pi wiring:

```bash
cargo run -p pete-tools -- robot
```

If AD0 is high, include the address in the device string:

```bash
cargo run -p pete-tools -- robot --imu /dev/i2c-1@0x69
```

u-blox7 GPS receivers are read over USB serial at 9600 baud using NMEA. Auto-detection prefers stable `/dev/serial/by-id/*u-blox*`, `*gps*`, or `*gnss*` paths and avoids the selected Create serial port. Pin a receiver explicitly when needed:

```bash
cargo run -p pete-tools -- robot --gps /dev/serial/by-id/<u-blox-device>
```

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
