# Pete Brainstem

Tiny deterministic firmware for bridging/control of the iRobot Create Open Interface. This crate intentionally contains no planning, behavior selection, LLM logic, mapping, or Pete runtime logic.

The crate name and control layer are chip-neutral. The default hardware backend is `arch::rp2040`, targeting a Raspberry Pi Pico/RP2040 and producing a UF2 image. A `pico-w` backend adds a Raspberry Pi Pico W Wi-Fi AP/status interface while keeping robot-affecting hardware in the safety/runtime lane.

## Configuration

The target robot body is declared in `body.toml`; the microcontroller board mapping is declared in `board.toml`. The firmware build reads both files and generates static constants; the embedded firmware does not parse TOML at runtime and does not allocate.

Current body:

```toml
[body]
name = "irobot-create-open-interface"
kind = "create_oi"
drive = "differential"
```

`body.toml` declares robot capabilities and timings: Create OI UART settings, supported body modes and sensor packets, differential drive, public verbs/events/sensors/outputs/safety features, feedback slots, motion limits, TTL limits, and the default OI mode. To support another robot body, add a new body kind, capability declaration, and driver implementation, then point the runtime at that driver without changing the generic capability renderers.

Current board:

```toml
[board]
name = "raspberry-pi-pico"
arch = "rp2040"
```

`board.toml` owns physical pin assignments for the RP2040 backend and reserves logical roles for later capabilities such as I2C, SPI, PWM, ADC, and emergency stop. This keeps robot-body capabilities separate from the board used to host the brainstem.

The r23 carrier uses separate logical names for the Create power signal and the
level translator enable:

```toml
[pins.power_toggle]
pin = "GP18"
gpio = 18
physical_pin = 24

[pins.txs_oe]
pin = "GP19"
gpio = 19
physical_pin = 25
```

## Wiring

| Signal | Pico GPIO | Pico physical pin | Direction |
| --- | ---: | ---: | --- |
| Create OI UART TX | GP0 | 1 | Pico TX to Create RX |
| Create OI UART RX | GP1 | 2 | Create TX to Pico RX |
| `POWER_TOGGLE` | GP18 | 24 | Pico output through TXS0108E channel 7 to Create DB-25 pin 3 |
| `TXS_OE` | GP19 | 25 | Pico output to TXS0108E OE; external 10 kΩ pull-down |
| Create charging indicator | GP20 | 26 | Create DB-25 pin 13 through TXS0108E channel 8 to Pico input |
| Unused external status output | GP17 | 22 | Pico output role; unconnected on r23 |
| Onboard LED | GP25 | onboard | Pico output |
| Shared sensor I2C SDA | GP2 | 4 | Pico I2C1 SDA to MPU-6050 and optional SSD1306 OLED SDA |
| Shared sensor I2C SCL | GP3 | 5 | Pico I2C1 SCL to MPU-6050 and optional SSD1306 OLED SCL |

UART is `57600 8N1`.

The IMU path is short-horizon inertial telemetry plus local tilt/impact reflex safety. It is not SLAM, global pose, or autonomous recovery; higher-level mapping treats IMU status as one discoverable sensor candidate with explicit freshness, clock, calibration, mounting, confidence, and provenance. Compact and JSON status include uptime at status generation, exact IMU and Create body-packet timestamps, a firmware-local clock epoch, orientation confidence, independent gyro-bias/mounting gates, and orientation source. The MPU path reports roll/pitch but does not advertise gyro-integrated yaw as absolute orientation. The IMU is optional at runtime: when absent, brainstem reports `imu.health = "absent"`, continues operating without inertial reflexes, and periodically probes for a newly attached device.

`board.toml` explicitly declares the fixed MPU X/Y/Z to base forward/left/up mounting. Firmware requires 50 stationary plausible-gravity samples before reporting gyro bias calibrated; the attended gravity-zero operation is a separate calibration and is never relabeled as gyro bias. Local tilt/impact safety consumes raw acquisition independently of Motherbrain selection or clock acceptance. Motherbrain's full handoff and bring-up procedure is documented in [`docs/brainstem-imu-handoff.md`](../../docs/brainstem-imu-handoff.md).

The Pico W backend also supports an optional 0.91-inch 128x32
SSD1306-compatible status OLED on the same I2C1 bus. Connect only VCC, GND, SDA,
and SCL; it assigns no buttons or additional GPIOs. Firmware probes `0x3c` and
then `0x3d`, refreshes display content at no more than 5 Hz, and sends screen
data in small I2C chunks after the scheduled MPU work.

For distance readability, the normal page uses the full panel for two
double-height words: robot state (`BOOT`, `READY`, `RUN`, `STOP`, or `WARN`)
and control authority (`CTRL` while a control lease is active, otherwise
`OPEN`). OI and IMU failures already have full-screen alerts, so they do not
consume small normal-status cells. A corner pixel blinks at 1 Hz as a
display-service liveness cue.

A secondary network page shows the exact `pete-xxxx` SSID, `192.168.4.1`, AP
startup/readiness, and the number of active DHCP leases. `LEASE n` means an
unexpired DHCP lease, not confirmed current Wi-Fi association. The page appears
during Create bring-up and rotates briefly when AP startup or failure needs
attention; a ready AP does not consume normal display time or permanently hide
robot state and battery.

Firmware requests complete Create packet 0 every 750 ms independently of any
client-requested sensor stream, so plausible charge/capacity telemetry remains
fresh during ordinary operation. The large battery page includes a gauge,
percent, lightning glyph, and charging state. Charging keeps this positive page
visible instead of rotating away. OI and packet-0 freshness allow two seconds
for poll and scheduling jitter. Once packet-0 telemetry has been observed,
staleness becomes a full-screen `BATT STALE` alert rather than silently hiding
a previously known low reading; the display still does not substitute a guessed
value when telemetry is missing or invalid.

Faults suspend page rotation. Each safety latch has a specific full-screen icon
and large reason: bump, cliff, wheel drop, E-stop, control heartbeat loss, tilt,
impact, or charging motion lockout. The latter two ambiguous labels are shown
as `CTRL LOST` and `NO DRIVE`. Stale OI, stale or low battery, and offline IMU
have dedicated alerts too. Startup diagnostics explicitly distinguish `WAIT
CREATE`, `POWER OFF`, and `OI NO RX`, while runtime errors name Create
no-response, UART framing, timeout, or invalid-packet failures instead of
showing a generic warning.

The OLED is indicator-only and optional. A missing display, either supported
address failing to acknowledge, initialization failure, or a later write
timeout only causes the display service to retry after five seconds. These
failures do not block boot, control or safety handling, and they do not disable
or suppress MPU probing and sampling.

Do not connect 5V Create TX directly to RP2040 RX. The firmware assumes external level shifting or a divider is present on the Create TX to Pico GP1 line.

The original Create drives Cargo Bay DB-25 pin 13 high at 5V while charging.
On r23, TXS0108E channel 8 level-shifts that signal to GP20, which firmware
configures as an active-high input with a pull-down. Firmware never drives GP20
as an LED output. Status reports both the normalized `charging_indicator` and
the inferred raw `charging_indicator_level`, plus GPIO, physical pin, and
configured polarity. While docked, verify the Create side of the interface
reaches 5V and the GP20 side reaches a valid 3.3V high before treating
`charging_indicator: off` as a firmware polarity result.

On r23, Create DB-25 pin 3 toggles robot power on a low-to-high transition.
Pico GP18 drives that input through TXS0108E channel 7. Firmware first
constructs GP18 as a driven-low output; only after that succeeds does it
construct GP19 high to enable the translator. GP19 stays high in normal
operation. Each explicit power request writes GP18 low, then high for the
configured pulse interval, then low again. Ordinary startup does not request a
power toggle. If firmware panics, it clears GP18 and GP19 before halting so the
translator is disabled and the toggle signal is low.

Because Create pin 3 is a toggle rather than separate on/off control, firmware
re-checks the observed power state when each sleep or wake command reaches the
runtime. Sleep is a no-op when Create is known off, pulses once when known on,
and is refused without a pulse when power is unknown. Wake pulses once when
known off and only probes when known on. When power is unknown, wake probes
first and permits one best-effort pulse after the first probe timeout; the
post-pulse probe cannot trigger another pulse. A failed probe never queues an
automatic power-cycle script. The attended, service-scoped `restart_create`
operation remains the explicit diagnostic path for a full restart.

The board's 10 kΩ OE pull-down keeps the translator disabled during reset and
early boot. It is a hardware backstop, not a substitute for the firmware's
ordered output initialization.

## Architecture

```text
body.toml
board.toml
  -> build.rs generated body constants

src/
  arch/rp2040.rs
  arch/pico_w.rs
  display.rs
  drivers/create_uart.rs
  drivers/create_power.rs
  drivers/leds.rs
  drivers/timers.rs
  body.rs
  capabilities.rs
  events.rs
  commands.rs
  runtime.rs
  main.rs
```

Hardware details stay inside `arch/` and `drivers/`. Body-owned capability facts are declared in `body.toml`, generated by `build.rs`, assembled in `body.rs`, and rendered generically by `capabilities.rs`. The runtime moves small typed commands and events through fixed-capacity `heapless::Deque` queues.

On Pico W, concurrency is split by ownership:

- The safety/runtime lane owns Create UART writes, motor stop, `POWER_TOGGLE`, and robot LEDs.
- The Wi-Fi/HTTP lane owns CYW43, AP setup, bounded local ICMP echo, TCP, UDP, HTTP, and mDNS only.
- Wi-Fi never receives robot GPIO/UART handles and cannot directly move motors or toggle Create power.
- HTTP `/status.json` serializes a copied `BrainstemStatus` snapshot and does not hold shared state while writing TCP responses.
- The runtime is tick-driven: each tick polls UART, enforces drive deadlines, advances at most one active action, and sends Stop on drive timeout or UART gating failure.
- A hardware watchdog feed point is reserved in the safety/runtime tick; it must remain owned by that lane.

### Emergency host transit

The Pico W does not expose a USB Ethernet gadget and does not bridge or NAT
general host traffic. Instead, motherbrain maintains a brainstem-side Wi-Fi
link alongside preferred USB control and registers its live DHCP address as
`motherbrain.pete.internal` through its identified session. A forebrain on the
same deterministic `192.168.4.0/24` AP can then reach motherbrain over local IP.
The registration must match the DHCP client, device, and boot identity and
expires with its lease.

Transit connectivity never changes control ownership. The existing identified
session and atomic control-lease surface decides possession; a host associated
with the AP remains non-controlling until that acquisition succeeds. Emergency
policy permits only discovery, health, role coordination, possession status or
acquisition, and handback. Model weights, experiences, Kinect/media streams,
packages, updates, and direct motion commands remain outside this host-transit
path. General host default routes must not point through the Pico W.

Host state, addressing, allowed service ports, diagnostics, and the full
failure matrix are documented in `docs/026-brainstem-transit-failover.md`.

## Public Surface

The public brainstem surface is body-neutral. The forebrain/motherbrain asks for motion, safety, feedback, telemetry, capabilities, and events; the active body driver maps those requests to Create OI opcodes, packet ids, power toggles, songs, LEDs, dock seeking, UART behavior, and sensor decoding.

The checked source of truth is `verb-classification.toml`; `build.rs` fails if
an advertised verb lacks an exposed classification. The production vocabulary
contains only telemetry, bounded actuator primitives, body-native operations,
explicit services, immutable stop/latch operations, and the final Create dock
opcode. Its actuator surface is:

```text
ping
status
get_capabilities
get_events
cmd_vel
drive_direct
drive_arc
stop
dock
careful_mode
escape_motion
```

Navigation, alignment, timed-motion, scanning, wall-follow, wiggle, and escape
procedures are deterministic motherbrain skills. In particular,
`face_bearing`, `track_bearing`, `hold_heading`, `turn_to_heading`, `turn_by`,
`drive_for`, `arc_for`, `creep_until`, `scan_arc`, `dock_align`, `wall_follow`,
`wiggle_align`, `bump_escape`, and `unstick` are not Brainstem capabilities.
All transports reject those legacy wire verbs as `unsupported`; the simulator
uses the same contract. `set_safety_policy` and `cliff_guard` are likewise
unsupported because a host cannot silently weaken or manufacture physical
safety.

`careful_mode` is the deliberately conspicuous exception for an attended
operator taking direct responsibility for the body during an exceptional
physical intervention. It requires operator-debug authority and is never
invoked by motherbrain recovery. Each request has an explicit 250–15,000 ms
TTL, keeps raw observations visible, and remains bounded by E-stop, authority,
heartbeat, and telemetry freshness. Expiry stops first and re-latches every
live condition.

Ordinary recovery uses `escape_motion`, not CAREFUL. Each request names the
currently acknowledged bump or cliff generation and carries one 250 ms
velocity segment. Brainstem accepts it only inside that hazard's reverse/turn
envelope and only while no wheel-drop, charging, tilt, impact, E-stop, link, or
authority hazard dominates. A new hazard ends the segment immediately.

`get_capabilities` reports the current body contract using the clean names above: body kind, drive type, supported verbs, sensors, outputs, safety features, limits, feedback/song slots, and supported sensor packet range. The facts come from the selected body descriptor, not from the generic renderer. HTTP/WebSocket return JSON; UDP and forebrain UART return a compact single-line representation.

Current outward event kinds:

```text
boot
command_accepted
command_rejected
command_started
command_completed
command_interrupted
command_timed_out
command_renewed
body_power_requested
body_power_changed
body_mode_requested
body_mode_changed
telemetry_received
sensor_frame_decoded
motion_requested
motion_stopped
safety_tripped
safety_cleared
bump_changed
cliff_changed
wheel_drop_latched
wheel_drop_cleared
heartbeat_expired
estop_latched
estop_cleared
error
```

`get_events` reads a fixed-size event log by sequence number. Events are compact records:

```text
seq kind a b c
```

The JSON response includes `oldest_seq`, `next_seq`, and `dropped_before_seq`. The compact UART/UDP response includes the same values as `oldest`, `next`, and `dropped_before`. If `dropped_before_seq`/`dropped_before` is non-zero, the caller asked for history older than the 128-record ring still contains. Responses expose at most 16 records at a time; `next_seq` advances to the next page rather than skipping retained records.

The numeric fields are intentionally small and transport-neutral. Command lifecycle events use `a` for command id. `command_started` is emitted when the runtime actually pops a queued runtime command and begins executing it. `command_completed` is emitted when that runtime step finishes normally, including normal TTL expiry for motion. `command_interrupted` is emitted when Stop/E-stop/new velocity/safety/reflex logic clears an active or accepted pending command. `command_timed_out` is reserved for actual failure/deadline paths such as Create wake response timeout, not ordinary motion TTL completion. An identical `cmd_vel` refresh renews the original streaming command without restarting the motor or transferring lifecycle ownership; `command_renewed` is the refresh command's terminal record, with the refresh id in `a`, owning stream command id in `b`, and refresh sequence in `c`.

Motion events pack wheel speeds or duration, sensor-frame events carry the body packet/frame id plus flags and odometry delta, and error events carry a small error code.

Contact withdrawal is not a `bump_escape` skill. A rising bumper edge invokes a
brainstem-local straight reverse of 80 mm/s for at most 300 ms, after first
stopping the preempted command. Only a fresh contact edge during positive
linear output can trigger it. A held-at-boot bumper, stationary press, or
restored level sample can latch and stop motion but cannot start
authority-independent movement; Home Base contact remains dock handling, and
motherbrain recovery uses acknowledged, generation-bound escape segments. The
local withdrawal runs without a possession lease and survives host/session
loss. Cliff, wheel-drop, charging, tilt/impact, disarm, stop, and e-stop remain
stronger and end it stopped.
`contact_withdrawal_started` records contact side,
repeat count, preempted command id, and the reverse bounds;
`contact_withdrawal_completed` records outcome, any dominating safety condition,
observed odometry displacement, elapsed time, and final stopped state. The
motherbrain possessor consumes those events as a safety preemption of its active
skill; it does not schedule or claim ownership of the reflex. If contact
remains after the reflex stops, motherbrain may submit successive
generation-bound `escape_motion` segments, observing sensors and odometry
between them. No uncommanded CAREFUL interval exists, and no reverse or turn is
reported until it has actually been submitted and observed.

The current internal Rust enums still include Create-specific variants such as `CreateOiMode`, `CreatePacketReceived`, and `CreateSensorPacketDecoded`. Those are driver/runtime implementation details, not the clean public vocabulary. Public capability and event renderers translate them into body-neutral names.

Motherbrain skills consume canonical fresh target/body state and renew 250 ms
`cmd_vel` primitives on a 100 ms cadence. If progress inputs, authority, the
transport, or the skill loop go stale, renewal stops and the Brainstem zeros
the motors. Brainstem reflex and invariant events terminate the owning skill as
`safety-preempted`, never as success.

Odometry integrates decoded Create distance/angle deltas into a planar SE(2)
pose. Packets `0`, `19`, and `20` update odometry: complete packet `0` carries
both distance and angle deltas, packet `19` carries distance, and packet `20`
carries angle. Translation is applied at the midpoint heading for combined
distance/angle packets. Status publishes coherent `x_mm`, `y_mm`, and
`heading_mrad` pose values plus the accumulated signed distance. Other decoded
sensor packets update status and events but do not integrate into odometry.
`reset_odometry` clears the pose and accumulated distance and increments a reset
count. Set/calibrate verbs and body-specific odometry calibration remain future
work.

Create OI power, Open Interface start, continuous Full mode, status-light
animation, watchdog stop, and recovery remain owned by the brainstem runtime
and Create body driver. Pico W starts the RP2040 hardware watchdog with the
body-configured two-second timeout, pauses it while a debugger is attached, and
feeds it only from the runtime safety lane through
`BrainstemHardware::feed_watchdog`. The bare RP2040 backend does not yet enable
the hardware watchdog.

## Current Boundaries

Generic public surface:

- Verbs use body-neutral names: motion, safety, feedback, telemetry, status, capabilities, and events.
- Outward events use body-neutral names and are readable by sequence number.
- Capabilities are body-owned. `body.toml` declares the active body kind, drive model, verbs, events, sensors, outputs, safety features, limits, feedback/song slots, and sensor packet range; `build.rs` generates no-alloc constants; `body.rs` exposes the selected descriptor.

Create-specific by design:

- The Create OI body/driver path owns opcodes, packet ids, Create UART behavior, OI modes, songs, LEDs, dock seeking, power-toggle requests, and Create sensor decoding.
- `/status.json` still exposes some Create diagnostic fields because they are useful during bring-up.
- `power_state`, `song_define`, and `song_play` are generic-ish surface names backed by Create-specific implementation details today.

Known remaining TODOs:

- Move the primitive axle-track conversion constant into `body.toml`/capabilities.
- Promote body descriptors from generated constants into a stronger typed driver contract.
- Add odometry get/set/calibrate verbs and body-specific wheel calibration.
- Enable a real hardware watchdog feed in the bare RP2040 backend once reset timing and bring-up policy are settled.

## Bring-Up Ladder

Start without robot hardware and climb only when the previous rung is boring:

1. Simulator tests: `cargo test -p pete-cockpit`.
2. Local simulator smoke loop: `cargo run -p pete-cockpit --example sim_servo_loop`.
3. UDP smoke test against a powered brainstem: `GET_CAPABILITIES`, `GET_EVENTS`, then a low-TTL `CMD_VEL`.
4. UART smoke test over the forebrain link with the same compact protocol.
5. Robot attached, wheels off floor: request control, heartbeat, short low-speed motion, stop, event cursor check.
6. Robot attached, low-speed motion on the floor with clear space and an operator stop path.
7. Motherbrain/forebrain integration later, after the brainstem contract and safety events remain stable under the local tests.

Phase-B contact-reflex promotion additionally requires recorded HIL evidence.
With wheels initially off the floor, command a low forward velocity, press each
bumper independently, and verify from the ordered event cursor that
`command_interrupted` precedes `contact_withdrawal_started`, the wheels reverse
within one 10 ms runtime cycle, and `contact_withdrawal_completed` reports a
stopped body within 300 ms. Repeat while expiring the control lease; withdrawal
must finish. Then repeat with a cliff or wheel-drop signal introduced during
withdrawal; the stronger safety event must terminate the reverse and the typed
outcome must name that preemption. Floor testing should measure displacement
and run repeated contacts before promotion. Unit/simulator success is not HIL
evidence and is not sufficient to close the reflex issue.

## Build

Install the embedded Rust target once:

```bash
rustup target add thumbv6m-none-eabi
```

From the repo root:

```bash
just brainstem-build
```

The direct Cargo equivalent for the current RP2040 backend is:

```bash
cd crates/pete-brainstem
cargo build --release
```

Build the Pico W AP/status firmware:

```bash
just brainstem-pico-w-build
```

The Pico W backend embeds CYW43 firmware blobs at compile time. They are not kept in version control; the Just target fetches them into `crates/pete-brainstem/firmware/cyw43/` before building. To fetch them without building:

```bash
just brainstem-fetch-cyw43
```

Set `CYW43_FIRMWARE_REF` to fetch from a specific Embassy branch, tag, or commit; it defaults to `main`.

## UF2

Install the converter once:

```bash
cargo install elf2uf2-rs
```

Build the UF2 from the repo root:

```bash
just brainstem-uf2
```

The UF2 is written to:

```text
crates/pete-brainstem/target/thumbv6m-none-eabi/release/pete-brainstem.uf2
```

Build the Pico W UF2:

```bash
just brainstem-pico-w-uf2
```

The Pico W UF2 is written to:

```text
crates/pete-brainstem/target/thumbv6m-none-eabi/release/pete-brainstem-pico-w.uf2
```

To flash, hold the Pico BOOTSEL button while plugging it into USB, then copy the UF2 file to the mounted `RPI-RP2` drive.

On Linux motherbrain hosts, install the repo's BOOTSEL automount once:

```bash
just setup-pico-bootsel
```

After that, a Pico/Pico W in BOOTSEL mounts at `/media/$USER/RPI-RP2` with write permission for the operator user.

For an already-running Pico W on its `pete-xxxx` AP, the repo root also has a one-command Wi-Fi BOOTSEL flash path:

```bash
just flash
```

It builds `brainstem-pico-w-uf2`, posts the BOOTSEL command to `http://192.168.4.1/command`, waits for the `RPI-RP2` drive, mounts it if needed, then copies the UF2. Override `PICO_W_BOOTSEL_URL`, `PICO_W_MOUNT`, or `PICO_W_MOUNT_TIMEOUT_SECS` when needed.

## Pico W Operator Interface

The Pico W backend starts an open AP:

```text
SSID: pete-xxxx, where xxxx is a 4-digit base-36 stable instance id
Device IP: 192.168.4.1
Hostname: pete.local via mDNS announcement
DHCP: offers 192.168.4.2-192.168.4.9/24 with router/DNS set to 192.168.4.1
```

The interface exposes the body-neutral brainstem surface:

- `http://192.168.4.1/` serves the operator interface.
- `http://192.168.4.1/events` streams status and outward events as server-sent events (SSE).
- `http://192.168.4.1/status.json` serves firmware/body/runtime/Create/UART status.
- `POST http://192.168.4.1/command` accepts one low-level command atom.
- `POST http://192.168.4.1/command` with `{"kind":"get_capabilities"}` returns the body capability contract.
- `POST http://192.168.4.1/command` with `{"kind":"get_events","since_seq":0}` returns the outward event log.
- WebSocket control is available at `ws://192.168.4.1:81/control`.
- UDP control is available on port `82` using the same ASCII line format as forebrain UART.
- `http://pete.local/` and `http://pete.local/status.json` may work on clients that support mDNS on the AP network.

Associated AP clients may use `ping -c 4 192.168.4.1` as the first network
bring-up check. The Pico W replies only to valid IPv4 ICMP echo requests
addressed to `192.168.4.1`, preserving the identifier, sequence, and payload.
It silently drops malformed, fragmented, unsupported, unowned, oversized, or
rate-limited traffic; replies are capped at four per second and do not create a
route, bridge, NAT path, or any path to robot hardware.

The status JSON includes firmware name/version plus immutable build identity (`git_commit`, `git_commit_short`, `git_dirty`, `build_timestamp`, profile, target, backend, and `build_id`), body name/kind, uptime, runtime state, body/Create diagnostic state, UART RX health, last UART packet timestamp, current command, command lifecycle ids, event sequence, last error, body state, Wi-Fi state, HTTPS state, HTTP request count, DHCP grant count, ICMP echo request/reply/dropped/rate-limited counters, forebrain UART status, sensor state, battery state, song state, and coherent integrated odometry pose/distance state. `get_capabilities` reports the same identity in JSON and the compact UART/UDP response.

Build identity is derived at compile time. Local builds use the exact Git `HEAD` and mark `git_dirty=true` when tracked changes are present. Cargo watches Git refs, index, and tracked sources so a branch/commit or dirty-state change does not reuse an old identity. Exported-source and CI builds can set `PETE_GIT_COMMIT`, `PETE_GIT_DIRTY` (`true`/`false`), and `PETE_BUILD_TIMESTAMP`; `SOURCE_DATE_EPOCH` is used when no explicit timestamp is supplied. Without Git metadata or an override, the identity visibly reports `unknown`. The timestamp defaults to `unknown`, keeping otherwise identical builds reproducible.

The crate keeps local self-signed certificate material out of version control under:

```text
crates/pete-brainstem/certs/pete-brainstem.local.cert.pem
crates/pete-brainstem/certs/pete-brainstem.local.key.pem
```

Regenerate it with:

```bash
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout crates/pete-brainstem/certs/pete-brainstem.local.key.pem \
  -out crates/pete-brainstem/certs/pete-brainstem.local.cert.pem \
  -days 3650 \
  -subj /CN=pete.local \
  -addext subjectAltName=DNS:pete.local,IP:192.168.4.1
```

Firmware HTTPS is currently reported as `unavailable` in `/status.json`; the HTTP liveness and status paths are intentionally kept small and independent of Create/body responsiveness.

Example command:

```bash
curl -s http://192.168.4.1/command \
  -H 'Content-Type: application/json' \
  -d '{"command_id":42,"kind":"cmd_vel","linear_mm_s":120,"angular_mrad_s":0,"ttl_ms":250,"seq":42}'
```

Capabilities:

```bash
curl -s http://192.168.4.1/command \
  -H 'Content-Type: application/json' \
  -d '{"command_id":43,"kind":"get_capabilities"}'
```

Events since boot:

```bash
curl -s http://192.168.4.1/command \
  -H 'Content-Type: application/json' \
  -d '{"command_id":44,"kind":"get_events","since_seq":0}'
```

UDP probes:

```bash
printf 'GET_CAPABILITIES 1\n' | nc -u -w1 192.168.4.1 82
printf 'GET_EVENTS 2 0\n' | nc -u -w1 192.168.4.1 82
printf 'CMD_VEL 3 80 0 250\n' | nc -u -w1 192.168.4.1 82
```

`cmd_vel`, `drive_direct`, and `drive_arc` must include `ttl_ms` and `seq`; the brainstem owns timing, stop deadlines, body startup, mode changes, watchdog stop, and recovery.

## Host Client Contract

The host-side crate `pete-cockpit` defines the cockpit protocol: the stable surface a pilot uses to operate the brainstem. A pilot may be a human web cockpit, an LLM over UART, a simulator test, or a future learned controller over another transport. The protocol is body-neutral for normal control and keeps transport choice out of pilot logic.

`SimCockpit`, `UdpCockpit`, `UartCockpit`, `HttpCockpit`, and `WebSocketCockpit` all implement the same `Cockpit` trait. UDP on port `82` and direct forebrain UART use the compact line protocol; HTTP and WebSocket use firmware JSON. They are peer transports under the same request/response/event model.

Capabilities are the live cockpit contract, not decoration. Pilots should call `get_capabilities`, build a `CockpitContract`, and validate requests before issuing body verbs. Unsupported verbs fail closed: agents should reject them, and UIs should hide or disable their controls instead of silently no-oping. The contract also carries body limits for motion speed, angular speed, and TTL/timeout ranges; safe pilots clamp or reject motion outside those limits before it reaches the brainstem.

`CockpitContract` checks:

```text
supports(verb)
requires_capability(request)
validate_request(request)
motion and TTL/timeout limits
event vocabulary against CockpitEventKind
local model drift against advertised verbs/events
```

Contract smoke examples:

```bash
cargo run -p pete-cockpit --example contract_check
cargo run -p pete-cockpit --example contract_check -- udp 192.168.4.1:82
cargo run -p pete-cockpit --example contract_check -- http 192.168.4.1:80
cargo run -p pete-cockpit --example contract_check -- ws ws://192.168.4.1:81/control
```

The embedded Pico W browser cockpit is a static implementation of this same logical contract. It fetches capabilities at startup, disables unsupported motion/safety/lights/music/dock/primitives/sensor-streaming controls, and consumes status plus outward events from the `/events` SSE stream. The stream carries event cursor IDs so browser reconnects resume without polling; the cockpit stops or warns on missed event history and safety events. LLMs and motherbrain code should use `pete-cockpit` directly, usually over simulator or UART first; transport choice is a deployment detail, not a different pilot API.

The Rust trait exposes the complete firmware public verb set through both named helper methods and the `CockpitRequest` enum:

```rust
ping()
bootsel() // service/debug, not advertised as a normal body capability
get_status()
get_capabilities()
get_events_since(since_seq)
stop()
estop()
clear_estop()
clear_motion_queue()
cmd_vel(linear_mm_s, angular_mrad_s, ttl_ms)
drive_direct(left_mm_s, right_mm_s, ttl_ms)
drive_arc(velocity_mm_s, radius_mm, ttl_ms)
heartbeat_stop(timeout_ms)
request_sensors(packet_id)
stream_sensors(enabled, packet_id, period_ms)
song_define(id, tones)
song_play(id)
define_chirp(kind, tones)
play_feedback(kind)
power_state(request)
calibrate_turn(angular_mrad_s, duration_ms)
reset_odometry()
dock()
```

Create OI mode and robot-light control are deliberately absent from the public
control surface. They belong to the brainstem supervisor.

Generic outward events are represented by `CockpitEventKind`, matching the public event names from `get_events`: `SafetyTripped`, `HeartbeatExpired`, `EStopLatched`, `MotionRequested`, `SensorFrameDecoded`, command lifecycle events, and the rest of the body-neutral vocabulary.

`EventCursor` tracks `next_seq`. Each poll checks `dropped_before_seq`; if history was missed, it returns `MissedEvents` so the motherbrain can stop instead of driving with a hole in its transcript.

Servo-loop example:

```bash
cargo run -p pete-cockpit --example servo_loop -- 192.168.4.1:82
```

Direct UART servo-loop example:

```bash
cargo run -p pete-cockpit --example uart_servo_loop -- /dev/ttyACM0 115200
```

The UART example also accepts `PETE_BRAINSTEM_UART` and `PETE_BRAINSTEM_BAUD`; baud defaults to `115200`.

The example does:

```text
stream_sensors true 0 250
loop:
  heartbeat_stop 900
  cmd_vel 70 0 300
  get_events_since cursor.next_seq
  stop on safety_tripped, heartbeat_expired, estop_latched, or missed events
stop
```

Manual UDP smoke test:

```bash
printf 'GET_EVENTS 1 0\n' | nc -u -w1 192.168.4.1 82
printf 'CMD_VEL 2 60 0 250\n' | nc -u -w1 192.168.4.1 82
printf 'GET_EVENTS 3 0\n' | nc -u -w1 192.168.4.1 82
printf 'STOP 4\n' | nc -u -w1 192.168.4.1 82
```

## Forebrain UART

On Pico W builds, UART0 GP0/GP1 remains dedicated to the iRobot Create OI link. UART1 GP4/GP5 is the forebrain control lane at 115200 8N1 with one ASCII command per line:

| Signal | Pico GPIO | Direction |
| --- | ---: | --- |
| Forebrain UART TX | GP4 | Pico TX to host RX |
| Forebrain UART RX | GP5 | Host TX to Pico RX |
| Ground | GND | Common reference |

Use 3.3V UART levels and cross TX/RX. Do not connect RS-232 voltage levels directly to the Pico. The host path is usually a USB UART such as `/dev/ttyUSB0`, `/dev/ttyACM0`, or a stable `/dev/serial/by-id/...` symlink.

```text
PING seq
STATUS seq
GET_CAPABILITIES seq
GET_EVENTS seq since_seq
ARM seq
DISARM seq
STOP seq
ESTOP seq
CLEAR_ESTOP seq
CMD_VEL seq linear_mm_s angular_mrad_s ttl_ms
DRIVE_DIRECT seq left_mm_s right_mm_s ttl_ms
DRIVE_ARC seq velocity_mm_s radius_mm ttl_ms
HEARTBEAT_STOP seq timeout_ms
REQUEST_SENSORS seq packet_id
STREAM_SENSORS seq true|false packet_id period_ms
CLEAR_MOTION_QUEUE seq
DEFINE_CHIRP seq ok|error|armed|lost_target|dock_seen|danger note duration_64ths...
PLAY_FEEDBACK seq ok|error|armed|lost_target|dock_seen|danger
POWER_STATE seq wake|sleep|start_oi
CALIBRATE_TURN seq angular_mrad_s duration_ms
RESET_ODOMETRY seq
SONG_PLAY seq id
SONG_DEFINE seq id note duration_64ths...
DOCK seq
SET_LIGHTS seq led_bits color intensity
SET_MODE seq passive|safe|full
BOOTSEL seq
```

`GET_CAPABILITIES` replies with one compact `CAPABILITIES` line. `GET_EVENTS` replies with one compact `EVENTS` line containing records newer than `since_seq`.

`ARM` expands internally to body wake, interface start, and safe mode.
`CMD_VEL` replaces the latest velocity mailbox instead of waiting behind
ordinary commands, and the runtime stops the drive when its `ttl_ms` expires.
`STOP` and `ESTOP` preempt immediately. Parse errors, line timeout, UART errors,
runtime errors, and the estop latch all drive the runtime toward stop.

`/status.json` includes `forebrain_uart` with `rx_bytes`, `rx_lines`, `last_seq`, `last_error`, `link_alive_ms`, and `last_command_age_ms`.

Minimal UART smoke test:

```bash
stty -F /dev/ttyUSB0 115200 cs8 -cstopb -parenb -ixon -ixoff raw
printf 'GET_CAPABILITIES 1\n' > /dev/ttyUSB0
timeout 1 cat /dev/ttyUSB0
printf 'GET_EVENTS 2 0\n' > /dev/ttyUSB0
timeout 1 cat /dev/ttyUSB0
```

Expected replies are single lines beginning with `OK 1 CAPABILITIES` and `OK 2 EVENTS`. For a movement smoke test, only run this with the robot lifted or safely staged:

```bash
printf 'HEARTBEAT_STOP 4 900\nCMD_VEL 5 80 0 250\nSTOP 6\nGET_EVENTS 7 0\n' > /dev/ttyUSB0
timeout 1 cat /dev/ttyUSB0
```

The Pico W onboard LED normally emits a one-blink heartbeat every 15 seconds. Event blink codes interrupt that heartbeat:

| Blinks | Meaning |
| ---: | --- |
| 1 | Boot or Wi-Fi starting |
| 2 | Create power request or AP started |
| 3 | Create power toggled or web services started |
| 4 | HTTP request |
| 5 | OI request or DHCP grant |
| 6 | Create UART packet received |
| 7 | Drive requested/stopped |
| 8 | Error |

Wi-Fi/AP/DHCP/HTTP/mDNS failure does not prevent motor stop, UART timeout handling, power safety, or the error blink pattern. The Wi-Fi lane is not allowed to call robot drivers directly; future operator commands must enter through a bounded command queue consumed by the runtime lane.

## Startup and supervision

On boot, the firmware blinks the onboard Pico LED as soon as RP2040 GPIO is
initialized, starts its safety and command lanes,
starts Open Interface, and requests `Full` mode. Receive-side UART health is
evidence, not a prerequisite for those writes: while RX is missing or the
reported mode is uncertain, the supervisor reasserts `Start` followed by
`Full`; otherwise it refreshes `Full` once per second. Motion remains disabled
until the Create has returned a valid OI-mode response (sensor packet 35).
Transmitting `Full` does not itself change the reported mode. The assertion
does not emit another `body_mode_requested` event when telemetry already
confirms Full, so supervision traffic cannot consume the lifecycle event ring;
it pauses when fresh battery telemetry says the battery is at or below 20% and
is actively charging.

Create 1 has no separate Open Interface undock command. Full mode terminates
charging, and leaving the Home Base requires reversing away from its contacts.
The brainstem polls charging-sources packet 34 independently of caller-requested
sensor streams. When its Home Base bit is present, the first nonzero,
non-docking motion request is held while the body driver backs straight away
from the dock at 200 mm/s for 1.5 seconds (300 mm nominal). The original
body-neutral request
starts afterward. This dock departure is an internal body-driver action rather
than a charging safety latch that callers must clear. While any bounded motor
program is active, Full-mode supervision observes fresh mode telemetry without
re-sending the mode byte because Create 1 zeros wheel output on that write.
Actual mode loss still stops motion before reacquisition. Stop, e-stop,
authority loss, heartbeat expiry, and local safety reflexes still preempt it.
If fresh packet-34 telemetry clears Home Base before departure starts, the
pending reverse is discarded so it cannot execute after Pete is carried or
rolled off the dock.
Create responsiveness also expires after one second without a decoded UART
packet. The cached OI mode is then invalidated, new motion is rejected, and
Full-mode supervision stops any active motor program.

For a non-motion electrical/UART diagnosis, run:

```bash
cargo run -p pete-cockpit --example create_link_debug
```

The probe records raw Create-side TX/RX bytes, last-byte timestamps, validated
packets, and break/parity/framing/overrun counts while testing checksummed OI
stream frames for mode packet 35 at 19200, 57600, and 115200 baud. It restores
57600 afterward. Add `-- --wake`
only when the Create is visibly off; this performs one explicit power-toggle
attempt before the scan. Automatic startup never blindly toggles the Create's
power button.

Once acquisition succeeds, the brainstem plays F-G-B, *fasolsi*, for “prepare /
make ready.” A control-lease transition gets its own short Solresol-style
acknowledgement; runtime errors get a descending warning phrase. While healthy,
POWER breathes between low and high intensity over 3.2 seconds while PLAY and
ADVANCE alternate every 800 ms. A safety latch holds all three red, and runtime
error alternates all three red/off. These lights are brainstem-owned rather than
caller-controlled.

Motor movement remains safety-gated. Timeout, UART framing error, or invalid response sends Stop and enters the repeating three-blink error pattern. All drive commands carry a duration; the tick-driven runtime treats that duration as a deadline and sends Stop when it expires, before power-cycle, before idle, and on errors.

## Porting

Runtime behavior lives in `src/runtime.rs` and depends on the `BrainstemHardware` trait in `src/hardware.rs`. Chip-specific code belongs under `src/arch/`; robot-body-specific code belongs under `src/drivers/`.

To add a new controller, implement `BrainstemHardware` for that board, provide its entry point and linker/build config, and keep pin polarity and voltage assumptions documented in the backend.

To add a new robot body, add a body kind in `body.toml`/`build.rs`, declare its verbs/events/sensors/outputs/safety/features/limits, add a descriptor constructor in `body.rs`, implement a driver that maps the generic runtime surface to that body's hardware protocol, and emit generic outward events from the driver/runtime boundary. The generic capability JSON/compact renderers should not need edits for the new body. A later Orange Pi process can send serialized `BrainstemCommand` values over UART, SPI, or USB into the same command queue.

If a command fails, the firmware sends a stop command and enters a repeating three-blink error pattern.
