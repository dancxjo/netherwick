# Netherwick Brainstem

Tiny deterministic firmware for bridging/control of the iRobot Create Open Interface. This crate intentionally contains no planning, behavior selection, LLM logic, mapping, or Netherwick runtime logic.

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

`body.toml` declares robot capabilities and timings: Create OI UART settings, differential drive, default OI mode, and demo timing. To support another robot body, add a new body kind and driver implementation, then point the runtime at that driver without changing the high-level event loop.

Current board:

```toml
[board]
name = "raspberry-pi-pico"
arch = "rp2040"
```

`board.toml` owns physical pin assignments for the RP2040 backend and reserves logical roles for later capabilities such as I2C, SPI, PWM, ADC, device detect, and emergency stop. This keeps robot-body capabilities separate from the board used to host the brainstem.

BRC is optional and disabled by default for 57600 baud bring-up:

```toml
[pins.create_brc]
enabled = false
pin = "GP19"
gpio = 19
```

## Wiring

| Signal | Pico GPIO | Pico physical pin | Direction |
| --- | ---: | ---: | --- |
| Create OI UART TX | GP0 | 1 | Pico TX to Create RX |
| Create OI UART RX | GP1 | 2 | Create TX to Pico RX |
| Create Power Toggle | GP18 | 24 | Pico output to external power-toggle interface |
| Create BRC | GP19 | 25 | Pico output to Create BRC, optional |
| External status LED | GP20 | 26 | Pico output, optional |
| Onboard LED | GP25 | onboard | Pico output |

UART is `57600 8N1`.

Do not connect 5V Create TX directly to RP2040 RX. The firmware assumes external level shifting or a divider is present on the Create TX to Pico GP1 line.

The Power Toggle and BRC outputs assume external wiring that is electrically safe for both the Pico and the Create. Review polarity and isolation before connecting a real robot.

For initial 57600 baud Open Interface bring-up, wire Power Toggle, UART TX/RX, and GND first. Leave BRC disabled unless the board configuration explicitly enables it.

## Architecture

```text
body.toml
board.toml
  -> build.rs generated body constants

src/
  arch/rp2040.rs
  arch/pico_w.rs
  drivers/create_uart.rs
  drivers/create_power.rs
  drivers/leds.rs
  drivers/timers.rs
  events.rs
  commands.rs
  runtime.rs
  main.rs
```

Hardware details stay inside `arch/` and `drivers/`. The runtime moves small typed commands and events through fixed-capacity `heapless::Deque` queues.

On Pico W, concurrency is split by ownership:

- The safety/runtime lane owns Create UART writes, motor stop, Power Toggle, BRC, and robot LEDs.
- The Wi-Fi/HTTP lane owns CYW43, AP setup, TCP, UDP, HTTP, and mDNS only.
- Wi-Fi never receives robot GPIO/UART handles and cannot directly move motors or toggle Create power.
- HTTP `/status.json` serializes a copied `BrainstemStatus` snapshot and does not hold shared state while writing TCP responses.
- The runtime is tick-driven: each tick polls UART, enforces drive deadlines, advances at most one active action, and sends Stop on drive timeout or UART gating failure.
- A hardware watchdog feed point is reserved in the safety/runtime tick; it must remain owned by that lane.

## Event Vocabulary

Events emitted by the brainstem:

```rust
BrainstemEvent::Boot
BrainstemEvent::TickMs(u32)
BrainstemEvent::CreatePowerOnRequested
BrainstemEvent::CreatePowerOffRequested
BrainstemEvent::CreatePowerToggled
BrainstemEvent::CreateBrcPulseRequested
BrainstemEvent::CreateBrcPulsed
BrainstemEvent::CreateOiStartRequested
BrainstemEvent::CreateOiModeRequested(CreateOiMode)
BrainstemEvent::CreatePacketReceived { packet_id, bytes }
BrainstemEvent::DriveRequested { left_mm_s, right_mm_s, duration_ms }
BrainstemEvent::DriveStopped
BrainstemEvent::Error(BrainstemError)
```

Commands accepted from the forebrain:

```rust
BrainstemCommand::Ping
BrainstemCommand::Arm
BrainstemCommand::Disarm
BrainstemCommand::EStop
BrainstemCommand::ClearEStop
BrainstemCommand::CmdVel { linear_mm_s, angular_mrad_s, ttl_ms, seq }
BrainstemCommand::Stop
BrainstemCommand::Status
BrainstemCommand::SongPlay { id }
BrainstemCommand::Dock
BrainstemCommand::SetLights { pattern }
```

Create OI power, BRC, Open Interface start, Safe mode, watchdog stop, and recovery are owned by the brainstem runtime. They are intentionally not exposed as forebrain commands.

Internal supporting enums:

```rust
CreateOiMode::{Passive, Safe, Full}
BrainstemError::{CreateNoResponse, UartFraming, Timeout, InvalidPacket}
```

These types deliberately know about power, serial, motor commands, sensors, LEDs, and time. They do not know about SLAM, goals, mood, planning, language, or Netherwick behavior.

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
cd crates/netherwick-brainstem
cargo build --release
```

Build the Pico W AP/status firmware:

```bash
just brainstem-pico-w-build
```

The Pico W backend embeds CYW43 firmware blobs at compile time. They are not kept in version control; the Just target fetches them into `crates/netherwick-brainstem/firmware/cyw43/` before building. To fetch them without building:

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
crates/netherwick-brainstem/target/thumbv6m-none-eabi/release/netherwick-brainstem.uf2
```

Build the Pico W UF2:

```bash
just brainstem-pico-w-uf2
```

The Pico W UF2 is written to:

```text
crates/netherwick-brainstem/target/thumbv6m-none-eabi/release/netherwick-brainstem-pico-w.uf2
```

To flash, hold the Pico BOOTSEL button while plugging it into USB, then copy the UF2 file to the mounted `RPI-RP2` drive.

For an already-running Pico W on the `pete-brainstem` AP, the repo root also has a one-command Wi-Fi BOOTSEL flash path:

```bash
just flash
```

It builds `brainstem-pico-w-uf2`, posts the BOOTSEL command to `http://192.168.4.1/command`, waits for the `RPI-RP2` drive, then copies the UF2. Override `PICO_W_BOOTSEL_URL`, `PICO_W_MOUNT`, or `PICO_W_MOUNT_TIMEOUT_SECS` when needed.

## Pico W Operator Interface

The Pico W backend starts an open AP:

```text
SSID: pete-brainstem
Device IP: 192.168.4.1
Hostname: pete.local via mDNS announcement
DHCP: offers 192.168.4.2/24 with router/DNS set to 192.168.4.1
```

The interface is read-only for Brainstem 0:

- `http://192.168.4.1/` serves a plain liveness response: `hello, I'm at least up`.
- `http://192.168.4.1/status.json` serves firmware/body/runtime/Create/UART/demo status.
- `POST http://192.168.4.1/command` accepts one low-level command atom.
- `http://pete.local/` and `http://pete.local/status.json` may work on clients that support mDNS on the AP network.

The status JSON includes firmware name/version, body name/kind, uptime, runtime state, Create power state, OI mode, UART RX health, last UART packet timestamp, current command, last error, demo state, Wi-Fi state, HTTPS state, HTTP request count, DHCP grant count, and last web request timestamp.

The crate keeps local self-signed certificate material out of version control under:

```text
crates/netherwick-brainstem/certs/pete-brainstem.local.cert.pem
crates/netherwick-brainstem/certs/pete-brainstem.local.key.pem
```

Regenerate it with:

```bash
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout crates/netherwick-brainstem/certs/pete-brainstem.local.key.pem \
  -out crates/netherwick-brainstem/certs/pete-brainstem.local.cert.pem \
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

Supported command `kind` values are `ping`, `arm`, `disarm`, `estop`, `clear_estop`, `cmd_vel`, `stop`, `status`, `song_play`, `dock`, and `set_lights`. `cmd_vel` must include `ttl_ms` and `seq`; the brainstem owns timing, stop deadlines, Create startup, BRC, Open Interface start, Safe mode, watchdog stop, and recovery.

## Forebrain UART

On Pico W builds, UART0 GP0/GP1 remains dedicated to the iRobot Create OI link. UART1 GP4/GP5 is the forebrain control lane at 115200 8N1 with one ASCII command per line:

```text
PING seq
ARM seq
DISARM seq
STOP seq
ESTOP seq
CLEAR_ESTOP seq
CMD_VEL seq linear_mm_s angular_mrad_s ttl_ms
STATUS seq
SONG_PLAY seq id
DOCK seq
SET_LIGHTS seq off|status|clean|dock|spot|max
```

`ARM` expands internally to Create wake, BRC pulse, OI start, and Safe mode. `CMD_VEL` replaces the latest velocity mailbox instead of waiting behind ordinary commands, and the runtime stops the drive when its `ttl_ms` expires. `STOP` and `ESTOP` preempt immediately. Parse errors, line timeout, UART errors, runtime errors, and the estop latch all drive the runtime toward StopDrive.

`/status.json` includes `forebrain_uart` with `rx_bytes`, `rx_lines`, `last_seq`, `last_error`, `link_alive_ms`, and `last_command_age_ms`.

The Pico W onboard LED normally emits a one-blink heartbeat every 15 seconds. Event blink codes interrupt that heartbeat:

| Blinks | Meaning |
| ---: | --- |
| 1 | Boot or Wi-Fi starting |
| 2 | Create power request or AP started |
| 3 | Create power toggled or web services started |
| 4 | BRC event or HTTP request |
| 5 | OI request or DHCP grant |
| 6 | Create UART packet received |
| 7 | Drive requested/stopped |
| 8 | Error |

Wi-Fi/AP/DHCP/HTTP/mDNS failure does not prevent motor stop, UART timeout handling, power safety, or the error blink pattern. The Wi-Fi lane is not allowed to call robot drivers directly; future operator commands must enter through a bounded command queue consumed by the runtime lane.

## Demo Behavior

On boot the firmware:

1. Blinks the onboard Pico LED as soon as RP2040 GPIO is initialized, then blinks the onboard LED and optional GP20 LED from the runtime.
2. Enqueues the demo `BrainstemCommand` script.
3. Pulses Create Power Toggle to wake the robot.
4. Polls the Create OI sensor stream until UART bytes confirm the robot is alive.
5. Pulses BRC low and releases it if `gpio.create_brc.enabled = true`.
6. Sends Open Interface `Start`.
7. Enters `Safe` mode.
8. Sends a tiny movement jig: short forward, short left turn, short right turn, stop.
9. Sends Stop, then pulses Create Power Toggle again.
10. Leaves the controller in a safe idle blink loop.

Motor movement is safety-gated: the built-in script only reaches drive commands after UART RX/responses confirm the Create is alive. Timeout, UART framing error, or invalid response skips the jig, sends Stop, and enters the repeating three-blink error pattern.

All drive commands carry a duration. The tick-driven runtime treats that duration as a deadline and sends Stop when it expires, before power-cycle, before idle, and on errors.

## Porting

Runtime behavior lives in `src/runtime.rs` and depends on the `BrainstemHardware` trait in `src/hardware.rs`. Chip-specific code belongs under `src/arch/`; robot-body-specific code belongs under `src/drivers/`.

To add a new controller, implement `BrainstemHardware` for that board, provide its entry point and linker/build config, and keep pin polarity and voltage assumptions documented in the backend.

To add a new robot body, add a body kind in `body.toml`/`build.rs`, implement a driver that consumes `BrainstemCommand` values or a body-specific command subset, and emit `BrainstemEvent` results from the driver. A later Orange Pi process can replace the built-in demo script by sending serialized `BrainstemCommand` values over UART, SPI, or USB into the same command queue.

If a command fails, the firmware sends a stop command and enters a repeating three-blink error pattern.
