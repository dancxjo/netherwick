# Netherwick Brainstem

Tiny deterministic firmware for bridging/control of the iRobot Create Open Interface. This crate intentionally contains no planning, behavior selection, LLM logic, mapping, or Netherwick runtime logic.

The crate name and control layer are chip-neutral. The current hardware backend is `arch::rp2040`, targeting a Raspberry Pi Pico/RP2040 and producing a UF2 image.

## Body Config

The target robot body is declared in `body.toml`. The firmware build reads that file and generates static constants; the embedded firmware does not parse TOML at runtime and does not allocate.

Current body:

```toml
[body]
name = "irobot-create-open-interface"
kind = "create_oi"
drive = "differential"
```

The file also declares Create OI UART settings, timing values, and the current RP2040/Pico pin map. To support another robot body, add a new body kind and driver implementation, then point the runtime at that driver without changing the high-level event loop.

## Wiring

| Signal | Pico GPIO | Pico physical pin | Direction |
| --- | ---: | ---: | --- |
| Create OI UART TX | GP16 | 21 | Pico TX to Create RX |
| Create OI UART RX | GP17 | 22 | Create TX to Pico RX |
| Create Power Toggle | GP18 | 24 | Pico output to external power-toggle interface |
| Create BRC | GP19 | 25 | Pico output to Create BRC |
| External status LED | GP20 | 26 | Pico output, optional |
| Onboard LED | GP25 | onboard | Pico output |

UART is `57600 8N1`.

Do not connect 5V Create TX directly to RP2040 RX. The firmware assumes external level shifting or a divider is present on the Create TX to Pico GP17 line.

The Power Toggle and BRC outputs assume external wiring that is electrically safe for both the Pico and the Create. Review polarity and isolation before connecting a real robot.

## Architecture

```text
body.toml
  -> build.rs generated body constants

src/
  arch/rp2040.rs
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

Commands accepted by the runtime:

```rust
BrainstemCommand::WakeCreate
BrainstemCommand::SleepCreate
BrainstemCommand::PulseBrc
BrainstemCommand::StartOi
BrainstemCommand::SetOiMode(CreateOiMode)
BrainstemCommand::Drive { left_mm_s, right_mm_s, duration_ms }
BrainstemCommand::StopDrive
```

Supporting enums:

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

To flash, hold the Pico BOOTSEL button while plugging it into USB, then copy the UF2 file to the mounted `RPI-RP2` drive.

## Demo Behavior

On boot the firmware:

1. Blinks the onboard LED and optional GP20 LED.
2. Enqueues the demo `BrainstemCommand` script.
3. Pulses Create Power Toggle to wake the robot.
4. Polls the Create OI sensor stream until UART bytes confirm the robot is alive.
5. Pulses BRC low and releases it.
6. Sends Open Interface `Start`.
7. Enters `Safe` mode.
8. Sends a tiny movement jig: short forward, short left turn, short right turn, stop.
9. Sends Stop, then pulses Create Power Toggle again.
10. Leaves the controller in a safe idle blink loop.

Motor movement is safety-gated: the built-in script only reaches drive commands after UART RX/responses confirm the Create is alive. Timeout, UART framing error, or invalid response skips the jig, sends Stop, and enters the repeating three-blink error pattern.

## Porting

Runtime behavior lives in `src/runtime.rs` and depends on the `BrainstemHardware` trait in `src/hardware.rs`. Chip-specific code belongs under `src/arch/`; robot-body-specific code belongs under `src/drivers/`.

To add a new controller, implement `BrainstemHardware` for that board, provide its entry point and linker/build config, and keep pin polarity and voltage assumptions documented in the backend.

To add a new robot body, add a body kind in `body.toml`/`build.rs`, implement a driver that consumes `BrainstemCommand` values or a body-specific command subset, and emit `BrainstemEvent` results from the driver. A later Orange Pi process can replace the built-in demo script by sending serialized `BrainstemCommand` values over UART, SPI, or USB into the same command queue.

If a command fails, the firmware sends a stop command and enters a repeating three-blink error pattern.
