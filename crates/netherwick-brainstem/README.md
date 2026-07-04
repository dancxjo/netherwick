# Netherwick Brainstem

Tiny deterministic firmware for bridging/control of the iRobot Create Open Interface. This crate intentionally contains no planning, behavior selection, LLM logic, mapping, or Netherwick runtime logic.

The crate name and control layer are chip-neutral. The current hardware backend is `arch::rp2040`, targeting a Raspberry Pi Pico/RP2040 and producing a UF2 image.

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
2. Pulses Create Power Toggle to wake the robot.
3. Polls the Create OI sensor stream until UART bytes are received.
4. Pulses BRC low and releases it.
5. Sends Open Interface `Start`.
6. Enters `Safe` mode by default.
7. Sends a tiny movement jig: short forward, short left turn, short right turn, stop.
8. Pulses Create Power Toggle again.
9. Leaves the controller in a safe idle blink loop.

## Porting

Create behavior lives in `src/create.rs` and depends only on the `BrainstemHardware` trait in `src/hardware.rs`. Chip-specific code belongs under `src/arch/`.

To add a new controller, implement `BrainstemHardware` for that board, provide its entry point and linker/build config, and keep Create pin polarity and voltage assumptions documented in the backend.

If the Create does not respond over UART, the firmware sends a stop command and enters a repeating three-blink error pattern.
