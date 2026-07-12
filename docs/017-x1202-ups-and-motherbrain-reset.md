# X1202 UPS and motherbrain reset

The Pi 5 motherbrain reads its X1202 locally. The long-running `pete-ups`
monitor publishes normalized JSON at `/run/pete/ups.json`, owns the configured
external-power and charging GPIO line handles for its entire lifetime, and
reports its commanded charging state. `pete-ups charge enable|disable` sends a
request to `/run/pete/ups-control.sock`; it never requests GPIO16 directly.
The systemd unit requests a graceful OS poweroff after five consecutive
critical samples on battery and treats a failed `systemctl poweroff` as a
service failure so systemd restarts monitoring. This is deliberately separate
from the brainstem's hard reset path.

Install `configs/pete-ups-x1202.toml` as `/etc/pete/x1202.toml`. It explicitly
configures the I2C device/address, gpiochip, line offsets, polarities, and
startup charging state. Its shipped values describe the Geekworm
X1202/MAX17040G profile documented for the board available in July 2026: I2C
bus 1 address `0x36`, MAX17040G VCELL/SOC registers `0x02`/`0x04`, GPIO6 high
for external power present, and GPIO16 low for charging enabled. On the Pi 5,
gpiochip enumeration can change with the kernel and overlays. Before enabling
the unit, identify both lines with `gpioinfo`, verify the incoming board
revision with `i2cdetect -y 1`, a meter, and observed GPIO transitions, and
update the profile. Do not substitute values from another X120x revision.

The brainstem reset output remains disabled in `board.toml` until its circuit
is inspected. The approved circuit is Pico GP21 -> gate/base resistor ->
open-drain/open-collector transistor, with drain/collector on Pi 5 RUN,
source/emitter on shared ground, and a pull-down keeping the transistor off
while the Pico pin floats. The Pico must never source voltage into RUN. Verify
inactive boot and unpowered-Pico behavior before changing `enabled` to true and
building with `motherbrain-reset`.
The Pico W panic handler also clears the GP21 output latch before halting; the
external gate pull-down covers reset, reconfiguration, and an unpowered Pico.

`reset_motherbrain` requires a matching service lease. Firmware additionally
requires the Create body to be stopped and in passive (disarmed) OI mode. It
uses a nonblocking 100 ms pulse and a 30 second cooldown. Reset replay identity
is `(service_session_hash, service_lease_hash, command_id)`; recent outcomes
are retained so retransmission replays the original refusal/assertion/completion
without producing another pulse. Command ID zero is invalid. Request,
assertion, completion, and refusal events include the service session and lease
hashes. The active pulse retains its own immutable identity. `build.rs` rejects
enabled GPIO collisions, including the dormant GP21 device-detect/reset
collision. Use this only for an unresponsive motherbrain; low UPS battery
follows the graceful shutdown path.

Hardware references: [Geekworm X1202](https://wiki.geekworm.com/X1202) and
[X1202 GPIO assignments](https://wiki.geekworm.com/X1202_Hardware).
