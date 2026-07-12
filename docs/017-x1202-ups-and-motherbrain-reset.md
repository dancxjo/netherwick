# X1202 UPS and motherbrain reset

The Pi 5 motherbrain reads its X1202 locally. The `pete-ups` agent publishes
normalized JSON at `/run/pete/ups.json`, observes GPIO6 for external-power
loss, and can drive GPIO16 low to enable or high to disable charging. Its
systemd unit requests a graceful OS poweroff after five consecutive critical
samples on battery. This is deliberately separate from the brainstem's hard
reset path.

These defaults are the Geekworm X1202/MAX17040G profile documented for the
board available in July 2026: I2C bus 1 address `0x36`, MAX17040G VCELL/SOC
registers `0x02`/`0x04`, BCM GPIO6 high for external power present, and BCM
GPIO16 low for charging enabled. Before enabling the unit, verify the incoming
board revision with `i2cdetect -y 1`, a meter, and observed GPIO transitions.
Do not substitute values from another X120x revision.

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
uses a nonblocking 100 ms pulse, a 30 second cooldown, and command-ID
deduplication. Request, assertion, completion, and refusal events include the
service session and lease hashes. Use this only for an unresponsive
motherbrain; low UPS battery follows the graceful shutdown path.

Hardware references: [Geekworm X1202](https://wiki.geekworm.com/X1202) and
[X1202 GPIO assignments](https://wiki.geekworm.com/X1202_Hardware).
