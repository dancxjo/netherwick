# RPi 5 direct Create brainstem

The RPi 5 can run the same Brainstem safety/runtime lane as the Pico W while
talking directly to the Create 1 external Mini-DIN Open Interface port. This is
an alternative board backend, not a Motherbrain bypass:

```text
Motherbrain possession process
        |
        | Cockpit compact protocol on 127.0.0.1:8787/UDP
        v
pete-brainstem process on the same RPi 5
        |
        | 57,600 baud Create Open Interface
        v
level-shifting USB cable -> Create 1 side Mini-DIN port
```

Keep the processes separate. If possession exits, stalls, or restarts, the
Brainstem process continues enforcing command TTLs, heartbeat expiry, sensor
latches, contact withdrawal, Full-mode freshness, and STOP. Loopback transport
does not grant authority; the normal session and possession lease still apply.

That process boundary is not an independent watchdog. Brainstem and
Motherbrain still share the RPi 5 kernel, scheduler, power, and hardware. A
whole-Pi freeze after a nonzero Create drive command can prevent both the next
renewal and the STOP that would normally end it; Create OI can retain that last
drive command. The direct-RPi contract therefore advertises
`independent_watchdog=false` and is a reduced safety class compared with the
Pico W brainstem, whose hardware watchdog continues to supervise the body when
Motherbrain or its host fails.

## Cable and electrical boundary

Use the iRobot Create serial cable, or an equivalent cable that explicitly
shifts the Create serial levels to USB. The Create 1 Mini-DIN TXD and RXD pins
are 0-5 V TTL, not PC RS-232 and not a promise that an arbitrary 3.3 V USB-UART
adapter is safe. The connector also exposes unregulated battery voltage; do not
connect its Vpwr pins to the RPi 5.

The standard serial cable gives this backend serial TX, RX, and ground. It does
not give the isolated power-toggle circuit, charging-indicator GPIO, or
external IMU present on the Pico board. The RPi 5 capability response therefore
omits those features. Turn the Create on manually before starting the service.

## Configure and run

Find the stable path for the Create cable:

```sh
ls -l /dev/serial/by-id
```

Put the direct Create device in `.env` for an interactive run:

```sh
PETE_CREATE_PORT=/dev/serial/by-id/usb-YOUR_CREATE_CABLE
PETE_CREATE_BAUD=57600
PETE_BRAINSTEM_LOCAL_ADDR=127.0.0.1:8787
```

Build and start Brainstem in one terminal:

```sh
just brainstem-rpi5-build
just brainstem-rpi5
```

In another terminal, verify the unchanged Cockpit contract and then possess:

```sh
just brainstem-rpi5-check
just possess-rpi5
```

After body-only possession is healthy, the strict higher-sensor wheels-up path
is:

```sh
just possess-sensorium-rpi5 \
  --require-kinect --require-lidar --require-imu --require-llm
```

It retains the same local Brainstem safety boundary while requiring first data
from each named stream. Optional camera, microphone, and GPS inputs can be made
strict with their corresponding `--require-*` flags.

The direct-RPi recipes declare `--wheels-off-floor` by default. Autonomous
motion on the physical floor is refused unless the operator replaces that
declaration with the explicit residual-risk acknowledgement:

```sh
just possess-rpi5 --acknowledge-no-independent-watchdog
```

Use that floor-motion form only with an independent, immediately reachable
external hard stop that removes drive power, such as a latching battery
disconnect installed for the robot. Do not count the RPi process supervisor,
network access, keyboard interrupt, or Create OI STOP command as independent:
all depend on software or hardware implicated in the residual failure mode.
The launcher prints the reduced class, the live dashboard session reports
`safety_class=reduced-shared-host` and `independent_watchdog=false`, and real
capture manifests record the same classification plus the selected motion
surface and acknowledgement state.

That command is deliberately body-only. After it succeeds, exercise configured
higher senses through the same local Brainstem and possession safeguards with:

```bash
just possess-sensorium-rpi5
```

See `docs/real-robot-readonly.md` for sensor selection, required-sensor flags,
and the evidence boundary of a successful sensorium run.

The first local possession reads the RPi-derived Brainstem identity and current
Linux boot identity, then pins them in `.env`. A different RPi 5 identity still
requires the existing explicit replacement acceptance.

## Install as an independent service

Install the release binary, unit, and root-owned environment file:

```sh
just brainstem-rpi5-build
sudo install -m 0755 crates/pete-brainstem/target/release/pete-brainstem \
  /usr/local/bin/pete-brainstem
sudo install -m 0644 configs/systemd/netherwick-brainstem-rpi5@.service \
  /etc/systemd/system/netherwick-brainstem-rpi5@.service
sudo install -d -m 0755 /etc/netherwick
sudoedit /etc/netherwick/brainstem-rpi5.env
```

The environment file must contain at least:

```sh
PETE_CREATE_PORT=/dev/serial/by-id/usb-YOUR_CREATE_CABLE
PETE_CREATE_BAUD=57600
PETE_BRAINSTEM_LISTEN=127.0.0.1:8787
```

Enable it for the operator account:

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now netherwick-brainstem-rpi5@"$USER".service
systemctl status netherwick-brainstem-rpi5@"$USER".service
```

The account must be able to open the cable. The supplied unit adds the
`dialout` supplementary group; distributions using a different serial-device
group need the corresponding local adjustment.

## Acceptance gates

With the wheels lifted clear of the floor:

1. `just brainstem-rpi5-check` reports the Create body contract without
   power-toggle or IMU capabilities and with `independent_watchdog=false`.
2. Status advances `create_rx_packets` and `create_body_packets`, and OI mode
   reaches `full`.
3. A 125 ms velocity pulse stops at its TTL without a possessor STOP.
4. Suspending only the possession process causes heartbeat STOP while the
   Brainstem service remains running.
5. Bump, cliff, and wheel-drop inputs stop motion and emit the same typed events
   used by the Pico build.
6. Stopping the systemd service while wheels are moving sends STOP before the
   serial device closes.

Software tests can prove the shared runtime and protocol path, but steps 2-6
remain physical hardware-in-loop evidence.
