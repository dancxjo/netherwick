# Physical motherbrain bootstrap

The Pico W USB connection is USB CDC over a serial byte stream. The host uses
the shared `UartCockpit` line-protocol implementation for both USB CDC device
nodes and GPIO UART; USB discovery always uses the stable `/dev/serial/by-id`
symlink and never assumes `/dev/ttyACM0`.

The production motherbrain can use that same Cockpit contract over the Pico W
HTTP service. Once USB has established the motherbrain identity, disconnect the
USB data cable and run:

```sh
PETE_COCKPIT_BACKEND=wifi \
PETE_BRAINSTEM_HTTP_HOST=192.168.4.1:80 \
just possess
```

`just possess` now defaults to Wi-Fi; set `PETE_COCKPIT_BACKEND=uart` when a
wired recovery session is intentional. Wi-Fi possession retains the pinned
`PETE_BRAINSTEM_DEVICE_ID` check, automatically records a changed boot ID for
that device in `.env`, and reconnects through the selected transport. The
firmware deliberately requires one wired identity establishment after a
brainstem cold boot because its AP is open and the current identity continuity
mechanism is not cryptographic authentication. When the pinned USB brainstem is
connected, `just possess` performs that wired identity bootstrap automatically.
If `.env` points at a stale `/dev/serial/by-id` symlink and exactly one Pete
brainstem is connected, `just possess` reports the detected replacement. Accept
it only while the intended brainstem is wired over USB:

```sh
PETE_ACCEPT_BRAINSTEM_REPLACEMENT=1 just possess
```

## Bring-up

1. Build service-mode firmware: `just brainstem-pico-w-uf2`.
2. Install the BOOTSEL automount once on the motherbrain: `just setup-pico-bootsel`.
   When a Pico/Pico W appears as the `RPI-RP2` BOOTSEL volume, udev starts a
   systemd helper that mounts it at `/media/$USER/RPI-RP2` with write
   permissions for the operator user.
3. Flash it with `just flash`. The recipe performs an HTTP handshake, acquires
   a five-second `bootsel` service lease, requests BOOTSEL, waits for `RPI-RP2`,
   and copies the UF2. If a writable `RPI-RP2` volume is already mounted after
   manual BOOTSEL, it skips the network BOOTSEL request. It never silently uses
   the legacy endpoint.
4. Connect Pico USB to the motherbrain and run `ls -l /dev/serial/by-id`.
5. Run `cargo run -p pete-cockpit --example motherbrain_bootstrap`.
6. Configure `PETE_BRAINSTEM_WIFI_IPV4`, `PETE_DHCP_CLIENT_ID_HEX`, and optionally
   `PETE_BRAINSTEM_INTERFACE` to exercise registration and DNS verification.
7. Lift the wheels, then run the same command with
   `-- --possess-smoke --wheels-off-floor` for the 50 mm/s, 125 ms TTL motion.

The non-motion authorization check is safe to run with the robot on the floor:

```sh
cargo run -p pete-cockpit --example motherbrain_bootstrap -- --lease-expiry-smoke
```

It proves that an expired lease and a superseded lease fail closed, a newly
issued lease advances identity and generation, and only the fresh lease can
send a lease-bound heartbeat stop. It finishes stopped while the brainstem
continues supervising the body. Create OI is not exposed to the motherbrain.

The example remains a bounded protocol proof. The production path is now
`pete-tools robot --mode regular`; use the guarded wheels-off-floor
command in [real-robot-readonly.md](real-robot-readonly.md). Acquiring the
motherbrain lease is possession—there is no second arm layer. On orderly exit,
the production runner requires STOP and exorcize acknowledgements. Exorcize
releases the motherbrain possession gate without changing Create OI power or
mode; the brainstem continues its independent Full-mode supervision.

Useful diagnostics:

```sh
lsusb
ls -l /dev/serial/by-id
cargo run -p pete-cockpit --example contract_check -- uart /dev/serial/by-id/DEVICE
ping -c 4 192.168.4.1
curl -fsS http://192.168.4.1/status.json
```

## Physical QA session record

Use `just physical-qa` for a guided safety-validation session. It performs the
firmware identity read below, presents the physical cases and their acceptance
gates, and saves operator evidence under `data/reports/physical-qa/`. Use
`just physical-qa --plan` when reviewing or staging the procedure without
hardware.

At the start of every physical session, record the firmware identity before
collecting evidence or commanding motion. This distinguishes a dirty local
flash from the clean commit it was based on:

```sh
curl -fsS http://192.168.4.1/status.json | jq '{
  firmware_name, firmware_version, git_commit, git_commit_short, git_dirty,
  build_timestamp, build_profile, build_target, build_backend, build_id
}'
```

Attach that output to the session notes. `cargo run -p pete-tools -- robot
--capture PATH` and `cargo run -p pete-tools -- capture-real --out PATH` also
store the same values in the capture manifest when the brainstem status
transport supplies them.

Run the ping check from a DHCP client associated with the `pete-xxxx` AP. It
must report zero packet loss on an otherwise idle link. While polling
`/status.json` or the event stream, verify the UI/control transports remain
responsive and inspect `icmp_echo_requests`, `icmp_echo_replies`,
`icmp_dropped`, and `icmp_rate_limited`; ICMP is local management traffic, not
a route through the brainstem.

If handshake fails, confirm the stable device link, stop other serial readers,
unplug/replug the Pico, and rerun bootstrap. A manual STOP can be issued through
the dashboard or a scoped connector; STOP remains an emergency-class operation
and does not require a control lease. `PETE_ALLOW_LEGACY_BOOTSEL=1` is an explicit
development-only recovery switch and is never an automatic fallback.
