# Physical motherbrain bootstrap

The Pico W USB connection is USB CDC over a serial byte stream. The host uses
the shared `UartCockpit` line-protocol implementation for both USB CDC device
nodes and GPIO UART; USB discovery always uses the stable `/dev/serial/by-id`
symlink and never assumes `/dev/ttyACM0`.

## Bring-up

1. Build service-mode firmware: `just brainstem-pico-w-uf2`.
2. Flash it with `just flash`. The recipe performs an HTTP handshake, acquires
   a five-second `bootsel` service lease, requests BOOTSEL, waits for `RPI-RP2`,
   and copies the UF2. If a writable `RPI-RP2` volume is already mounted after
   manual BOOTSEL, it skips the network BOOTSEL request. It never silently uses
   the legacy endpoint.
3. Connect Pico USB to the motherbrain and run `ls -l /dev/serial/by-id`.
4. Run `cargo run -p pete-cockpit --example motherbrain_bootstrap`.
5. Configure `PETE_BRAINSTEM_WIFI_IPV4`, `PETE_DHCP_CLIENT_ID_HEX`, and optionally
   `PETE_BRAINSTEM_INTERFACE` to exercise registration and DNS verification.
6. Lift the wheels, then run the same command with
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
`pete-tools robot --mode possession-slow`; use the guarded wheels-off-floor
command in [real-robot-readonly.md](real-robot-readonly.md). Acquiring the
motherbrain lease is possession—there is no second arm layer. On orderly exit,
the production runner requires STOP and exorcize acknowledgements. Exorcize is
translated internally to the brainstem's DISARM wire command.

Useful diagnostics:

```sh
lsusb
ls -l /dev/serial/by-id
cargo run -p pete-cockpit --example contract_check -- uart /dev/serial/by-id/DEVICE
curl -fsS http://192.168.4.1/status.json
```

If handshake fails, confirm the stable device link, stop other serial readers,
unplug/replug the Pico, and rerun bootstrap. A manual STOP can be issued through
the dashboard or a scoped connector; STOP remains an emergency-class operation
and does not require a control lease. `PETE_ALLOW_LEGACY_BOOTSEL=1` is an explicit
development-only recovery switch and is never an automatic fallback.
