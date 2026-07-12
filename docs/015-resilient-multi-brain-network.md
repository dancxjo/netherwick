# Resilient multi-brain communications

Pete uses routed failure domains, not one implicit LAN. The Pico W brainstem
owns physical safety and the `192.168.4.0/24` management network. The Pi 5
motherbrain connects to that network on `wlan1`, to infrastructure on `wlan0`,
and to the independent `10.42.0.0/24` interbrain Ethernet backbone on `eth0`.
The network configuration is in `configs/network`. No interface is bridged.

The interbrain network survives Pico reboot, body power loss, and either Wi-Fi
failure. The motherbrain routes it with a default-deny firewall. Forebrains may
reach discovery, diagnostics, handshake, and recovery-policy endpoints, but
network reachability never grants a session or control lease.

The motherbrain provides split-horizon DNS and mDNS directly on Ethernet, so a
forebrain can resolve and use motherbrain services while the Pico is absent.
If the motherbrain itself fails, recovery requires an independent forebrain
path to the Pico AP (normally the forebrain's own Wi-Fi interface); a dead Pi
cannot route Ethernet packets. This is an explicit physical redundancy
requirement, not something routing software can synthesize.

## Authority layers

Communication has four separate layers:

1. A transport locates and carries bytes.
2. HELLO/WELCOME identifies a device boot and creates a fresh session.
3. A role policy may grant one expiring control lease.
4. Arming and motion remain explicit operations under that lease.

Motherbrain replacement, operator debug, and forebrain recovery all execute
`STOP -> clear queue -> revoke heartbeat/lease -> install new lease -> remain
disarmed`. Diagnostic operator sessions occupy bounded read-only slots and do
not disturb the motherbrain. E-stop and STOP remain recovery-safe without a
lease. BOOTSEL and host power operations require a future service authorization
and are rejected by ordinary control sessions.

Forebrain recovery is allowed only after the active lease has expired. A
returning motherbrain does not automatically preempt recovery authority; an
explicit reconciliation and hand-back is required. Brainstem loss leaves the
interbrain backbone and higher cognition intact, while motherbrain loss leaves
the Pico management network available for policy-governed recovery.

Takeover is default-deny. Network motherbrain control handshakes are accepted
only after USB/UART has recognized the same device identity. Operator debug
leases require the firmware `operator-debug` feature. Forebrain recovery
requires `PETE_RECOVERY_FOREBRAIN_ID` at firmware build time and a matching
session identity in addition to lease expiry. These continuity checks are not
a substitute for the future signature-based authentication layer.

## Discovery and diagnostics

The brainstem is authoritative for `pete.internal`. `pete`, `brainstem`, and
`gateway` resolve to the AP address. `motherbrain.pete.internal` exists only
after a session-bound registration matches a live DHCP client identifier and
expires no later than that lease. Reserved names are never populated from DHCP
hostname claims.

`lease_identity` is the lowercase hexadecimal encoding of DHCP option 61; when
option 61 is absent it is the lowercase hexadecimal client MAC. Encoding is
explicit so arbitrary binary DHCP identifiers compare consistently in JSON and
the compact protocol.

mDNS announces `_pete-brainstem._tcp`, `_pete-debug._tcp`, and both TCP and UDP
`_pete-control` services. Metadata is advisory; every client still handshakes.
HTTP exposes status plus bounded `/network.json` and `/sessions.json`
diagnostics. These endpoints contain counts/generations and locators, not
authority secrets.

Motherbrain bootstrap enumerates `/dev/serial/by-id`, opens USB first,
handshakes, checks the configured brainstem device ID, builds the live contract,
registers the `wlan1` DHCP identity/address, then explicitly requests a control
lease. Wi-Fi failover always performs a new handshake and session. A different
brainstem device requires explicit replacement acceptance.

This protocol establishes freshness and identity continuity. It is not yet
cryptographic authentication; key IDs, challenges, signatures, and trusted
replacement certificates remain an intentionally separate extension.
