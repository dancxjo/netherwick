# Resilient multi-brain communications

Pete uses routed failure domains, not one implicit LAN. The Pico W brainstem
owns physical safety and its management network. The checked-in networkd files
under `configs/network` remain one concrete deployment profile (`wlan1`,
`wlan0`, `eth0`, and `10.42.0.0/24`), not protocol requirements. Higher-brain
services use `DataPlaneConfig` to enumerate working interfaces, prefer
Ethernet, fall back to ordinary LAN or Wi-Fi, and explicitly exclude configured
brainstem interfaces. No interface is bridged.

The higher-brain network can be a direct Ethernet link, an isolated switch, a
normal LAN, or a dedicated Wi-Fi association; local operation needs neither a
household router nor internet access. It survives Pico reboot and body-network
loss. A dedicated routed installation may use the included default-deny
firewall profile. Forebrains may
reach discovery, diagnostics, handshake, and recovery-policy endpoints, but
network reachability never grants a session or control lease.

The motherbrain provides split-horizon DNS and mDNS directly on Ethernet, so a
forebrain can resolve and use motherbrain services while the Pico is absent.
If the motherbrain itself fails, recovery requires an independent forebrain
path to the Pico AP (normally the forebrain's own Wi-Fi interface); a dead Pi
cannot route Ethernet packets. This is an explicit physical redundancy
requirement, not something routing software can synthesize.

## Authority layers

Communication has five separate layers:

1. A transport locates and carries bytes.
2. HELLO/WELCOME identifies a device boot and creates a fresh session.
3. A role policy may grant one expiring control lease.
4. Arming and motion remain explicit operations under that lease.
5. Maintenance requires a distinct, expiring service lease and never inherits
   control authority.

Every request has one explicit authorization class: read-only, emergency,
session, control lease, or service lease. The control matrix is deliberately
narrow: a control-purpose motherbrain session may request motherbrain
authority; a control-purpose forebrain session may request recovery authority
only after takeover conditions; and a control-purpose operator session may
request debug authority only when policy enables it. Diagnostic sessions and
service tools cannot acquire motion authority. Service leases are limited to
direct USB and an explicitly compiled `service-mode`; acquiring one first
stops motion and revokes the control lease.

Motherbrain replacement, operator debug, and forebrain recovery all execute
`STOP -> clear queue -> revoke heartbeat/lease -> install new lease -> remain
disarmed`. Diagnostic operator sessions occupy bounded read-only slots and do
not disturb the motherbrain. E-stop and STOP remain recovery-safe without a
lease. BOOTSEL and component restart operations require service authorization
and are rejected by ordinary control sessions. BOOTSEL remains operationally
disabled in this milestone even when a service lease exists.

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

Higher-brain discovery is the advisory, versioned
`netherwick-higher-brain/1` UDP advertisement described in
[higher-brain framework](018-higher-brain-framework.md). It advertises role,
stable node identity, per-boot identity, capabilities, readiness, and every
eligible service endpoint. Multicast and a configurable ordered list of
unicast seeds are both supported. Discovery never authorizes transfer, jobs,
or activation.

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
