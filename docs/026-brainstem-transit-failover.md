# 026 Brainstem transit and host-aware failover

The brainstem emergency network is a last-resort bodily and recovery plane. A
host joining it has gained a path, not authority. The authoritative body owner
is always the fresh controller reported by the brainstem possession surface.

## Default and degraded topologies

| Failure | Body controller | Forebrain-to-motherbrain path | Motherbrain-to-brainstem path |
| --- | --- | --- | --- |
| None | motherbrain | direct Ethernet | USB |
| Forebrain Ethernet | motherbrain | ordinary Wi-Fi, then brainstem transit | USB |
| Forebrain absent | motherbrain | unavailable | USB |
| Motherbrain USB | motherbrain | unchanged | brainstem Wi-Fi |
| Motherbrain absent and no controller | none, then atomic-acquisition winner | brainstem Wi-Fi | brainstem Wi-Fi |
| Motherbrain returns during fallback | forebrain until safe handback | best available peer path | no competing control |
| No hosts | none | unavailable | unavailable; brainstem safety remains local |

The shared implementation is `pete_higher_brain::failover::HostFailover`.
Both Linux roles feed it link, peer, and authoritative controller observations.
It emits path/role actions and one structured event per state transition. It
does not hold a Cockpit connector and cannot manufacture a lease.

The forebrain fallback order is fixed:

1. keep or restore direct Ethernet;
2. search ordinary Wi-Fi for `motherbrain.local`;
3. join the configured brainstem network;
4. resolve the session-bound `motherbrain.pete.internal` record and look for
   motherbrain on the emergency segment;
5. remain subordinate whenever a fresh controller exists;
6. after a configured grace period, request atomic acquisition only when a
   fresh authoritative observation says the body is uncontrolled.

Unknown, missing, or stale controller status fails closed. Failed acquisition
does not start the body-facing role. Stable node and boot identities prevent a
restarted stale process from being confused with its predecessor. Path return
uses hysteresis, and handback waits for a safe command boundary before the
fallback controller releases possession.

## Transit mechanism

The current RP2040/Pico W firmware exposes a deterministic `192.168.4.0/24`
Wi-Fi segment with DHCP and local DNS. Motherbrain maintains a brainstem-side
Wi-Fi link in addition to its preferred USB control link and registers its
leased address as `motherbrain.pete.internal` through its identified session.
The brainstem verifies that the registration matches the live DHCP client,
device identity, and boot identity and expires it with the lease. A forebrain
that joins the AP can therefore reach the still-controlling motherbrain over a
real local IP path without a USB Ethernet gadget, NAT, or an external router.

The emergency policy permits only peer discovery, health, role coordination,
possession status/acquisition, and handback ports. It does not advertise SSH,
bundle transfer, model services, perception streams, packages, or updates.
Association and IP reachability do not modify the controller lease. Direct
brainstem control still requires its own identified session and atomic lease.

The service allowlist and ports are explicit in
`[failover.permitted_emergency_services]`; there is no allow-all fallback and
no default-route replacement for general traffic.

## Configuration and credentials

See `configs/higher-brain/forebrain.example.toml`. Configure stable node
identity separately from boot identity. `brainstem_credential_ref` names an
external NetworkManager or systemd credential; it must never contain the
credential and is safe to expose in diagnostics. Fallback takeover defaults to
disabled for incomplete deployments.

The status/reporting surface should publish the state, advertised role, active
path, peer visibility, authoritative controller node, state age, actions, and
transition reason from each `FailoverDecision`. Secret values are never part of
that object.

## Failure injection and operator recovery

Run the deterministic matrix without hardware:

```bash
cargo run -q -p pete-higher-brain -- failover-check
cargo test -p pete-higher-brain failover
```

The matrix covers normal operation, direct-link loss, ordinary-Wi-Fi recovery,
brainstem transit without takeover, motherbrain USB-to-Wi-Fi transport change,
last-resort atomic acquisition, acquisition races, stale controller status,
path flapping, and orderly handback.

On hardware, diagnose in this order:

1. inspect both hosts' node/boot identity and failover decision;
2. inspect the brainstem controller owner, generation, and lease freshness;
3. verify `motherbrain.local` on ordinary Wi-Fi before joining the emergency
   SSID;
4. verify the session-bound motherbrain DNS registration and bounded emergency
   service health rather than sending bulk test traffic;
5. leave takeover disabled until the atomic acquisition and handback matrix has
   passed for the installed hardware.
