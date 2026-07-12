# Cockpit session handshake and Pete network identity

## Topology and trust boundaries

Pete has three distinct computing roles. The Pico W is the **brainstem** and
owns Create OI, GPIO, deadlines, reflexes, E-stop, DHCP/DNS, and safety. The
installed Raspberry Pi 5 is the **motherbrain** and owns Linux, perception,
storage, firmware management, supervision, and the cockpit client. A
**forebrain** is higher-level cognition; it initially runs on the motherbrain
but may later run on another computer. A remote forebrain normally talks to a
motherbrain service and never acquires unrestricted brainstem authority.

USB serial is the primary motherbrain-to-brainstem transport. The Pico access
point (HTTP, WebSocket, or UDP) is a backup and management path. Both paths use
the same logical protocol. Network tasks only parse and submit bounded messages
to the runtime lane; they do not own motor GPIO, Create UART, power, or safety.

These values are deliberately different:

| Concept | Example | Meaning |
| --- | --- | --- |
| device identity | `pete-motherbrain-primary` | stable handshake identity |
| role | `motherbrain` | claimed and authorized function |
| hostname | `motherbrain.pete.internal` | network locator |
| address | `/dev/serial/by-id/...`, `192.168.4.2` | temporary transport location |

The role vocabulary is `brainstem`, `motherbrain`, `forebrain`, `operator`,
`simulator`, and `service_tool`. The production control pairing is
`motherbrain <-> brainstem`. A name, IP, MAC, DHCP order, or Wi-Fi association
does not prove role or identity.

## State machine and messages

Every motherbrain boot creates a `boot_id`; every attempt creates a fresh
`handshake_nonce`. The brainstem has a board-derived device ID and a fresh boot
ID. Protocol compatibility uses explicit major/minor numbers, never package or
firmware versions. Majors must match. The selected minor is the highest value
in the intersection (currently the lower supported value).

`HELLO` contains role, device/boot IDs, nonce, protocol range, named supported
and required features, preferred heartbeat, and diagnostic software/build
metadata. Unknown optional features are ignored. `WELCOME` echoes the nonce
and returns the brainstem identity, fresh session ID, negotiated protocol and
features, heartbeat/TTL bounds, event cursor, complete live capabilities,
software metadata, and a safety snapshot. `REJECT` carries a reason from
`wrong_role`, `protocol_major_mismatch`, `protocol_minor_incompatible`,
`missing_required_feature`, `invalid_identity`, `busy`, or `internal_error`.

```text
Motherbrain                   Brainstem
    |                            |
    |-------- HELLO ------------>|
    |                            | validate identity/version
    |                            | stop and invalidate old motion
    |                            | revoke leases; remain disarmed
    |<------- WELCOME -----------|
    |                            |
    |  validated, disarmed,      |
    |  no motion/control lease   |
```

A session ID is derived from both device IDs, both boot IDs, the hello nonce,
and a brainstem session serial. Duplicate identical hellos are idempotent, but
a reconnect or failover uses a new nonce and session. Before returning WELCOME,
the brainstem synchronously stops prior motion, clears queued motion and
heartbeat state, disarms, installs the session, and retains E-stop/fault state.
Handshake success is not motion authorization and does not authorize BOOTSEL,
flashing, service actions, or Raspberry Pi power control.

Read-only status, capabilities, events, and a new handshake are recoverable
without a session. Arming, motion, mode, power, safety-policy, firmware/service,
and reboot operations carry the active session ID and fail closed before the
runtime queue if it is missing, expired, unknown, or replaced. Heartbeats are
also session-scoped.

Session/network event payloads use compact numeric fields: `session_opened` and
`session_replaced` carry generation/previous-session-hash/new-session-hash;
`session_rejected.a` is the rejection code; `transport_changed.a/b` are old/new
transport codes (`1` HTTP, `2` WebSocket, `3` hardware UART, `4` UDP, `5` USB
CDC); `peer_reboot_detected.a/b` are old/new boot hashes; DHCP and DNS events
carry identity or registration generation plus packed IPv4; authority events
carry generation/session hash/lease hash. Hashes are diagnostic correlation
values, not credentials.

## Encodings

HTTP uses `POST /handshake` with the JSON representation. WebSocket uses the
same JSON object with `kind: "hello"`; a socket connection is not a session.
UART/USB and UDP use newline-delimited compact frames:

```text
HELLO <json-object> LF
WELCOME <json-object> LF
REJECT <json-object> LF
```

The JSON object is exactly the shared logical object. The maximum encoded
handshake line is 4096 bytes; oversized, non-UTF-8, and malformed JSON are rejected
without changing session state. Stream readers accumulate partial input and
extract every complete line; UDP treats one line as one datagram. The nonce
makes delayed/reordered replies stale. Commands append `session_id=<token>` in
compact form and use a `session_id` JSON property.

## Reconnect and failover

Same device and boot means a transport interruption; a new session is still
required. Same device with a new boot means brainstem reboot: rebuild the live
contract, reset event assumptions, and remain disarmed. A different device is
replacement hardware and requires explicit caller acceptance.

```text
Motherbrain          USB          Brainstem          Wi-Fi
    |--- command ---->|               |                |
    |   USB fails     X               |                |
    | best-effort stop/disarm         |                |
    |--------------------------------- HELLO ---------->|
    |<-------------------------------- WELCOME ---------|
    | compare device/boot; install fresh session        |
```

Only one connector object owns command authority. Backup probes are read-only
and cannot replace it. Failover automatically accepts the same brainstem device
ID, classifies a changed boot as reboot, and rejects a changed device unless
replacement policy explicitly permits it. USB discovery should enumerate
`/dev/serial/by-id`, never assign identity from `/dev/ttyACM0`.

## Internal DNS and registration

The configurable unicast domain defaults to `pete.internal`, avoiding `.local`
mDNS ambiguity. `brainstem.pete.internal` and aliases `pete`/`brainstem` point
to the AP gateway. `.local` remains available for mDNS. Candidate advisory
services are `_pete-brainstem._tcp`, `_pete-control._udp`,
`_pete-control._tcp`, and `_pete-handshake._tcp`; advertised IDs and versions
must still be verified by HELLO/WELCOME.

After a successful USB handshake, the motherbrain may send the session-bound
`REGISTER_NETWORK_ENDPOINT` with its role/device ID, interface, address,
hostname, DHCP client identity, and TTL. The brainstem verifies all of these
against the active motherbrain session and a current lease, then returns
`NETWORK_ENDPOINT_REGISTERED` with FQDN, address, bounded TTL, and generation.
Only this operation may publish the reserved `motherbrain` name. Reserved names
are `pete`, `brainstem`, `motherbrain`, `forebrain`, `gateway`, and `control`;
ordinary DHCP hostname proposals cannot claim them. Records expire with the
lease or registration and duplicates replace one unambiguous record.

Motherbrain startup discovers USB, handshakes, verifies the brainstem device,
builds the live contract, inspects/joins Wi-Fi, registers its leased endpoint,
verifies DNS, starts session heartbeat/supervision, and remains disarmed. It
logs both brainstem device identity and hostname because they answer different
questions. DNS failure never disables USB control.

Future forebrains discover `motherbrain.<internal-domain>` and establish a
separate forebrain-to-motherbrain session. The motherbrain advertises/proxies
cockpit, perception, and event services and decides how goals become brainstem
commands. Forebrain failure triggers process/service recovery first; only loss
of the motherbrain host reaches later brainstem host-recovery policy. This
phase establishes freshness and identity continuity, not cryptographic
authentication; future challenge/signature/key fields remain separate.
