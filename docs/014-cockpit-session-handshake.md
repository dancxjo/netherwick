# Cockpit session handshake and Pete network identity

## Topology and trust boundaries

Pete has three distinct deployment roles. The Pico W is the **brainstem** and
owns Create OI, GPIO, deadlines, reflexes, E-stop, DHCP/DNS, and safety. The
installed Raspberry Pi 5 is the **motherbrain** and owns Linux, perception,
storage, firmware management, supervision, and the cockpit client. A
**forebrain** is higher-level cognition; it initially runs on the motherbrain
but may later run on another computer. A remote forebrain normally talks to a
motherbrain service and never acquires unrestricted brainstem authority.

These established protocol names map to the stable architectural roles Body
Controller, Organism Runtime, and Cognitive Accelerator. They are not host or
organism identities. New cognitive APIs use the stable names while this
Cockpit protocol retains its compatibility vocabulary; see
[022-cognitive-roles.md](022-cognitive-roles.md).

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
    |                            | revoke leases; stop motion; retain Full-mode supervision
    |<------- WELCOME -----------|
    |                            |
    |  validated, stopped,       |
    |  no motion/control lease   |
```

A session ID is derived from both device IDs, both boot IDs, the hello nonce,
and a brainstem session serial. Duplicate identical hellos are idempotent, but
a reconnect or failover uses a new nonce and session. Before returning WELCOME,
the brainstem synchronously stops prior motion, clears queued motion and
heartbeat state, installs the session, and retains E-stop/fault state. Create
OI acquisition and Full-mode maintenance remain brainstem-owned throughout.
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

The JSON object is exactly the shared logical object. Firmware accepts at most
1024 bytes for a compact request line and reserves 4096 bytes for the larger
WELCOME/capability response; the hosted decoder independently caps either
direction at 4096 bytes. Oversized, non-UTF-8, and malformed JSON are rejected
without changing session state. Stream readers accumulate partial input and
extract every complete line; UDP treats one line as one datagram. The nonce
makes delayed/reordered replies stale. Commands append `session_id=<token>` in
compact form and use a `session_id` JSON property.

The RP2040 buffers are fixed and live in Embassy task futures/static executor
storage, not on an unbounded heap. USB owns a 1024-byte line and 4096-byte
response plus 768 bytes of descriptors and packets. UDP owns 1024-byte request
and 4096-byte response buffers plus 2560 bytes of socket storage. WebSocket
owns 1024-byte payload, 4096-byte response, 3072 bytes of TCP storage, and a
512-byte upgrade request. Each of the three HTTP task instances owns a
1024-byte request, 4096-byte JSON response, and 3072 bytes of TCP storage
(about 8 KiB per instance). Hardware UART owns one 1024-byte line; its
4096-byte handshake response exists only while processing HELLO. Overflow
clears the partial frame, emits `line_too_long` where a reply is possible, and
never mutates session state. The release-link audit currently reports roughly
92 KiB of static RAM for the complete firmware.

Compact v1 deliberately uses bounded JSON after the `HELLO` token so every
transport shares one field model. It does not serialize Rust memory layouts.
If request growth approaches 1024 bytes, a future protocol minor must define a
smaller field grammar rather than silently increasing firmware buffers.

The hosted conformance harness runs the same vertical slice against the real
`UartCockpit` (and therefore USB CDC byte-stream connector), `UdpCockpit`,
`HttpCockpit`, and `WebSocketCockpit` implementations using protocol peers:
HELLO/WELCOME, unscoped-motion rejection, session-only motion rejection,
session-derived network registration, control-lease acquisition, leased
motion, and rejection of BOOTSEL under an ordinary control lease. Because the
trait no longer provides handshake or session fallbacks, a new connector must
implement every scoped operation before it compiles.

## Reconnect and failover

Same device and boot means a transport interruption; a new session is still
required. Same device with a new boot means brainstem reboot: rebuild the live
contract, reset event assumptions, and remain stopped. A different device is
replacement hardware and requires explicit caller acceptance.

```text
Motherbrain          USB          Brainstem          Wi-Fi
    |--- command ---->|               |                |
    |   USB fails     X               |                |
    | best-effort stop                |                |
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
`REGISTER_NETWORK_ENDPOINT` with its interface, address, hostname, DHCP client
identity, and TTL. Role, device ID, and boot ID come exclusively from the
active session record; they are not repeated as caller-controlled registration
fields. The brainstem verifies that session against a current lease, then returns
`NETWORK_ENDPOINT_REGISTERED` with FQDN, address, bounded TTL, and generation.
Only this operation may publish the reserved `motherbrain` name. Reserved names
are `pete`, `brainstem`, `motherbrain`, `forebrain`, `gateway`, and `control`;
ordinary DHCP hostname proposals cannot claim them. Records expire with the
lease or registration and duplicates replace one unambiguous record.

Motherbrain startup discovers USB, handshakes, verifies the brainstem device,
builds the live contract, inspects/joins Wi-Fi, registers its leased endpoint,
verifies DNS, and starts session heartbeat/supervision. The brainstem already
owns Create OI startup and continuous Full-mode maintenance; acquiring a
control lease grants full command authority without a separate arm or mode step. It
logs both brainstem device identity and hostname because they answer different
questions. DNS failure never disables USB control.

Future forebrains discover `motherbrain.<internal-domain>` and establish a
separate forebrain-to-motherbrain session. The motherbrain advertises/proxies
cockpit, perception, and event services and decides how goals become brainstem
commands. Forebrain failure triggers process/service recovery first; only loss
of the motherbrain host reaches later brainstem host-recovery policy. This
phase establishes freshness and identity continuity, not cryptographic
authentication; future challenge/signature/key fields remain separate.

Session and lease tokens in protocol v1 use compact FNV-derived 32-bit hashes
plus monotonic generations. They are collision-prone bookkeeping identifiers,
not authenticators or secrets. Possessing a session or motion lease never
authorizes maintenance. BOOTSEL and component restart requests require a
separate service lease; firmware grants such a lease only in an explicitly
compiled service mode over direct USB according to role policy. BOOTSEL itself
remains disabled in this milestone even after service authorization.
