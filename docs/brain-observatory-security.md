# Brain Observatory access security

The Observatory contains live robot state, imagery, maps, evidence, model and
calibration identities, and portable diagnostic bundles. Read-only means it has
no control authority; it does not mean the data is public.

## Binding and credentials

Simulation, diagnostic replay, and documented dashboard defaults bind to
`127.0.0.1`. Loopback is treated as a locally trusted transport. A
non-loopback bind fails before listening unless all of the following hold:

- `PETE_OBSERVATORY_TOKEN` is a non-empty read-only bearer credential;
- native TLS is enabled, or `PETE_OBSERVATORY_BEHIND_TLS_PROXY=1` declares that
  a trusted same-host reverse proxy terminates TLS;
- browser origins are same-host or explicitly listed, comma separated, in
  `PETE_OBSERVATORY_ALLOWED_ORIGINS`.

Send API credentials only in `Authorization: Bearer ...`; never put them in a
URL, diagnostic bundle, or command line that is retained in shell history. The
server stores only SHA-256 token digests and security events contain only the
route and rejection class. They never contain the credential or raw Origin.

`PETE_CONTROL_TOKEN` is separate and is not implied by Observatory read access.
When remote authentication is active, a read token receives `404` for command,
hardware-arm, calibration mutation, behavior promotion, inline-learning
mutation, and command-WebSocket routes. If no control token is configured those
routes stay unavailable remotely. Existing runtime, hardware-arm, watchdog,
calibration-promotion, navigation, and physical-safety gates still run after
control authentication; authentication grants no safety authority.

Browser WebSockets cannot attach a normal bearer header. For remote browser use,
put the Pi service behind an authenticated TLS reverse proxy or access gateway
that injects the read bearer only after its own user authentication and applies
the same Origin allowlist to HTTP and WebSocket upgrades. Keep the Pete process
bound to loopback when the proxy is on the Pi. Strip incoming Authorization
before setting the upstream credential, disable proxy request-body logging, and
mark diagnostic responses `Cache-Control: no-store`.

Native TLS is available through the existing `--live-tls` and
`--dashboard-tls` options. A safe proxy deployment terminates TLS on 443,
authenticates the engineer, proxies only to `127.0.0.1:8787`, preserves `Host`
and WebSocket upgrade headers, and does not expose Reign/control locations in
its read-only route set.

## Endpoint inventory

Every route below requires read authorization when remotely exposed. All are
origin checked; raw/live image and general live-state routes outside this table
are protected by the same global policy.

| Method | Route | Data | Bound |
| --- | --- | --- | --- |
| GET | `/view/observatory` | diagnostic UI | static page |
| GET | `/api/observatory/history` | retained and durable events | 2,000 events/page |
| GET | `/api/observatory/health` | transport/durability health | constant size |
| GET | `/api/observatory/snapshots/{id}` | historical robot state | one snapshot |
| GET | `/api/observatory/snapshot` | historical robot state | one snapshot |
| GET | `/api/observatory/events/ws` | live event stream | bounded broadcast; 30 upgrades/minute |
| GET | `/api/observatory/provenance/{id}` | evidence/artifact graph | bounded graph; shared 60/minute budget |
| GET | `/api/observatory/authority` | command/safety history | bounded history; shared 60/minute budget |
| GET | `/api/observatory/calibration` | calibration evidence/artifacts | bounded history; shared 60/minute budget |
| GET | `/api/observatory/spatial` | imagery/map/spatial evidence | bounded retained assets; shared 60/minute budget |
| GET | `/api/observatory/component-health` | component/resource health | bounded history |
| GET | `/api/observatory/diagnostic-export` | portable evidence bundle | 50,000 events; 8/minute |
| POST | `/api/observatory/diagnostic-verify` | untrusted bundle | 16 MiB; 8/minute |
| GET | `/api/observatory/compare` | multi-lane state comparison | bounded history; shared 60/minute budget |

Malformed, oversized, rate-limited, unauthorized, and cross-origin requests
fail before expensive parsing or graph work. Each rejection authors a
loss-intolerant `observatory.access_rejected` event without secrets, so security
failures remain visible in durable Observatory history to authorized readers.
