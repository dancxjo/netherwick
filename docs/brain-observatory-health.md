# Brain Observatory component health

The component-health view is a read-only, historical-as-of matrix for the
brainstem, Motherbrain runtime, optional higher-brain workers, sensor and
cognition providers, jobs, resources, queues, capture writer, model artifacts,
and Observatory transport. It consumes canonical `BrainEvent` provider, job,
resource, and queue state rather than inventing browser-only health.

Availability, health, occupancy, and authority are separate fields. An optional
provider may be truthfully missing without making the organism unhealthy. A
component may be available but busy, saturated, stale, thermally pressured, or
operating with reduced watchdog coverage. Authority is displayed as reported;
health never implies control authority.

Rows retain heartbeat and lease age, component-event age, tick period and
budget, stage timing, CPU, memory, temperature, queue depth and capacity,
drops, replacements, expired deadlines, inference p50/p95, reconnects, model
identity, candidate and rollback state, capture bytes and streams, missing
intervals, writer backlog, disk space, and the latest error. Missing metrics say
`not reported` rather than displaying fabricated zeroes.

Threshold history is derived from recorded event payloads. Queue pressure,
thermal pressure, stale heartbeats, low disk space, and tick-budget overruns are
linked back to their source event so an operator can move the whole Observatory
to that point in time. Current transport health additionally exposes bounded
ingress/history behavior and lag. Critical ingress rejection, telemetry drops,
intentional replacement, expected bounded-history expiry, viewer-local
broadcast lag, and browser reconnects are shown separately; none is presented
as a single generic loss counter. Browser reconnect count is page-session
evidence because it is not part of the replay capture.

The endpoint is `GET /api/observatory/component-health?at_ms=<observed-ms>`.
It returns only component events observed at or before the requested time and
chooses the maximum `(clock_epoch, observed time, delivery order)` state per
component, so a late older observation cannot overwrite a newer historical
state. It uses the nearest retained `Now` snapshot for snapshot-backed provider
health.
If the snapshot is no longer retained, event-backed rows remain available and
snapshot-backed fields remain absent.
