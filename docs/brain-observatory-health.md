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
ingress/history loss and lag. Browser reconnect count is labeled separately as
page-session evidence because it is not part of the replay capture.

The endpoint is `GET /api/observatory/component-health?at_ms=<observed-ms>`.
It returns only component events observed at or before the requested time and
uses the nearest retained `Now` snapshot for snapshot-backed provider health.
If the snapshot is no longer retained, event-backed rows remain available and
snapshot-backed fields remain absent.
