# BrainEvent Live Transport and History

`pete_server::BrainEventHub` is the asynchronous observability boundary between
BrainEvent producers and live/history consumers. It is part of the existing
`LiveViewState` and Axum server; it does not run client I/O, history queries, or
asset retrieval in Pete's physical control loop.

Producers hand complete `BrainEvent` values to `LiveViewState::publish_brain_event`.
Publication validates the schema and inline-payload limit, performs one bounded
ingress operation, and wakes the independent worker. A slow or disconnected
client cannot block publication or the worker.

## Endpoints

- `GET /api/observatory/history` returns bounded history and an opaque numeric
  `next_cursor`.
- `GET /api/observatory/health` returns ingress/history depth and capacity,
  sequence bounds, connected clients, maximum observed client lag, and
  dropped/coalesced/expired counters.
- `GET /api/observatory/events/ws` upgrades to a read-only WebSocket. The same
  query parameters apply. `after_sequence` resumes after a prior cursor.

WebSocket and history records use one tagged shape:

```text
event { sequence, BrainEvent }
gap   { from_sequence, to_sequence, reason, TransportGap BrainEvent }
```

The stream accepts no control commands. Client text/binary data is ignored;
close and ping/pong are handled only as transport lifecycle messages.

## Ordering and cursors

The worker assigns a strictly increasing stream sequence when it accepts an
event from ingress. Stream sequence is the delivery/replay order and is the
only cursor order.

`BrainEvent.times.occurred` remains the source's claim about when something
happened. `times.observed` is when its producer observed or formed it. Either
may use a different clock epoch, so neither timestamp is used to reorder the
stream. Queries can filter both independently, but clients must not compare raw
millisecond values across clock epochs.

Reconnect after a retained sequence resumes exactly. If retention or
coalescing removed any required sequence, the response contains a
loss-intolerant `TransportGap` with the unavailable inclusive range. A lagged
WebSocket client receives the same typed gap rather than silently jumping to a
newer event.

## Bounded overload policy

Ingress, retained history, and each WebSocket broadcast receiver have separate
fixed capacities.

- Coalescible telemetry uses the stable key declared by `BrainEvent.loss_policy`.
  A newer pending value replaces the older pending value with the same key.
- New telemetry is dropped explicitly when ingress is full; health counters
  expose the loss.
- A loss-intolerant event arriving behind telemetry evicts telemetry and is
  queued. It is never coalesced.
- If ingress contains only loss-intolerant events, another critical event is
  explicitly rejected with `CriticalQueueFull`; it is never silently lost.
- History evicts/coalesces telemetry before critical events. Normal bounded
  retention may eventually expire critical history, but increments the
  critical-expiry counter and forces a typed cursor gap.
- Broadcasting never waits for clients. A stalled client falls behind its own
  bounded receiver and gets a typed lag gap.

Gate decisions, commands, outcomes, calibration transitions, transport gaps,
and safety/authority-significant records are validated as loss-intolerant by
the canonical contract. Lease, STOP/E-stop, and command acknowledgement
producers must classify their records through those event/significance fields.

## Historical filters

Both history and live WebSocket requests support:

- `after_sequence` and `limit`;
- `occurred_from_ms`, `occurred_to_ms`, `observed_from_ms`, and
  `observed_to_ms`;
- `event_type`, `component`, `snapshot`, `entity`, `goal`, and `command`;
- `trust` and `disposition`.

Ranges are inclusive. Empty identifiers, reversed ranges, zero limits, and
limits above the configured maximum are rejected. Axum rejects unknown enum
values during query decoding. Filtering advances the cursor over inspected
records, not only matches, so a sparse filter can still reconnect without
re-reading the same retained window.

Large images, audio, depth, point clouds, lidar, crops, and vectors remain lazy
asset references under the BrainEvent contract. Inline JSON above 16 KiB is
rejected before entering either bounded queue.

## Shutdown

Closing the hub wakes the worker, drains already accepted ingress, closes live
receivers, and rejects later publication. Dropping the last external hub handle
also closes it, so server teardown does not leave an observability task alive.
