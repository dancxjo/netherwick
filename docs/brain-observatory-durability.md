# Brain Observatory durable critical history

The live Observatory keeps bounded in-memory history, but every `BrainEvent`
whose loss policy is `loss_intolerant` is also sent to a dedicated append-only
writer. The runtime publisher and Observatory ingestion worker never wait for
disk I/O. Queue saturation, disk-full errors, and writer failures leave the live
event path running and raise explicit durability failures and gaps.

## Storage and recovery

Real-robot and simulation dashboards use separate logs under
`data/observatory/` by default. Set `PETE_OBSERVATORY_HISTORY_DIR` to place both
logs on another local filesystem. Records are newline-framed JSON containing
the original global sequence and complete event plus a SHA-256 checksum of the
canonical serialized envelope. Event IDs, causal references, clock epochs,
artifact identities, and payload checksums therefore round-trip unchanged.

At startup the reader scans rotated segments from oldest to newest, verifies
each record, restores the highest valid sequence, and deduplicates by stable
event ID. A partial or invalid tail is truncated back to the last complete
checksummed record. All valid preceding records remain replayable and
`durability_gaps` records that recovery found an unavailable tail. Re-publishing
an identity already present in the retained log returns `duplicate` and does
not allocate another sequence.

The default policy keeps an active 64 MiB segment plus eight rotated segments.
Rotation discards the oldest complete segment; historical queries then expose
the missing sequence interval as normal retention expiry. High-rate
`coalescible` projections remain only in bounded live history and are not
written to the critical log.

## Health and failure semantics

`/api/observatory/health` and component health expose:

- `durable_writer_backlog`;
- `durable_write_failures`;
- `last_durable_sequence`;
- `durability_gaps`;
- recovered-record and rotation counts.

Any write failure or durability gap makes `observatory.transport` failed and
produces a critical component-health alert. This is an evidence-integrity
failure, not permission to relax navigation, calibration-promotion, or physical
safety gates. Operators should stop relying on the diagnostic record, restore
disk capacity or permissions, and restart the session. The valid prefix remains
available; unavailable tails are never presented as complete evidence.
