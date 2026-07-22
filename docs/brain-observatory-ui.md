# Brain Observatory timeline and Now inspector

Open `/view/observatory` on a running `pete-server`. The page is a read-only
engineering surface: it consumes the bounded BrainEvent transport and has no
control endpoints or authority in the runtime loop.

## Timeline

The left pane keeps a bounded browser window and virtualizes visible rows. It
can filter by event class, kind/search text, brain or component, modality,
trust, disposition, goal, entity, model artifact, and calibration epoch. Each
row shows occurred time separately from observed time and reports their
latency. Safety, authority, gate, command, outcome, calibration, and transport
gap events remain visually prominent.

Pause freezes the inspected event window while WebSocket ingestion continues.
The pending counter reports newly received events. Resume or Follow live moves
back to the newest event. The scrubber and keyboard selection operate over the
same filtered event set.

## Canonical Now

Every live `WorldSnapshot` update produces a canonical `now.snapshot`
BrainEvent and stores the corresponding serialized `Now` in a separate bounded
history. Selecting an event uses its `snapshot_id`; events without an explicit
snapshot reference use the nearest retained snapshot at or before their
occurred time. That fallback is labeled and never reads future state.

Dashboard-enabled simulation and physical possession also publish each actual
`RuntimeTick` through this same snapshot boundary. The publication expands the
tick's recorded sensations, impressions, and experiences, then records the
available higher-brain and forebrain exchanges, conductor proposal, safety
gate, accepted actuator command, observed actuator outcome, and provider/job/
queue/resource projections. These events retain frame, snapshot, goal,
command, and causal parent IDs. Scene calibration metadata publishes a
loss-intolerant epoch transition only when its checksum changes. Observatory
publication remains asynchronous and cannot grant or alter control authority.

The inspector recursively flattens the selected `Now` and virtualizes its
field rows. Metadata is inherited only from an enclosing field `meta` object,
so the interface reports source, age, confidence, uncertainty, freshness,
trust, calibration epoch, and evidence only where the canonical data actually
contains them. Missing metadata stays visibly unavailable. Changed fields are
compared with the immediately preceding retained snapshot. Pinned field paths
are stored in browser session storage; raw JSON remains a secondary view.

The live retained snapshot capacity is 2,048 ticks and the browser event
capacity is 20,000 records. Older selections remain visible as transport or
retention gaps rather than being silently replaced with the latest state.
