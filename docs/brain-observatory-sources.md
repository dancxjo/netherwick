# Brain Observatory Sources and Navigation

`BrainEventSource` is the shared data boundary for live and recorded Brain
Observatory sessions. UI code asks every source for identity, filtered history,
snapshots, assets, health, seek results, and an optional incremental
subscription. Causal meaning stays in `BrainEvent`; it never branches on
whether the source is live or replay.

`LiveBrainEventSource` wraps the bounded `BrainEventHub` from the live server.
`ReplayBrainEventSource` accepts native recorded envelopes without changing IDs,
times, clock/calibration epochs, trust, disposition, payloads, or typed links.
Legacy Worldlab captures are adapted frame-by-frame: snapshots and recorded
event payloads are preserved, asset paths remain lazy references, clock resets
start a new explicit epoch, missing frame ranges become gaps, and missing
streams/manifests remain health warnings.

Recorded truth and optional recomputation are separate lanes:

- `recorded` is the immutable captured event stream used by normal replay;
- `reprocessed { model_id }` contains candidate-model output for comparison.

Source queries select `recorded`, `reprocessed`, or `all`, with an optional
reprocessed model ID. Every returned envelope retains its lane metadata, so
candidate identity does not depend on naming conventions in `kind`. Recorded
remains the default, and candidate output cannot replace or silently amend
historical trust decisions. Live snapshot discovery paginates through bounded
history rather than truncating at the per-query limit.

`ObservatoryNavigationState` is serializable deep-link state. It retains source,
playback mode, selected time/event, active panel, filters, speed, and loop range.
It supports play, pause, step, bounded speed, seek, loop, and follow-live. A seek
selects only the newest snapshot at or before the requested time, preventing
future state from leaking backward. Follow-live clears the replay time while
retaining source, panel, selection filters, and other navigation context.

Stream sequence remains delivery order. Occurred/observed times and clock epochs
are display/filter facts and never reorder replay. Incomplete sources remain
navigable, but `ObservatorySourceHealth.complete` is false and its typed gaps and
warnings are always available to the UI. Intentional state-projection
replacement is a `coalesced` discontinuity with the replacement sequence, not
an unavailable-history gap. Capture frame gaps end immediately before the next
present frame.
