# 020 Now World-Model Contract

`Now` is PETE's canonical, immutable snapshot of its current best beliefs about
self and world. It carries uncertainty and may contain competing hypotheses. It
does not claim objective, complete truth.

The canonical `self_model` region is specified in
[`021-self-model.md`](021-self-model.md). It distinguishes organism, body,
device/boot, host/process, session, capability, and action-authority identity;
a running process or LLM host is not PETE's identity.

```text
reality and external messages
  -> raw and derived Sensations / Features / typed claims
  -> WorldModelUpdater belief integration and revision
  -> immutable Now.world
  -> goals, memory, prediction, language, encoding, reporting
```

## Representation roles

- A raw `Sensation` records a provider observation. A derived Sensation records
  a transformed observation and preserves its lineage.
- A `Feature` is a typed, reusable evidence unit. It can participate in
  integration, similarity, binding, memory, and training; it is not world
  state by itself.
- An evidence claim adds source, time, authority, and transformation identity
  to a proposed fact. Direct observations, derived perception, memory recall,
  learned prediction, maps, action outcomes, humans, and LLMs are distinct
  source kinds.
- `Now.world` is the one canonical current belief snapshot. It contains typed
  self/body state, entities, local geometry, hazards, context, task/authority,
  temporal context, social beliefs, epistemic questions, and its update trace.
  Grounded semantic relations are a typed persistent region of the same
  snapshot; they do not replace current entity or capability beliefs.
- `ExperienceInstant` is the durable embodied record for a tick or interval.
  It captures the evidence batch and associated `Now` for replay, memory, and
  encoding; it is not a competing live world model.
- `Experience` links Instants, outcomes, actions, reward, and temporal context
  into a memory/training record.
- `ExperienceLatent` is compressed learned geometry for prediction,
  similarity, retrieval, and learned components. It is optional additional
  evidence, not the sole source of truth.
- Memory recalls, predictions, and imagined futures may re-enter the updater as
  evidence. Their provenance remains distinguishable from observation.

No parallel Sensorium or goal-private canonical world model is permitted.

## Belief metadata and missingness

Action-relevant beliefs use `BeliefMeta`: confidence, observation and validity
time, freshness, provenance, contradiction references, optional coordinate
frame, and `BeliefSourceKind`. Entity identity, bearing, distance, and
reachability can age independently. Canonical state remains typed rather than
being flattened into one JSON map.

Missing evidence is represented by `None`/absence. It is not silently changed
to `false` or `0.0`. Stale beliefs may remain as identity or historical memory
after their actionable bearing, distance, or presence has expired.

Initial policy:

| Belief | Current/aging policy | Invalidation behavior |
| --- | --- | --- |
| bumper/contact | current tick, short-lived | disappears when the body observation clears |
| entity bearing/distance | current 500 ms, aging through 2 s | removed after 2 s of aging; identity remains |
| person/sound identity | current 1 s, aging through 5 s | removed after 15 s |
| charger/object identity | current 2 s, aging through 15 s | removed after 60 s |
| map/memory claim | retained by its owning map/memory policy | re-enters as `MemoryRecall` or `Map`, never direct observation |
| learned prediction | valid for its declared horizon | expires or is superseded; never erases structured safety facts |

Confidence decays with age. Contradictory typed hypotheses coexist unless
evidence justifies resolution; input order is not a truth rule.

## Update authority

Only `WorldModelUpdater` emits the next `Now.world`. Evidence producers,
memories, learned models, humans, and LLMs publish claims. They do not mutate a
snapshot, select a goal, or overwrite canonical beliefs directly. Goal
arbitration remains downstream and outside the updater.

Goals receive `GoalInterpretationContext` built from `WorldModelSnapshot`.
Conductors and arbiters do not depend on camera, Kinect, lidar, audio, serial,
or model-provider payload types. Skills receive typed targets and bounded local
feedback rather than arbitrary global sensor payloads.

## Replay and audit

Every snapshot has a schema version, monotonically increasing revision, input
evidence ids, additions/updates/removals, confidence and freshness changes, contradiction
decisions, and builder implementation/version. Stable maps and sorted trace
entries make a fixed evidence sequence deterministic when the updater and
teachers are deterministic.

An actionable entity includes its evidence lineage, so goal diagnostics can
trace a selected target back to the observation, recall, prediction, or claim
that supported it. Replay compares rebuilt snapshots by revision and trace;
differences identify either changed evidence or a changed implementation.

The bounded temporal, social, and epistemic regions are specified in
[`023-temporal-social-epistemic-cognition.md`](023-temporal-social-epistemic-cognition.md).
Sleep/replay clock handling and the grounded relation contract are specified
in [`024-sleep-and-grounded-semantics.md`](024-sleep-and-grounded-semantics.md).
