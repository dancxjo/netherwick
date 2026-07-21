# BrainEvent Contract

`pete_events::BrainEvent` is the canonical, versioned observability envelope for
live observation, capture, replay, explanation, and Brain Observatory clients.
It adapts existing domain records; it does not replace them or grant an
observability consumer control authority.

The causal vocabulary is:

```text
Evidence -> Interpretation -> BeliefUpdate -> Proposal -> GateDecision
         -> Command -> Outcome
```

Calibration transitions and provider, job, resource, queue, snapshot, and
transport-gap records use the same envelope and typed references.

## Identity and replay

`event_id` is stable within the producing domain. `event_type` is the canonical
class used by transport and loss policy, while `kind` is the producer's stable,
domain-specific name such as `safety.decision` or `vision.object_detection`.
Existing UUID-bearing records
use namespaced IDs such as `sensation:<uuid>`, `impression:<uuid>`,
`experience:<uuid>`, and `reign-input:<uuid>`. Producers without a durable
domain ID must allocate one before publication and retain it in capture/replay.
They must not mint a different ID when replaying the same record.

`BrainEventIndex` defines deduplication semantics:

- an identical envelope with an existing ID is a replay duplicate;
- different content under an existing ID is an integrity error;
- a new ID is a new immutable historical record.

`record_kind` distinguishes immutable `historical_event` records from
`state_projection` records such as canonical `Now`. Projections identify their
snapshot and the historical events supporting it; they never rewrite those
events. `BrainEvent::from_now_snapshot` is source-neutral, so live and replay
produce the same contract when given the same recorded snapshot identity and
times.

## Time

`times.occurred` records when the source says the event happened;
`times.observed` records when the producer received or formed it. `valid_from`
and `expires_at` describe the assertion's validity window. Every timestamp may
carry a clock epoch. Times from different epochs or clock domains must not be
ordered by their raw millisecond values. Expiry is evaluated only inside the
same epoch.

## Causality and quality

`links.parents`, `links.supports`, and `links.contradicts` contain typed event
references. A missing target remains a visible broken reference; it is not
deleted or turned into prose. `causal_references()` provides recursive “why?”
consumers with the relation and expected target class without parsing a
summary.

Each envelope separately carries confidence, uncertainty (including its
measure and optional unit), freshness, trust,
and disposition. Unknown or unavailable values stay explicit. Disposition
includes `unknown`, `unavailable`, `rejected`, `expired`, `superseded`,
`vetoed`, and `accepted`; none of these should be inferred from confidence.

Calibration epochs and model/configuration/vector/capture identities are
first-class fields. Artifact checksums should be supplied whenever the source
has them.

## Payload boundary

Inline payloads are compact JSON and are limited to 16 KiB. Images, audio,
depth, point clouds, lidar scans, crops, vectors, and other large assets are
represented by `PayloadReference` and `ArtifactIdentity`. Existing sensation
adapters enforce that default. A locator is an opaque reference for an
allowlisted, read-only asset resolver; it is not an unrestricted URL or control
endpoint. Redacted assets retain a reference and manifest metadata with
`redacted: true`.

## Loss policy

Every event declares either `loss_intolerant` or a stable coalescing key.
Validation rejects coalescing for the following classes:

| Event or significance | Required policy | Reason |
| --- | --- | --- |
| gate decision | loss-intolerant | preserves acceptance, rejection, and veto |
| command | loss-intolerant | preserves requested physical effects |
| outcome | loss-intolerant | preserves acknowledgement and actual result |
| calibration transition | loss-intolerant | prevents epoch/trust ambiguity |
| transport gap | loss-intolerant | makes missing history explicit |
| safety transition | loss-intolerant | preserves STOP, E-stop, and reflex changes |
| authority transition | loss-intolerant | preserves lease, Reign, and handoff changes |
| high-rate evidence/state | coalescible when safe | bounded by stable field/source key |

Proposals may be loss-intolerant when they cross a human, Reign, or other
authority boundary. LLM advisory telemetry remains explicitly advisory and may
be coalesced; the adapter never upgrades it to executable authority.

## Existing-domain mapping

Producers adapt their existing domain records at the asynchronous observability
boundary. The mapping is deliberately one-way so observability cannot enter the
control loop.

| Existing source | BrainEvent mapping |
| --- | --- |
| `Now` | `Snapshot` state projection with stable snapshot reference; full state lazy-loaded |
| `Sensation` | `Evidence`; parent/provenance sensations become typed links; heavy payloads referenced |
| `Impression` | `Interpretation`; `about` sensations become supporting evidence |
| `Experience` / Instant | `BeliefUpdate`; impressions are parents, sensations are supports, experience ID retained |
| ledger frame/transition | snapshot reference plus historical outcome links; keep ledger frame/transition UUID |
| brainstem sensor/reflex/ack | `Evidence`, `GateDecision`, or `Outcome` from `Brainstem`; preserve device boot/clock epoch |
| goal evaluation / arbitration | `Proposal` or `GateDecision`; preserve all candidate goal IDs and scores |
| skill run | `Proposal`, `Command`, and `Outcome`; preserve run, command, and preemption IDs |
| Reign input/outcome | built-in proposal/gate adapters; human source, TTL, disposition, and input ID retained |
| LLM response | `Interpretation` or advisory `Proposal`; model identity and input snapshot required, never command authority |
| autonomic/final safety | built-in loss-intolerant `GateDecision`; proposal and snapshot are typed parents/references |
| calibration estimator | loss-intolerant `CalibrationTransition`; estimator and mount epochs plus evidence links retained |
| capture writer/manifest | `ResourceState` or `QueueState`; capture artifact identity, backlog, drops, and gaps retained |
| cognition/higher-brain provider job | `ProviderState`, `JobState`, or advisory `Interpretation`; request/deadline/model identity retained |

The generic `BrainEvent::historical` constructor covers domain crates that are
downstream of `pete-events` and therefore cannot have direct adapter
implementations here without a dependency cycle.

## Schema evolution

The current schema is `1`. Producers serialize that version. Consumers call
`BrainEvent::decode_json`, which validates v1 and migrates the checked-in v0
fixture at `crates/pete-events/tests/fixtures/brain_event_v0.json`. Unknown
future versions fail closed rather than being silently misread. A future schema
change must add a migration fixture and retain stable IDs, typed references,
clock/calibration epochs, disposition, authority significance, and loss policy.
