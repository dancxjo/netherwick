# 023 Temporal, Social, and Epistemic Cognition

Phase C extends the canonical `Now.world` blackboard with three bounded,
inspectable regions. They are coordinated world-model products, not new
conductors and not private goal state.

```text
typed evidence and memory recall
  -> WorldModelUpdater
       -> temporal context and episodes
       -> people, relationships, and interactions
       -> uncertainty questions and information-gathering affordances
  -> drives and goal-specific interpretation
  -> ordinary goal arbitration and safety
```

Snapshots are immutable for a tick. Their persistent builders carry only the
state needed to produce the next snapshot; durable history belongs in the
ledger and memory stores.

## Temporal cognition

Time values carry a `ClockDomain`. Monotonic runtime time, wall-clock time,
event occurrence time, observation time, recall time, replay position, and
predicted horizons are not interchangeable. Durations and commitment ages use
monotonic time. Delayed evidence retains both its event time and the later time
at which PETE observed it. Replay position cannot masquerade as live elapsed
time, and a wall-clock correction cannot shorten an active goal.

`TemporalContext` contains the current typed timestamps, bounded active and
recently completed episodes, ongoing durations, current temporal beliefs, and
pending expectations. `TemporalIntegrator` deterministically forms charging,
conversation, recovery, exploration, and task episodes. Episodes carry typed
intervals, participants, active goals, significant events, predecessors, and a
closure reason. Completed episodes are written to typed memory records and the
durable graph rather than accumulating without bound inside `Now`.

Predictions use `TimedPrediction` or `PendingTemporalExpectation` with a
`Predicted` interval. A prediction is therefore a claim about what may happen
during a future horizon, not a timeless scalar.

## Social world model

`SocialWorldSnapshot` owns persistent, uncertain beliefs about people,
relationships, and current/recent interactions. A `PersonModel` separates:

- identity hypotheses and contradictions;
- current presence, location, and attention;
- familiarity and relationship references;
- communication preferences and interaction history.

Presence expires independently of identity. Losing sight of Alex removes
current visual affordances and eventually closes the interaction; it does not
erase the durable hypothesis that Alex exists. Recalled people re-enter as
historical identity evidence with `present = false`.

Face, voice, text self-identification, direct observation, and memory are
distinct identity modalities with provenance. Repetition and independent
modalities can strengthen an identity. Conflicting biometric or claimed
identities remain explicit until evidence resolves them. A generic,
high-confidence `person` detection is strong presence evidence but weak
identity evidence.

Relationships may represent familiarity, trust, affiliation, commitments, and
scoped social roles. They never grant Reign or motor authority. Authority
continues to come from possession/session evidence and the canonical
self-model. Requests are attributed only when an interaction has an
unambiguous participant.

`Socialize` consumes this shared model. It may greet a known person by name or
ask an uncertain person how to address them; it does not build a second person
database from raw detections.

Social records can contain personal data. Default reports should expose stable
ids, confidence, freshness, and decision-relevant summaries, not raw audio,
images, embeddings, or unrestricted transcripts. Retention and export policy
belongs to the memory/ledger boundary.

## Epistemic cognition

Uncertainty becomes actionable only when it is represented as an
`EpistemicQuestion`: a stable id, affected belief, alternatives, current and
initial uncertainty, importance, provenance, attempts, and expiry. Initial
question families cover charger identity or bearing, clearance, person
identity, sound direction, place familiarity, skill failure, and uncertain
predicted danger.

Every question may publish several `EpistemicAffordance`s. In addition to the
ordinary behavior fields they expose expected information gain, expected
uncertainty after acting, action and energy cost, risk, duration, confidence,
and the affected belief. Selection favors targeted, safe evidence gathering
over random motion.

`Investigate` is the general inquiry goal and can scan, inspect, listen, ask,
compare a prediction, or stop and observe. Inquiry is also compositional:
`SeekCharger` can turn for evidence, inspect a hypothesis, or search
systematically while retaining charger commitment. Confidence changes the
method; uncertainty does not erase the underlying homeostatic need.

The expected gain is diagnostic, not reward by fiat. On the next world-model
update PETE measures actual reduction in the named question's uncertainty.
Curiosity drive and progress consume that structured change. Tried methods are
discounted; three distinct methods with no gain mark a question temporarily
unanswerable so the robot changes strategy instead of repeating a sterile
loop.

Reports record active questions, competing hypotheses, affordance utilities,
attempted methods, expected and observed information gain, resolution, and
unanswerable outcomes. The core metrics are total information gain, resolved
questions, unresolved questions, and repeated-question count.

## Replay and replacement

A deterministic evidence sequence and deterministic teacher implementations
produce identical temporal, social, and epistemic snapshots. Memory records
retain all three typed regions; social people, interactions, relationships, and
temporal episodes also become queryable graph entities and edges.

Future learned components may replace episode segmentation, identity scoring,
question generation, information-gain estimation, or affordance utility. They
must publish the same bounded typed outputs with provenance. They do not gain
goal-selection, Reign, safety, or world-model mutation authority merely by
being learned.
