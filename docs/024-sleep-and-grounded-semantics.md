# 024 Sleep, Consolidation, and Grounded Semantics

Phase D adds two related but separately authoritative facilities: a bounded
maintenance lifecycle and a persistent semantic relation graph. Neither is a
new conductor. Sleep schedules work; the semantic graph supplies inspectable
meaning evidence. Ordinary goal arbitration and autonomic safety retain their
existing authority.

```text
immutable Now.world + completed episodes + action outcomes
        |                                  |
        v                                  v
sleep eligibility and work plan     semantic graph integration
        |                                  |
        v                                  v
local consolidation / replay        goal-specific semantic queries
optional candidate training         affordance and explanation traces
        |                                  |
        +----------> next fresh Now <------+
```

## Sleep is a lifecycle, not a Rest behavior

`Rest` is an ordinary homeostatic goal. Sleep is a typed runtime lifecycle:
`Awake`, `Preparing`, `Quiescent`, `Consolidating`, `Training`, `Evaluating`,
`Finalizing`, `Waking`, or `Interrupted`. Entry requires a stopped body, stable
body communication, interruptible possessor work, no Direct Reign, no urgent
safety or homeostatic condition, and an eligible fatigue, operator, charging,
episode, or deferred-work trigger.

Preparing for sleep releases the current goal commitment and clears its
pending progress expectation. While quiescent, the possessor layer publishes
no goal behavior or skill request and requests `Stop`. Brainstem reflexes and
the ordinary safety filter remain active. Direct control, cliffs, wheel drop,
contact, body-link loss, critical power, external-power loss, important social
cues, and explicit wake requests can interrupt maintenance. Typed wake reasons
are prioritized; safety outranks communication, homeostasis, operator, social,
and routine completion.

Waking never resumes a stale skill. Deliberation is rebuilt against the newly
integrated world-model snapshot for that tick.

## Bounded maintenance and candidate lifecycle

A `SleepWorkItem` declares stable input artifact references, schema versions,
dependencies, estimated wall/CPU/memory/disk/energy/network resources,
locality, cancellation policy, output contract, verification rule, and
promotion policy. The initial deterministic plan:

1. verifies durable state;
2. consolidates completed episode and semantic-relation references without
   replacing source history;
3. replays recent failures with separate historical and replay clock domains;
4. optionally produces a versioned candidate;
5. evaluates it against a fixed teacher baseline.

Deferred maintenance is keyed by stable episode and failure references, not a
recomputed count. A successfully finalized session records those references as
consumed in a bounded controller history; the same charging snapshot therefore
cannot immediately start the same plan again. A higher failure count or a new
episode id is new work. Fatigue-triggered entry uses hysteresis and is re-armed
only after activation falls below 0.65. Operator sleep requests are
edge-triggered, so a held request cannot create a sleep/reawaken loop.

Local work remains useful without an accelerator. Accelerator-preferred work
is deferred with an explicit reason. Replay artifacts cannot enter live `Now`
as current observation. Candidate artifacts contain role/interface version,
data and configuration references, seed, metrics, warnings, known failure
slices, fallback artifact, and promotion policy. Sleep has no authority to
activate a candidate; the first transparent teacher intentionally evaluates
below its baseline and leaves the accepted implementation unchanged.

Resource permission is evaluated again before each work item; entry eligibility
does not authorize the entire session. Candidate training and evaluation
declare an external-power requirement and are deferred unless PETE is both
charging and stably docked. Thermal state is also live: exceeding the session's
thermal ceiling produces a typed high-priority wake interruption before later
work executes.

Every tick serializes the normalized lifecycle snapshot under the `sleep`
extension of the durable `ExperienceFrame`. Completed reports identify work,
artifacts, deferrals/failures, wake reason, and the fresh-world/no-stale-skill
invariants.

## Grounded semantic graph

`Now.world.semantic` is a canonical snapshot of persistent, typed semantic
relations. Nodes refer to entities, people, places, actions, skills, behaviors,
goals, drives, outcomes, properties, concepts, and episodes. Relations include
affordance, requirement, goal/drive relevance, prediction, causal, identity,
spatial, episodic, naming, ownership, and purpose predicates.

Each relation carries confidence, context and conditions, supporting and
contradicting evidence, grounding kinds, learned/confirmed times, and a status.
Contradictions reduce or invalidate a relation without relying on source order.
Identical evidence is idempotent: merely advancing a tick cannot strengthen a
belief. Human and LLM claims retain their provenance and begin as evidence,
not fact or authority.

`Causes` requires an intervention, action outcome, confirmed prediction,
simulator teacher, or configured mechanism. Temporal order and co-occurrence
alone are retained as `Predicts`; they cannot manufacture causality.

## Charger and obstacle vertical slice

The initial stable vocabulary expresses that chargers restore/satisfy energy,
help `SeekCharger`, conditionally afford approach and docking, and require
bounded skills. Current availability remains separate:

- a far charger is still a charger but does not afford docking now;
- stale or low-confidence localization selects inspection/search rather than
  direct locomotion;
- an obstacle can contextually block the current route without erasing charger
  identity or purpose;
- successful approach, docking, and withdrawal outcomes add independent
  prediction evidence after canonical-world-model comparison and integration;
- outcome evidence is admitted only when the autonomous goal primitive was
  actually executed unchanged through final selection and safety, and targeted
  progress retains the same stable entity identity across the transition;
- a failed charger hypothesis can be contradicted without deleting its audit
  trail.

Goal evaluations expose a bounded semantic explanation and affordances expose
the exact relation ids they used. The arbiter remains semantic-agnostic: it
still consumes immutable evaluations only. Memory projects semantic nodes and
relations into durable graph records with the complete typed relation in the
edge payload.

## Replacement boundary and second passes

This phase is deliberately a transparent first pass. Future implementations
may replace sleep eligibility estimation, work planning, consolidation,
replay selection, training, evaluation, relation scoring, causal discovery,
or semantic querying behind the same contracts. Later passes should add real
artifact executors and broader grounded vocabulary without granting learned
components world-model mutation, goal selection, model-promotion, Reign, or
safety authority.
