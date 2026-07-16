# 019 Goal Architecture

PETE's executive is deliberately small:

```text
build the next canonical Now.world snapshot
update drives
interpret the world for each goal
produce immutable goal evaluations
arbitrate with commitment
execute the selected goal
apply safety
actuate
observe progress
```

## Shared canonical world model

The runtime blackboard is the revisioned `WorldModelSnapshot` inside `Now`. The
`WorldModelUpdater` fuses object, range, memory, authority, action-outcome, and
prediction evidence into persistent typed beliefs. A
charger, person, obstacle, landmark, or sound source keeps one identity across
temporary occlusion. Entity bearings are recomputed from persistent world poses
rather than reusing stale camera-relative bearings.

The public goal tick receives only the same immutable `WorldModelSnapshot` plus
typed suggestions. Goal interpretation/evaluation contexts expose neither
`Now`'s raw evidence fields nor sensor-provider types. Goals do not maintain
competing canonical copies of charger or person location. Evidence provenance
and the builder update trace remain attached for replay and later training.

## Homeostatic drives

Each drive records `desired`, `actual`, `predicted`, current and predicted
error, satisfaction, and smoothed activation. The initial drives cover energy,
safety, curiosity, social connection, rest, and certainty. Existing scalar
`DriveSense` fields are projections of these detailed states for compatibility.

An absent LLM opinion is neutral. LLM confidence affects certainty only when an
actual command or critique exists.

## Goals, behaviors, and evaluations

A goal module contains three independently replaceable components:

- an evidence interpreter with explicit state;
- an evaluator that emits an immutable `GoalEvaluation`;
- an executor that chooses among the goal's current affordances.

`GoalEvaluation` separates `Motivation { activation, urgency, satisfaction }`
from `Competence { confidence, affordances }`. An affordance records
availability, expected reward, expected progress, risk, energy cost, duration,
target, primitive, and provenance.

The initial registry contains `SeekCharger`, `EscapeDanger`, `Explore`,
`Socialize`, `Rest`, `Investigate`, and `FollowTask`. Adding a goal changes its
module and registry, not the arbiter.

## Commitment and progress

The arbiter reads evaluations without modifying them. The incumbent receives a
0.10 persistence bonus. A challenger pays a 0.15 switching cost and a 750 ms
minimum dwell; urgency reduces the cost and dwell but is never added to
activation. Satisfaction, completion, failure, or loss of every safe affordance
releases the incumbent immediately.

Commitment belongs to a goal. `SeekCharger` can search, turn, approach, and dock
without causing goal oscillation.

Every executed affordance predicts progress. `GoalRuntimeState` tracks elapsed
time, failed attempts, recent progress, confidence trend, and computational
frustration. Progress failure becomes shared stalled-goal evidence rather than
a hard-coded transition.

## Reign and safety

Reign `Assist` is an affordance-matched activation bias. `Suggest` is a weaker
bias. `Direct` remains an immediate override, and every resulting motor command
still passes through autonomic safety.

Safety is a veto, not a goal selector. Physical charging, wheel drop, cliff,
stale sensors, invalid commands, disarm, and terminal battery conditions remain
imperative. Contact blocks forward motion while preserving reverse/turn escape.
Critical battery permits bounded motion owned by `SeekCharger`.

## Running and inspecting

The CLI and virtual-live defaults now execute `goal`. Use
`--action-selector goal-shadow` to compare without changing execution, or
`--action-selector baseline` to run the retained legacy conductor explicitly.
Each `Now` frame contains:

- `goal_system`: world snapshot, detailed drives, interpretations, immutable
  evaluations, selection, and behavior;
- `action_selector`: executed or shadow goal, behavior, switch status,
  commitment retention, and selection reason.

Scenario reports aggregate goal switches, commitment-retained ticks, behavior
transitions, mean goal dwell, and goal/behavior histograms.

The complete belief contract is [020-now-world-model.md](020-now-world-model.md).
Goal use of shared temporal context, social identity, and information-gathering
questions is specified in
[023-temporal-social-epistemic-cognition.md](023-temporal-social-epistemic-cognition.md).
