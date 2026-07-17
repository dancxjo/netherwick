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
time, bounded attempt history, recent progress and its trend, the last measured
progress time, confidence trend, and computational frustration. Progress
failure becomes shared stalled-goal evidence rather than a hard-coded
transition. Progress is `Option`-valued: stale or missing target beliefs produce
unknown progress and do not count as zero or as a failed attempt. Autonomic
safety preemption is recorded as unmeasurable rather than credited to the
possessor skill it interrupted.

Each `GoalCycle` also records a replayable `GoalProgressReport` per registered
goal. The report retains the behavior- or skill-level expectation—including its
metric, baseline, horizon, and tolerance—latest observation, previous and
selected behaviors, bounded failure count, strategy-failure pressure, and a
typed response: started, retained, changed, help requested, or abandoned. Its
reason explains the expected/observed comparison that led to the response.
Skill-local progress and goal progress remain distinct: for example, charger
search can expose frontier coverage to the skill while the owning goal measures
whether charger uncertainty actually falls.

Each possessor `SkillStatus` identifies one execution with a stable
`execution_id`. `attempts` advances only when that intention begins a new
execution; the 100 ms command renewals used within an execution accumulate in
`dispatch_count` instead. A terminal execution is admitted to goal progress
exactly once, then the possessor may start a genuine retry with a new execution
id. Cached terminal observations therefore cannot manufacture failures or
prevent retry.

The skill runtime is also the architectural motor boundary. Deterministic
motherbrain skills such as bearing/heading control, approach, timed drive/turn,
scan, wall-follow, dock alignment, wiggle, and post-contact escape renew only
short-lived Brainstem primitives. Post-contact escape specifically renews
generation-bound 250 ms segments and observes body motion between them; it
never opens CAREFUL or reports an unsent phase. They do not depend on an LLM. The Brainstem
owns bounded `cmd_vel`/direct/arc output and may preempt it with immutable
safety or the contact-withdrawal reflex; the resulting typed interruption is
fed back into skill and goal progress.

`SeekCharger` escalates repeated low-confidence search failure to a help
request, then abandons only at the bounded failure limit. `Explore` publishes
independent random-walk, wall-follow, frontier-follow, and novelty-inspection
strategies; rising strategy-failure pressure changes the behavior while the
Explore goal remains committed.

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
transitions, mean goal dwell, goal/behavior histograms, mean measured progress,
no-progress dwell, bounded failures, within-goal strategy changes, help
requests, unmeasurable progress, and false-stall rate.
The serialized `goal_system.progress` records explain individual strategy
retention, change, help, and abandonment decisions and can be consumed by
ledger replay or future learned replacements.

The complete belief contract is [020-now-world-model.md](020-now-world-model.md).
Goal use of shared temporal context, social identity, and information-gathering
questions is specified in
[023-temporal-social-epistemic-cognition.md](023-temporal-social-epistemic-cognition.md).
Goals may also consume bounded grounded-semantic explanations and relation ids
without teaching the arbiter semantic policy; see
[024-sleep-and-grounded-semantics.md](024-sleep-and-grounded-semantics.md).
