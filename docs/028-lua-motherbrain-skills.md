# Runtime-loaded Lua motherbrain skills

Netherwick uses embedded Lua 5.4 for deterministic motherbrain skill programs.
Lua is small, designed to be embedded, and its cooperative coroutines let a
normal function retain its call stack while a bodily operation is in flight.
Lua owns semantic sequencing; Rust owns numerical controllers, primitive
renewal, possession, resource scheduling, and safety validation. Brainstem
invariants and reflexes remain outside this runtime.

## Execution model

There is exactly one foreground top-level invocation. It includes every nested
Lua call and every child created by `together`. Competing goals remain pending
until the conductor explicitly replaces the foreground commitment.

Only one Lua coroutine executes instructions at an instant. Several organs can
still be active: locomotion, gaze, and voice commands already dispatched to
disjoint resources continue concurrently while the scheduler resumes their
child coroutines one at a time.

Host operations look synchronous:

```lua
function inspectObject(object)
    face(object)
    approach(object)
    observe(object)
end
```

Internally each call acquires its resource, dispatches bounded work, suspends
the coroutine, and resumes it with a value or typed error. Authors do not use
coroutines, yields, callbacks, locks, Cockpit sessions, or command renewal.

## Files, identity, and reload

The default directory is `skills/motherbrain`; set
`PETE_MOTHERBRAIN_SKILL_DIR` to use another directory. Every `.lua` file is
loaded. A file named `reachForFood.lua` must export `reachForFood`; its ID is
`motherbrain.reachForFood`. The implementation version is the SHA-256 of its
source. Status records the path, hash, load time, and Lua/mlua versions. Helper
functions need no metadata.

The whole directory is validated in a fresh sandbox before activation. A
syntax error, missing export, or validation failure leaves the prior set
active. A valid reload is atomic and affects subsequent invocations. A running
invocation retains its exact VM and source generation, so nested calls never
mix versions.

Lua stacks are intentionally not serialized. On shutdown or restart, owned
commands stop or expire at their short TTL, the interruption is recorded,
`Now` is rebuilt, and the conductor chooses again.

## Resources and `together`

Operations acquire resources implicitly:

| Resource | Operations |
| --- | --- |
| `locomotion` | stop, face/turn, drive, approach, bearing/heading/wall control, dock motion, bounded retreat |
| `gaze` | scan, lookAt |
| `manipulator` | grasp, release, bringToMouth, chew, swallow |
| `voice` | say, playFeedback |
| `body_mode` | undock and configured body-mode transitions |

Read-only `Now` queries and observations do not take an exclusive organ.
Resources release on every terminal path. A child awaiting an exclusive
operation cannot acquire a second resource, preventing lock-order deadlocks.

`together` creates child coroutines in argument order:

```lua
together(
    function() approach(food) end,
    function() lookAt(food) end,
    function() say("I found food.") end
)
```

Disjoint organs overlap. Same-resource children queue in child order without
polling. The result is an array in argument order; each entry is the array of
values returned by that child. It is fail-fast: the original typed failure
propagates, sibling commands stop, and sibling resources release. Parent
cancellation and Brainstem preemption use the same unwind path. Nested
`together` is supported.

## `Now`, progress, and provenance

Each activation sees one internally consistent read-only `Now` snapshot. A
later resumption may see a newer snapshot. Stable entity userdata comes from
`nearestVisible` and `visible`, then can be queried with `distanceTo` and
`bearingTo`. Other queries include `contactActive`, `cliffActive`,
`cliffIsClear`, and `charging`.

`observe`, `hypothesize`, `remember`, and `reportProgress` are explicit host
paths. They retain skill provenance rather than mutating `Now`. The originating
goal ID, progress metric, and baseline stay attached to the invocation.
Closed-loop controllers report raw measurements; the runtime normalizes
decreasing metrics such as bearing error and target distance. Author-reported
progress is bounded to `0..1`. The conductor consumes `SkillStatus.progress`
when updating commitments and failure pressure.

`Now["motherbrain.skill_execution"]`, the experience ledger, and real-robot
debug output carry a bounded record containing the selected skill and
arguments, source hash, child calls, resource transitions, primitive intents,
progress, Brainstem preemption details, postconditions, outcome, and duration.
Debug status also exposes the current Lua function and bodily operation, held
and waiting resources, children, last yield/resume, and last preemption.

## Authority, preemption, and CAREFUL

Authority remains:

1. Brainstem invariant/reflex
2. E-stop and explicit attended operator authority
3. foreground motherbrain skill
4. resting/default body command

Lua never receives a Cockpit transport or credential. Brainstem events,
authority loss, transport loss, cancellation, timeout, script error, and budget
exhaustion all resume or terminate Lua with typed outcomes and release owned
organs. `try(function)` returns `(true, value)` or `(false, error_table)`;
`require(condition, message)` produces `postcondition_failed`. Typed errors
carry `kind`, broad outcome, operation/resource identity, details, and the
preserved Brainstem event where applicable.

Ordinary Lua recovery cannot enter the broad attended `careful_mode`. Its
`carefully(hazard, function)` form becomes generation-bound `escape_motion`
segments. Rust requires the named hazard to be acknowledged, permits only a
bounded retreat envelope, renews exactly one 250 ms act at a time, and observes
sensors and odometry between acts. Bump escape cannot suppress a cliff; cliff
escape cannot move toward the cliff. Wheel drop, incompatible hazards,
charging, E-stop, authority loss, link loss, and new Brainstem preemption
remain absolute. Broad CAREFUL remains an explicitly attended operator
capability and is never invoked by Lua recovery.

## Sandbox

The VM allowlists table, string, math, and UTF-8 facilities plus the Netherwick
API. It exposes no filesystem, sockets, subprocesses, environment, dynamic
libraries, package loader, arbitrary `require`, `io`, `os`, `debug`, `load`,
`dofile`, `loadfile`, raw clock, unjournaled randomness, pointers, transport,
or credentials. Raw metatable and raw table mutation facilities are removed.

Each invocation has instruction, activation wall-clock, memory, Lua stack,
value-conversion, result, trace, operation-duration, and child-count bounds.
The instruction hook applies to the top-level thread and every child. Infinite
loops and excessive allocation or recursion become `budget_exceeded`; active
organs stop before the outcome returns.

## Vocabulary

```text
stop
face / faceBearing
turn / turnBy / turnToward
drive / driveDistance
approach
followBearing / holdHeading / followWall
scan / lookAt / observe
searchForDockSignal / alignWithDock / verifyCharging / undock
retreatFromContact / retreatFromCliff / releasePersistentBumper
grasp / release / bringToMouth / chew / swallow
say / playFeedback
waitUntil / require
carefully / together / try / blocked
nearestVisible / visible / distanceTo / bearingTo
contactActive / cliffActive / cliffIsClear / charging
reportProgress / hypothesize / remember
```

An operation for an organ Pete does not possess returns
`capability_unavailable`; it is never silently accepted. Closed-loop control,
decoding, authority checks, primitive TTL renewal, and hard real-time
supervision remain Rust or Brainstem responsibilities.

## Writing and testing a skill

Create one file whose stem matches its exported function:

```lua
-- skills/motherbrain/approachCarefully.lua
function approachCarefully(food)
    local ok, result = try(function()
        approach(food)
    end)
    if not ok and result.kind == "contact_withdrawal" then
        observe(food)
        return blocked(result)
    end
    if not ok then error(result) end
    return result
end
```

Save the file; no Rust rebuild is required. Check status for the new source
hash or inspect the reload error while the prior set remains active. Tests can
instantiate `LuaSkillRuntime` with a temporary directory and an `OrganDriver`;
simulation and hardware use that same runtime, scheduler, Lua source, typed
outcomes, and CAREFUL validator.

```sh
cargo test -p pete-skills
cargo test -p pete-runtime possessor_
```

Complete examples live in `skills/motherbrain`: `reachForFood`,
`eatNearestFood`, `inspectObject`, `searchForDock`, `returnToDock`,
`retreatFromCliff`, and `releasePersistentBumper`. The same directory contains
canonical policies for stop/stabilize, target turn and approach, bearing and
heading control, bounded driving, wall following, IR-gradient docking, search,
undocking, bump recovery, and cliff retreat.
