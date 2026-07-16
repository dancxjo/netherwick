# 001 Architecture

The main control flow is sensors and external claims → typed evidence/Features →
belief integration → immutable `Now.world` → homeostatic drives → goal
interpretation → immutable goal evaluations → commitment-aware arbitration →
goal-owned behavior → typed action primitive → autonomic safety → body
actuation → the next `Now`.

`Now` is PETE's canonical current belief-state snapshot. Its typed
`world.self_model` region defines PETE's body, capabilities, ownership,
authority, motivation, active control, continuity, and cognitive-service
availability; see `021-self-model.md`. It is explicitly a
best, uncertainty-carrying hypothesis, not objective truth. The stateful
`WorldModelUpdater` owns identity, freshness, confidence, contradiction,
coordinate-frame, and provenance semantics. Consumers do not build parallel
sensor-derived mini-world-models. See
[020-now-world-model.md](020-now-world-model.md).

The world model also owns bounded temporal context, persistent uncertain social
beliefs, and explicit epistemic questions. Goals consume these shared regions:
they do not infer social identity or temporal meaning privately, and curiosity
is rewarded by measured belief improvement rather than novelty alone. See
[023-temporal-social-epistemic-cognition.md](023-temporal-social-epistemic-cognition.md).

Optional learned or compute-heavy faculties are role-neutral cognitive
providers, not parts of PETE's identity. The organism runtime sends bounded,
deadline-bearing requests and continues its control tick without waiting;
accepted results return as evidence, suggestions, or candidate artifacts.
Provider loss changes self-model capability availability without transferring
authority or changing organism identity. See
[022-cognitive-roles.md](022-cognitive-roles.md).

The legacy `SimpleConductor` remains available for comparison. `goal-shadow`
maintains the complete blackboard and goal state while executing the legacy
choice; `goal` executes the selected goal behavior. See
[019-goal-architecture.md](019-goal-architecture.md).

The canonical durable observation record is [ExperienceInstant](instant.md),
not a separate Sensorium layer. It records the embodied evidence batch and the
structured `Now` used at a tick for replay, memory, and encoding. It is not a
second current world model. `ExperienceLatent` is a compressed learned
representation derived from that record and may supply additional evidence;
it cannot replace or erase structured `Now` beliefs.

The canonical reusable observation unit is [Feature](013-feature-registry.md):
perception, body, memory, language, and prediction subsystems can emit Features
as evidence before belief integration, clustering, memory, graph storage, or
prediction consumes them. A Feature is evidence, not canonical world state.

## Events and Robot Output

Runtime events are the boundary between sensed state changes and behavior
selection. `EventExtractor` turns `Now` state into typed events such as bump,
charging, face recognition, Reign commands, safety vetoes, and robot
initialization. Ordinary event responders may add sensations, impressions,
memory notes, teaching, or drive impulses; they do not select goals or ordinary
actions. Explicit event-script capabilities remain typed replaceable behaviors.

Event scripts are replaceable behaviors, not one-off side effects. A script can emit `Say`, `Chirp`, `Song`, or motion-oriented script actions. Motion primitives still pass through autonomic safety and SafeCockpit. Spoken `Say` output passes through the robot mouth gate and is rendered by the queued Piper/CPAL mouth when available. `Chirp` and `Song` are non-motion Cockpit feedback/song requests.

The current TypeScript script path uses `tsrun` as an embedded, no-Node runtime for compact event behaviors. A TypeScript script receives structured event input and returns JSON-like script actions. This is useful for expressive startup/status/mouth behaviors, but it is still a teacher implementation behind `ReplaceableBehavior`; learned models must be able to shadow it, compare against it, and eventually replace it under the normal behavior promotion rules.
