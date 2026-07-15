# 001 Architecture

The main flow is sensors → Features → `Now` → persistent blackboard world model → homeostatic drives → goal interpretation → immutable goal evaluations → commitment-aware arbitration → goal-owned behavior → typed action primitive → autonomic safety → body actuation → next `Now` → reward/surprise → ledger → training.

The legacy `SimpleConductor` remains available for comparison. `goal-shadow`
maintains the complete blackboard and goal state while executing the legacy
choice; `goal` executes the selected goal behavior. See
[019-goal-architecture.md](019-goal-architecture.md).

The canonical moment representation is [ExperienceInstant](instant.md), not a separate Sensorium layer.

The canonical observation representation is [Feature](013-feature-registry.md): every perception, body, memory, language, and prediction subsystem should be able to emit Features before downstream clustering, binding, constellations, memory, graph storage, or prediction consumes them.

## Events and Robot Output

Runtime events are the boundary between sensed state changes and behavior selection. `EventExtractor` turns `Now` state into typed events such as bump, charging, face recognition, Reign commands, safety vetoes, and robot initialization. Event responders may add sensations, impressions, memory notes, teaching, or proposed actions. Event-script behaviors then map selected events into `EventScriptAction` sequences.

Event scripts are replaceable behaviors, not one-off side effects. A script can emit `Say`, `Chirp`, `Song`, or motion-oriented script actions. Motion primitives still pass through autonomic safety and SafeCockpit. Spoken `Say` output passes through the robot mouth gate and is rendered by the queued Piper/CPAL mouth when available. `Chirp` and `Song` are non-motion Cockpit feedback/song requests.

The current TypeScript script path uses `tsrun` as an embedded, no-Node runtime for compact event behaviors. A TypeScript script receives structured event input and returns JSON-like script actions. This is useful for expressive startup/status/mouth behaviors, but it is still a teacher implementation behind `ReplaceableBehavior`; learned models must be able to shadow it, compare against it, and eventually replace it under the normal behavior promotion rules.
