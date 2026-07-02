# 004 Behavior Replacement

Important behaviors live behind `Replaceable` wrappers with explicit operating modes: hardcoded use, shadow training, shadow inference, model inference, train+infer, compare, and fallback.

## Contract

Every behavior that might later be learned should have:

- a stable behavior id,
- typed input and output structs,
- one or more hardcoded teacher implementations,
- optional model implementations,
- a fallback policy,
- behavior-run records that can be written into ledger frames and UI state.

The hardcoded teacher is allowed to be procedural Rust, TypeScript running through `tsrun`, or another deterministic implementation, but it must still sit behind the same `ReplaceableBehavior<I, O>` wrapper. Direct calls from the CLI or robot runner should be limited to gates and renderers, not policy decisions.

## Event Scripts

Event scripts are behavior nodes whose input is an event-specific context and whose output is an `EventScriptOutput` sequence. Examples include:

- `event_robot_initialized`: receives startup mode, body, battery, sensor, ledger, dashboard, and capture metadata; emits bring-up `Song`, `Chirp`, and `Say` actions.
- `event_bump`: receives bumper state; uses host-provided TypeScript randomness to choose a small lament (`Uh-oh`, `Oh no!`, `Oopsie!`, `Oh dear!`, or a mournful tune) before emitting recovery script actions.
- `event_face_detected`: receives face familiarity/person context; emits greeting speech.

`event_robot_initialized` currently uses a TypeScript teacher script evaluated by `tsrun`. That implementation is intentionally just the teacher surface. The behavior id, record shape, shadow model slot, and fallback policy exist so a learned model can shadow-train from the TypeScript output, run in shadow inference, compare outputs, and only later enter model inference if promotion evidence says it should.

## Replacement Rule

For event and mouth behaviors, model replacement must preserve the output boundary:

- learned outputs are still `EventScriptOutput`, not raw audio or arbitrary commands,
- `Say`, `Chirp`, and `Song` remain mouth actions,
- body motion actions remain typed primitives and still pass safety,
- model-controlled modes must be opt-in and observable through behavior-run records.

This keeps cute/status behaviors learnable without letting a generated script or model bypass the robot safety and mouth gates.

Randomness belongs inside the teacher behavior input/output record, not outside the behavior system. If a TypeScript teacher uses randomness, the selected output is still recorded as the teacher sample. Shadow models learn from the emitted `EventScriptOutput`, and model inference remains constrained to the same typed output shape.
