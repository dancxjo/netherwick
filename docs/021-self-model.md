# 021 Canonical Self-Model

PETE's self-model is the typed `Now.world.self_model` region. It is a computational belief model of one persistent embodied agent, not a consciousness claim or prose persona. Like every other `Now` belief, it is immutable for a tick and carries confidence, freshness, provenance, and contradictions where applicable.

## Identity boundaries

- `OrganismId` identifies PETE across body repairs, brainstem boots, higher-brain hosts, processes, and control sessions.
- `BodyId` and body implementation/version identify the current physical body.
- `BrainstemDeviceId` and `BootId` identify the controller and one boot. A changed boot invalidates current authority until a fresh lease is established.
- `HostId` and `ProcessId` identify replaceable cognitive-service infrastructure; neither is PETE's identity.
- `SessionId` identifies an interaction or control session, never the organism.

Contradictory boot, body, or session claims remain explicit and fail closed for authority. Host loss removes host-supplied capabilities without changing `OrganismId`.
Provider roles, health, versions, locality, and resource class use the shared
`pete-cognition` registry contract described in
[022-cognitive-roles.md](022-cognitive-roles.md); service host/process fields
remain infrastructure identities.

## Typed regions

`SelfModelSnapshot` contains:

- `body`: identity, pose, envelope, energy, charging, health, faults, and known moved/tilted/blocked/carried state;
- `capabilities`: current sensor, actuator, goal, behavior, skill, and cognitive-service availability with dependencies, confidence, performance summaries, and unavailable reasons;
- `agency`: Reign source/mode, possession, lease/session, armed/stopped/moving state, pending direct control, and authority conflicts;
- `motivation`: bounded drive state, selected goal, commitment age, progress, uncertainty, and strategy-failure pressure;
- `active_control`: goal/behavior/skill/action plus autonomous, operator, autonomic-reflex, or safety-veto provenance;
- `continuity`: bounded references to episodes, experiences, relationships, places, self-actions, outcomes, and capability changes;
- `service_state`: replaceable cognitive services and their host/process
  availability, with request occupancy reported separately as `busy` so a
  healthy in-flight service remains usable.

Compatibility body projections remain temporarily for existing goal and safety consumers. They are views of the same integrated evidence, not a second self-model.

## Update and consumer rules

The world-model updater is the only writer. It integrates body telemetry, brainstem possession, Reign, capability registration and health, goal runtime, action/safety outcomes, continuity references, and service discovery into the next snapshot.

Capabilities fail unavailable when action-relevant health or freshness is missing. Remembered entities persist when a camera disappears, but do not imply current vision. A capability never grants authority by itself.

Goals reject affordances whose required capability is unavailable. Skills receive the same body and capability bounds. Language receives a bounded view listing available and unavailable capabilities and control provenance, and must not invent capabilities. Debugging and replay serialize this same region to explain what PETE thinks it is doing, who controls it, what is unavailable, and why action stopped or failed.

Recalled experiences remain references in `continuity`; recall never turns history into current body observation.
