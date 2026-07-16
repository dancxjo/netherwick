# 022 Cognitive Roles and Graceful Degradation

PETE is the organism. A host, process, model, or accelerator is replaceable
infrastructure, not PETE's identity and not an authority source. Cognitive work
is described by stable roles and typed capabilities; deployment names remain
compatibility aliases only.

## Stable roles and deployment aliases

| Stable role | Existing deployment alias | Typical placement | Authority |
| --- | --- | --- | --- |
| Body controller | brainstem | Pico W or simulator | autonomic safety and bounded actuation |
| Organism runtime | motherbrain | installed Linux host | canonical `Now`, goals, control orchestration, accepted state |
| Cognitive accelerator | forebrain, higher brain | local process, GPU host, or trusted LAN service | advisory results and model artifacts only |
| Remote advisor | cloud service, operator tool | remote service | advisory claims only |

`brainstem`, `motherbrain`, `forebrain`, and `higher-brain` remain valid protocol,
configuration, crate, and deployment names during migration. New APIs use the
stable role vocabulary. A hostname may locate a provider but never defines its
role, identity, trust, or action authority.

## Authoritative state ownership

The organism runtime owns the current `Now` snapshot, self-model, drive and
goal runtime, control provenance, accepted model registry, and durable ledger
references. The body controller alone owns hard real-time safety and physical
actuation. Cognitive providers may cache inputs, interpret evidence, predict,
plan, review, train, consolidate, or return candidate artifacts. They do not
mutate `Now`, select goals, activate models, acquire a Cockpit lease, or issue
motor commands.

Model transfer and model activation are separate operations. A returned model
candidate is staged and remains inactive until the organism-side promotion
gate explicitly approves activation. Accelerator failure cannot change the
currently accepted model.

## Provider contract

`pete-cognition` defines the role-neutral boundary:

```text
provider descriptor + health + capability versions
                         ↓
bounded typed request + immutable snapshot reference + deadline
                         ↓
deterministic routing by capability, freshness, locality, latency, and trust
                         ↓
typed response + confidence + cost + provenance
                         ↓
accept, reject, fail, or retain as stale evidence
```

Requests carry bounded payloads or references, privacy and persistence policy,
cancellation identity, caller provenance, and an absolute deadline. Providers
publish health, locality, resource class, expected latency, implementation and
model versions, and versioned capabilities. Routing is deterministic and never
consults motor authority.

Responses are tied to the exact request and input snapshot. A result with the
wrong snapshot is rejected. A result completed after its deadline is stale. An
online result for a frame that has since been replaced is also stale and cannot
overwrite newer beliefs. Learned output returns as evidence, a suggestion, or
a candidate artifact; acceptance is an organism-runtime decision.

Raw inputs are capability-specific and bounded. Registering a provider does
not grant it arbitrary access to sensors, the ledger, `Now`, memory, or the
network. In particular, an image-description provider receives the selected
bounded image and snapshot reference, not a general sensor stream.

## Degradation policy

Optional cognition never blocks the organism control tick. Slow work runs in a
background request. A missing, busy, disconnected, timed-out, or incompatible
provider updates provider health in the self-model and the organism continues
with local structured beliefs.

| Optional capability | Degraded behavior |
| --- | --- |
| Scene description | preserve local object, range, hazard, and body beliefs; do not invent a caption |
| Rich language generation | use local scripted or quiet behavior |
| Entity recognition | retain an unknown hypothesis and remembered identity separately |
| Planning or failure review | continue transparent goal evaluation and local behavior sequencing |
| Training or consolidation | retain data/candidates and defer the job |
| Counterfactual prediction | continue with available teacher/runtime estimates |

Capability loss changes competence and affordance availability, not organism
identity. Reconnection or provider restart may restore the capability under a
new host/process identity without creating a new PETE. Safety, body telemetry,
current authority, local goals, and the accepted model remain available or
fail closed according to their own contracts.

## Current vertical slice

Live scene description is the first routed online capability. The real-robot
runner polls it without waiting for provider I/O, records the provider registry
in `Now` integration input, and applies only an accepted response for the still
current frame. `WorldModelUpdater` projects provider capability and health into
`Now.world.self_model.service_state` and the canonical capability view.

The existing `pete-higher-brain` training plane remains intentionally separate
from `pete-cockpit`. Its capability probe can be projected into the same
role-neutral provider descriptor while compatibility names and its explicit
candidate-activation approval remain intact.

## Migration rules

- Prefer `CognitiveAccelerator` and `AcceleratorCapabilities` in new code.
- Keep wire/config aliases until their protocols are deliberately versioned.
- Add a capability implementation by registering a provider; do not add a
  host-specific branch to the organism runtime.
- Project provider health through the world-model updater; do not create a
  second service-status truth source.
- Keep local behavior complete enough to remain safe and coherent without any
  optional provider.
