# Changelog

All notable changes are grouped by date.

## Unreleased

- Keep this section for post-merge notes before the next release boundary.

## 2026-07-15

### Added

- Introduced canonical self-model integration for goal selection:
  `Now.world.self_model` now carries organism/body/capability/authority/motivation/action-state identity in a typed model.
- Added capability-aware action gating in the goal evaluator:
  behaviors now require specific capabilities (`actuator:drive`, `actuator:speaker`, `sensor:vision`) and are rejected with explicit reasons when unavailable.
- Added a capability registry in `pete-now` with availability (`available`, `degraded`, `unavailable`, `unknown`) and typed IDs.
- Added canonical self-model serialization/tests and documented the new model contract.

### Changed

- The system now uses `goal` as the default action selector in:
  - `Justfile` virtual-live startup
  - `pete-tools` CLI simulation/eval defaults (`sim`, `eval-scenario`)
  - live server status default mode
- LLM prompts now include a rendered canonical self-model context so prompts reflect current capability/state boundaries.
- Runtime `Now` updates now include richer self-model context:
  registered goals/behaviors, registered drive summaries, commitment age/progress, uncertainty/failure pressure,
  and active control provenance.
- Goal architecture now emits and tests a stronger capability-aware failure path for action rejection.

### Fixed

- Missing capabilities now correctly propagate to both planning and language reasoning paths, preventing unsafe/invalid action assumptions.
- Service capability handling now handles higher-brain availability cleanly (e.g., richer cognition disabled state is reflected as unavailable when appropriate).

### Documentation

- Added `docs/021-self-model.md` describing the self-model schema and update/consumer rules.
- Updated architecture docs (`001`, `019`, `020`) to describe self-model as canonical PETE state and the new default behavior.

## 2026-07-15 (prior)

- Persisted `Now.world` as canonical belief state.
- Introduced goal-driven arbitration with a persistent world model pipeline.

## 2026-07-14

- Improved NEAT species persistence and selection workflows.
- Added training/evaluation scenario support and model promotion gates in locomotion training.
