# Changelog

All notable changes are grouped by date.

## Unreleased

### Cognitive provider health follow-up (2026-07-15)

- Keep optional scene providers degraded while a request is in flight, mark them available only after a successful response, and record deadline failures as degraded health.
- Serialize cognitive role, locality, and resource-class fields with their stable snake-case identifiers in self-model service beliefs.

- Added a `sup` target in `Justfile` as a short alias for `./scripts/codex-sync.sh`.
- Refactored `scripts/codex-sync.sh` to cache both short and porcelain git status once per run.
- In clean-sync mode, `codex-sync.sh` now prints branch/status context and syncs from origin with `git pull --ff-only`.
- In clean-sync mode, local ahead status now triggers an automatic `git push` after successful pull.
- Added temporary-file based output capture in `codex-sync.sh` for robust summary propagation from nested `codex` execution.
- Updated the embedded codex instructions in `scripts/codex-sync.sh` to enforce an expanded repo-sync workflow (status inspection, changelog update, conflict-aware pull/push, and final summary format).
- Refined `codex-sync` workflow instructions to preserve in-progress/Ongoing edits (e.g., TODO, FIXME, WIP, debug scaffolding) and commit only clearly ready, substantial chunks.

### Auto-sync (2026-07-16)

- Added a new `pete-cognition` crate with common cognitive provider contracts, request/response types, and router abstractions.
- Wired `pete-cognition` into the workspace and into `pete-llm`/`pete-now` through path dependencies.
- Reworked live image enrichment in `pete-llm` around an asynchronous `LiveImageCognition` pipeline with request routing and registry snapshots.
- Implemented an ollama-backed `CognitiveProvider` for scene-description requests and embeddings, producing structured scene/caption vectors.
- Extended runtime `Now` cognition bookkeeping to include provider registry snapshots, last scene response, and vision enrichment error metadata.
- Extended cognitive service belief state to persist provider metadata and capabilities in `pete-now` world summaries.

### Cognitive provider refinement (2026-07-15)

- Reject completed scene-description responses when their source frame has been replaced, so stale enrichment cannot overwrite newer perception.
- Project cognitive provider health and capabilities into the canonical self-model service state while keeping organism identity independent of provider restarts.
- Added role-neutral accelerator capability descriptors and documentation for cognitive roles, bounded requests, and graceful local fallback.
- Added focused coverage for unavailable or slow optional cognition and provider disconnect/restart behavior.

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
