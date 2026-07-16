# Changelog

All notable changes are grouped by date.

## Unreleased

### Auto-sync (2026-07-15)

- Register four additional `tongues` workspace packages in the lockfile.
- Expand `tongues-tts`'s locked dependency set for model and pronunciation support.
- Add the older `indicatif`/`console` stack required by the new workspace packages.
- Add plotting support through `textplots` and `drawille`.
- Record dependency metadata updates for `burn-core`, `ort`, and `rgb`.
- Disambiguate coexisting versions of `colored`, `console`, and `indicatif`.

### Added

- Add a provenance-backed semantic relation graph to the canonical world model,
  grounding charger, obstacle, behavior, skill, goal, drive, and outcome
  meanings with contextual confidence and contradiction tracking.
- Expose semantic evidence observations and graph queries, including
  charger explanations and causal-grounding safeguards that downgrade
  unsupported causal claims to predictions.
- Surface charger semantic explanations, supporting relation IDs, and grounded
  energy-meaning confidence in goal evaluations and affordances.
- Project canonical semantic relations into durable graph-memory entities and
  edges, preserving grounding metadata and confidence for recall.
- Ground charger approach, docking, and withdrawal predictions in observed
  runtime action outcomes, retaining their evidence provenance across ticks.
- Add a safety-gated sleep lifecycle that quiesces deliberative control, plans
  bounded consolidation, replay, training, and evaluation work, defers
  unavailable accelerator work, and wakes for operator, safety, power, or
  body-link events without automatically promoting candidates.

### Changed

- Document the grounded semantic graph and safety-gated sleep lifecycle,
  including their authority boundaries, artifact contracts, and replacement
  points.
- Removed retired behavior configuration spellings (`mode`, `hardcoded_on_error`, and `stop_on_error`) and compatibility type aliases; configurations must use `regime` and the canonical fallback values.

### Fixed

- Preserve context-distinct semantic relations through graph-memory
  deduplication and Neo4j persistence by carrying `SemanticRelationId` as the
  stable edge identity instead of collapsing edges by triple alone.
- Preserve lifecycle telemetry when velocity and heartbeat commands coalesce:
  smoothed velocity refreshes transfer the active command ID without restarting
  the motor, and every replaced accepted command receives a terminal event.
- Enforce sleep power and thermal budgets on every work item: candidate
  training/evaluation require a stable powered dock, while a rising thermal
  limit interrupts sleep before subsequent maintenance can run.
- Make sleep-session admission edge-triggered and consume completed input
  evidence exactly once, preventing continuous fatigue, a held operator request,
  or already-consolidated work from immediately starting duplicate sessions.
- Add regression coverage and lifecycle documentation for consumed deferred
  references, fatigue recovery, newly observed failures, and held/released
  operator sleep requests.
- Keep every E-stop latched through bump/contact recovery unless an operator
  explicitly clears it; possessor recovery no longer infers E-stop provenance
  from event timing or clears an E-stop while releasing the bump latch.
- Keep Pico-W cockpit operator-control refresh aligned with the granted lease,
  prevent concurrent browser handshakes, and retry control acquisition after
  transient session or lease failures.
- Use `portable-atomic` for brainstem registry and session counters in the default RP2040 build as well as Pico-W builds.
- Pin the Pico-W firmware's `fixed` dependency to the Rust-1.92-compatible 1.30 release so the documented embedded build remains reproducible.

### Ready

- Add typed temporal clock domains, timed predictions, and deterministic charging, conversation, recovery, exploration, and task episodes to the canonical world model.
- Add persistent uncertain people, relationship, presence, identity, and interaction beliefs shared by goals and durable memory without granting social trust any motor authority.
- Add explicit epistemic questions and information-gathering affordances whose progress is measured by actual uncertainty reduction, including charger-seeking inquiry that preserves goal commitment.
- Persist temporal, social, and epistemic snapshots in memory and expose social and episode structure as queryable durable graph records.
- Make a bump-triggered contact withdrawal a bounded brainstem-local reflex: it continues across authority changes, records start and terminal safety events, and remains preemptible by stronger safety conditions.
- Expose the contact-withdrawal lifecycle through typed cockpit events and mirror its bounded, authority-independent behavior in the simulator.
- Route selected goal affordances through typed, bounded possessor skills, with target-based turn, approach, docking, search, and retreat requests that report completion, timeout, unavailable-target, authority-loss, and safety-preemption outcomes.
- Feed possessor-skill progress and terminal outcomes back to the originating goal, including explicit progress expectations and failure pressure without discarding the active goal.

### Ongoing

- Migration of repository automation from shell-based `Justfile` recipes and helper scripts to the Rust `xtask` command layer. This includes BOOTSEL mounting, synchronization, hardware setup, and training workflows; the migration is still being reviewed and is not yet a completed release change.
- Migration of face, object, and voice features to `VectorArtifact`-only schema fields. Downstream runtime, training, and event consumers still require coordinated updates, so this is not yet a completed release change.

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
