# Repo audit and steal plan for pete

## Executive summary

`pete` should steal ideas asymmetrically, not evenly.

`daringsby` has the richest real memory/graph/vector machinery and the best evidence for "memory returned as sensation", but much of it is overgrown and mixed with Pete-specific tooling. Steal the storage patterns and a few graph/query seams, not the whole psyche loop.

`listenbury` has the strongest live audio runtime organs: CPAL capture, frame/ring buffering, VAD seams, self-hearing suppression, playback tracing, and transcript/timing formats. Steal those for robot ears and mouth runtime, but do not inherit its monolithic CLI orchestration as your runtime architecture.

`mortar-sea` has the cleanest cognitive vocabulary. Its `Sensation -> Impression -> Experience -> Memory -> Sensation` doctrine is the right conceptual spine for `pete`, and its explicit provenance and timestamp semantics are better than `daringsby`'s older impression/stimulus model.

`tongues` is useful mainly as a workspace/training/checkpoint pattern source. Its Burn setup is practical and minimal in `common-phone`; its artifact manifest and checkpoint conventions are worth copying. It is not yet a good source for embodied model-runtime boundaries.

The main recommendation is:

- keep `pete`'s embodied/control architecture,
- adopt `mortar-sea` terminology for cognition,
- port a narrowed memory subsystem from `daringsby`,
- port live audio/runtime seams from `listenbury`,
- copy `tongues`' training artifact discipline,
- and keep the hardcoded-to-model replacement wrapper as a first-class architecture rule.

## Key architectural decisions to preserve

- Preserve the doctrine that memory recall is not only prompt context. It must re-enter runtime state as sensed material.
- Preserve explicit provenance on all derived and recalled items.
- Preserve a strict distinction between immediate noticing and later meaning-making.
- Preserve a separate mouth/speech gate instead of letting the LLM directly emit raw audio actions.
- Preserve simulator-first runtime and replayable traces as training fuel.
- Preserve a hot-path / cold-path split: reflexes and body safety on the hot path, memory indexing and slow LLM work off the hot path.
- Preserve hardcoded behavior wrappers that support direct execution, shadow training, shadow inference, compare, replace, and fallback.

## Key architectural decisions to change

- Change `daringsby`'s old `Stimulus`/generic `Impression<T>` style to a stricter `Sensation`/`Impression`/`Experience` split.
- Change `daringsby`'s memory shape from "store arbitrary impression summaries" to "store typed experiences plus auxiliary sensor artifacts and embeddings".
- Change `listenbury`'s command-oriented live loop organization into crates with reusable services; the current CLI files are too monolithic.
- Change `mortar-sea`'s "recall all on explicit pull" reference implementation into targeted retrieval over ledger slices, embedding search, and typed memory senses.
- Change `tongues`' model-family sprawl into a very small set of neural-organ crates with shared artifact/checkpoint support.
- Change direct stringly LLM action paths into typed `ActionPrimitive` outputs plus a mouth/body gate.

## Daringsby findings

### Reuse

- `psyche/src/wits/memory.rs`
  Good reuse source for Qdrant and Neo4j client seams, vector collection conventions, graph merge record style, graph timeline queries, face/voice/place/image storage helpers, and graph snapshot tooling.
- `psyche/src/wits/memory.rs:BasicMemory`
  Reuse the pattern, not the exact type: vectorize summary, store graph record, tolerate vector-store failure.
- `psyche/src/wits/sensation_graph_observer.rs`
  Reuse the idea of graph observers that store sensations passively as they happen.
- `pete/src/bin/remember.rs`
  Best real implementation of memory retrieval feeding back into the graph as a new sensation. It is the closest thing here to recall-as-sense that actually works.
- `psyche/src/wits/combobulator.rs`
  Reuse the idea that a current awareness summary can be emitted both as LLM context and as a looped-back sensation.
- `pete/src/bin/will.rs`
  Reuse the constrained function-calling/TypeScript tool pattern only as a temporary operator-facing LLM control surface.
- `pete/Cargo.toml`, `pete/src/web.rs`, `frontend/psychic/*`
  Reuse the server/dashboard idea set: WebSocket event feed, timeline browser, movie/timeline index, graph browser.

### Rewrite

- `psyche/src/wits/will.rs`
  The XML-ish tag parsing and general instruction path are too loose for robot control. Replace with typed action plans and explicit safety-reviewed action primitives.
- `psyche/src/voice.rs`
  Good for sentence streaming to mouth, but too conversation-centric and not safe enough for embodied output control.
- `psyche/src/sensation.rs`
  Rich, but too heterogeneous and `Any`-driven. `pete` should keep typed sensor records and avoid `Box<dyn Any>` as the canonical runtime substrate.
- `pete/src/bin/will.rs::recall`
  The retrieval itself is useful, but returning plain formatted strings to the LLM is only half the design. `pete` should feed retrieval into both LLM brief and numeric/model inputs.
- `psyche/src/wits/memory_wit.rs`
  Avoid the naive "concat summaries every N ticks" memory model. It is too lossy and too untyped.

### Avoid

- `psyche/src/wits/face_memory_wit.rs`
  Too heuristic and stateful-local; it only compares to the previous face vector.
- `psyche/src/wits/voice_memory_wit.rs`
  Same issue as face memory; it is a local familiarity heuristic, not a durable memory system.
- `psyche/src/wits/will.rs::handle_llm_output`
  Do not cargo-cult the generic XML motor invocation parser for robot actions.
- The giant all-in-one `psyche` architecture as a whole
  Too much of it mixes core cognition, graph maintenance, dashboards, tools, and Pete-specific workflows.

### Specific files/types/functions

- Reuse or port:
  `psyche/src/wits/memory.rs::QdrantClient`
- Reuse or port:
  `psyche/src/wits/memory.rs::Neo4jClient`
- Reuse or port:
  `psyche/src/wits/memory.rs::GraphStore`
- Reuse or port:
  `psyche/src/wits/memory.rs::BasicMemory`
- Reuse or port:
  `psyche/src/wits/memory.rs::find_vector_clusters`
- Reuse or port:
  `psyche/src/wits/memory.rs::vector_cluster_items`
- Reuse or port:
  `psyche/src/wits/memory.rs::attach_remembrance`
- Reuse or port:
  `psyche/src/wits/memory.rs::attach_offline_combobulation_summary`
- Reuse or port:
  `pete/src/bin/remember.rs::RememberProcessor::remember`
- Reuse or port:
  `pete/src/bin/will.rs::recall`
- Reuse or port:
  `pete/src/bin/will.rs::store_function_result_sensation`
- Reuse or port:
  `psyche/src/wits/combobulator.rs::digest`
- Reuse or port:
  `psyche/tests/combobulator.rs::bus_backed_digest_loops_summary_back_as_sensation`
- Reuse or port:
  `frontend/psychic/psychic.js`
- Reuse or port:
  `pete/src/bin/timeline.rs`, `pete/src/bin/movie.rs`, `pete/src/bin/psychic.rs`

- Rewrite:
  `psyche/src/wits/will.rs`
- Rewrite:
  `psyche/src/voice.rs`
- Rewrite:
  `psyche/src/psyche.rs`
- Rewrite:
  `psyche/src/sensation.rs`

- Avoid:
  `psyche/src/wits/memory_wit.rs`
- Avoid:
  `psyche/src/wits/face_memory_wit.rs`
- Avoid:
  `psyche/src/wits/voice_memory_wit.rs`

Memory and recall verdict:

- Can it store impressions?
  Yes. `BasicMemory::store` persists impression summaries into Neo4j and stores a memory embedding in Qdrant.
- Can it search/retrieve memories?
  Yes. `QdrantClient::search_vectors` plus graph queries in `pete/src/bin/remember.rs` and `pete/src/bin/will.rs::recall`.
- Can it connect Qdrant hits back to graph nodes?
  Yes. Payloads carry `neo4j_node_id`, and graph-side helper queries reconstruct context via vector ids.
- Can it recall relevant context into an LLM prompt?
  Partly. `pete/src/bin/will.rs::recall` returns formatted text for the Will TypeScript tool path.
- Can it emit first-person summaries?
  Yes. `Combobulator` and remembering produce first-person-ish summaries.
- Can it represent face/voice/place memory?
  Yes. There are explicit face, voice, geolocation, image, image-description, and scene-vector paths.

Cleanest real implementation of memory + recall:

- Storage: `psyche/src/wits/memory.rs`
- Retrieval into a new sensation: `pete/src/bin/remember.rs` plus `Neo4jClient::attach_remembrance`

Which version actually hooked RAG/recall into the agent loop:

- The strongest actual hookup is not generic RAG in the main loop. It is the external remembering loop in `pete/src/bin/remember.rs`, which polls recent sensations, retrieves related memories, and writes a new derived sensation back into the graph.
- `pete/src/bin/will.rs::recall` is a live tool for the Will loop, but it is tool-returned text, not fully integrated recall-as-sensation.

Where direct `<speak>` parsing is implemented:

- There is no strong direct `<speak>` path in `daringsby`. The active control tag is closer to `<take_turn>` in `psyche/src/psyche.rs::extract_tag` and test coverage in `psyche/tests/voice_control.rs`.
- Speech itself is mostly routed through `Mouth::speak` and `Voice`, not a robust `<speak>...</speak>` structural parser.

Which code path most closely resembles “memory returned as sensation”:

- `pete/src/bin/remember.rs` writing remembered output through `Neo4jClient::attach_remembrance`
- `psyche/src/wits/combobulator.rs` looping `CombobulationSummary` back through `Topic::Sensation`

Recommendation for how pete should model Sensation/Impression/Experience:

- Use `mortar-sea`'s explicit typed split as the main ontology.
- `Sensation`: typed sensor or memory input, timestamped, provenance-bearing, machine-readable payload.
- `Impression`: local claim by a faculty about one or more sensations.
- `Experience`: integrated meaning across time, suitable for memory indexing and training labels.
- Do not store freeform impression summaries as the primary durable semantic memory unit; store experiences, with links back to impressions and source sensations.

Recommendation for how pete should implement recall-as-sense:

- Use a real `MemoryRecall` sensation family, not just injected prompt text.
- Retrieval should return:
  a compact LLM brief,
  typed `MemorySense` numeric fields for `Now`,
  and one or more `memory.related_experience`-style sensations with provenance and original timestamps.
- The recall trigger should be policy-driven:
  periodic,
  event-triggered,
  and query-triggered by higher cognition.
- Recalled items should enter both:
  `Now.memory_*` fields for lower models,
  and the LLM/context frame for conscious critique/planning.

## Listenbury findings

### Reuse

- `src/hearing/vad.rs`
  Good backend seam for energy VAD and WebRTC VAD selection.
- `src/hearing/suppression.rs`
  Strong reusable self-hearing suppression logic. This is one of the clearest living organs in the repo.
- `src/audio/ring.rs`
  Good minimal audio-frame ring buffer.
- `src/word/stream.rs`
  Good reusable timed-word and commitment model for speech traces and replay.
- `src/trace/viewer_payload.rs`
  Strong replay/view payload pattern for timelines and live session inspection.
- `src/web/server.rs`
  Useful lightweight web viewer/input arbitration patterns.
- `src/soundscape/pipeline.rs`
  Useful adapter for turning mic/playback/VAD/ASR events into attributed soundscape frames.
- `src/mouth/planner.rs`
  Good boundary planner for chunked speech emission and face/speech separation.
- `src/mouth/tts.rs`
  Good tiny TTS trait seam.

### Rewrite

- `src/cli/commands/live_half_duplex.rs`
  Valuable as a reference implementation, but far too monolithic to reuse directly.
- `src/cli/commands/mic_transcribe.rs`
  Contains reusable capture code, but wrapped in command behavior and web-transcription concerns.
- The whole live loop orchestration
  Should be extracted into `EarRuntime`, `SpeechPlanner`, `MouthRuntime`, `TraceRuntime`, and `InputRouter` services inside `pete`, not left in one command file.

### Avoid

- Application/demo specific browser transcript player packaging as a direct dependency of robot runtime.
- Heavy reuse of all the language-pack, vocoder, or alternate TTS backend machinery for initial `pete`; that scope is too broad.
- The current live command file layout as architectural truth.

### Specific files/types/functions

- Reuse or port:
  `src/hearing/vad.rs::VoiceActivityDetector`
- Reuse or port:
  `src/hearing/vad.rs::create_vad_backend_with_profile`
- Reuse or port:
  `src/hearing/suppression.rs::SelfHearingState`
- Reuse or port:
  `src/hearing/suppression.rs::SpeakerReferenceMask`
- Reuse or port:
  `src/audio/ring.rs::make_audio_ring`
- Reuse or port:
  `src/mouth/tts.rs::TextToSpeech`
- Reuse or port:
  `src/mouth/planner.rs::SyntheticPlanner`
- Reuse or port:
  `src/mouth/planner.rs::MouthSyntheticPlan`
- Reuse or port:
  `src/soundscape/pipeline.rs::SoundscapePipelineAdapter`
- Reuse or port:
  `src/word/stream.rs::TimedWordStream`
- Reuse or port:
  `src/word/stream.rs::WordCommitment`
- Reuse or port:
  `src/trace/viewer_payload.rs::live_trace_jsonl_to_viewer_payload`
- Reuse or port:
  `src/web/server.rs::InputRouter`

- Rewrite:
  `src/cli/commands/live_half_duplex.rs`
- Rewrite:
  `src/cli/commands/mic_transcribe.rs`
- Rewrite:
  `src/web/server.rs` as a crate-level dashboard server instead of an all-purpose app

Audio pipeline reuse plan:

- Steal `AudioFrame`, ring buffering, timed frames, VAD backends, self-hearing suppression, and trace payload formats.
- Extract CPAL capture into a standalone `pete-ear-capture` service from the capture code currently embedded in `mic_transcribe.rs` and `live_half_duplex.rs`.
- Use `SoundscapePipelineAdapter` ideas to attribute mic frames, speaker playback, and ASR output to sources in the robot body.

Voice command / speech output reuse plan:

- Use `SyntheticPlanner` and `MouthSyntheticPlan` as a template for chunked speech output.
- Keep `TextToSpeech` as the narrow mouth boundary.
- Adapt mouth output so `ActionPrimitive::Speak` produces:
  text intent,
  chunk plan,
  audio render,
  speaker playback events,
  and self-hearing events.

Dependencies to copy:

- `cpal`
- `rtrb`
- `webrtc-vad` through the existing feature-gated pattern
- `crossbeam-channel`
- trace/view payload dependencies already used by the viewer path

Things too brittle or too application-specific to reuse:

- the giant feature-gated live CLI command files
- the current browser transcript player packaging as a runtime dependency
- all alternative TTS/vocoder backends for MVP

What pete should steal for EarSense:

- VAD backend seam
- CPAL capture patterns
- frame-ring buffering
- self-hearing suppression
- timed transcript/word stream model

What pete should steal for Speak/Chirp/Speech actions:

- `TextToSpeech`
- chunk planner concepts from `mouth/planner.rs`
- playback tracing and attribution

Is there reusable code for microphone capture:

- Yes, but it lives inside `src/cli/commands/mic_transcribe.rs` and `src/cli/commands/live_half_duplex.rs`, so it should be extracted rather than copied verbatim.

Is there reusable code for streaming audio frames:

- Yes. `src/audio/ring.rs` and the `AudioFrame`-based loop patterns are reusable.

Is there useful echo suppression or turn-taking logic:

- Yes. `SelfHearingState` and `SpeakerReferenceMask` are strong.
- Turn-taking logic exists in the live command loop, but it is still command-embedded rather than cleanly isolated.

How Listenbury’s voice loop should be adapted to a robot body:

- Replace "assistant conversation loop" framing with body-centric components:
  ear capture,
  ear interpretation,
  speech planner,
  mouth renderer,
  speaker playback,
  self-hearing,
  and body action arbitration.
- Keep the timing seams and traces, but have the body runtime own scheduling.

## Mortar-Sea findings

### Reuse

- `psyche/src/sensation.rs`
  Best typed sensation contract of the four repos.
- `psyche/src/impression.rs`
  Best explanation of impression semantics.
- `psyche/src/experience.rs`
  Best explanation of experience semantics.
- `psyche/src/memory.rs`
  Best small reference contract for memory recall as sensation.
- `psyche/src/pipeline.rs`
  Best small canonical statement of the cognition loop.
- `src/voice_stream.rs`
  Best direct `<say>` parser and structural speech gating path.
- `src/mouth.rs`
  Best explicit mouth gate abstraction.

### Rewrite

- `psyche/src/memory.rs::Memory::recall`
  The reference trait recalls all experiences; `pete` needs queryable and typed recall.
- `psyche/src/pipeline.rs`
  Useful as a reference, but too synchronous and in-memory for the embodied runtime.
- `face/src/realtime_experience.rs`
  It has useful prompt/context discipline, but it is too application-specific and large to steal whole.

### Avoid

- Preserving `faculties` and `wits` as crate names or user-facing runtime components in `pete`.
- Treating `<say>` as the primary robot action language beyond speech itself.

### Specific files/types/functions

- Reuse or port:
  `psyche/src/sensation.rs::Sensation`
- Reuse or port:
  `psyche/src/sensation.rs::Provenance`
- Reuse or port:
  `psyche/src/impression.rs::Impression`
- Reuse or port:
  `psyche/src/experience.rs::Experience`
- Reuse or port:
  `psyche/src/memory.rs::Memory`
- Reuse or port:
  `psyche/src/memory.rs::LinkedMemory`
- Reuse or port:
  `psyche/src/pipeline.rs::Pipeline::recall_into_timeline`
- Reuse or port:
  `psyche/src/realtime_experience.rs::RealTimeExperienceWit`
- Reuse or port:
  `src/voice_stream.rs::VoiceStreamParser`
- Reuse or port:
  `src/mouth.rs::VoiceMouthGate`
- Reuse or port:
  `src/mouth.rs::MouthGate`

- Rewrite:
  `psyche/src/pipeline.rs` into async runtime services
- Rewrite:
  `face/src/realtime_experience.rs`

Terminology to preserve:

- `Sensation`
- `Impression`
- `Experience`
- `Memory`
- `Provenance`
- `occurred_at` vs `observed_at`
- `recall_into_timeline` as a concept, though not necessarily the exact method name

Terminology to discard:

- `faculties`
- `wits`

Architectural pieces that should replace or modify the pete spec:

- Replace any vague "experience latent" input semantics with an explicit ladder:
  sensations feed local encoders,
  impressions feed meaning-making,
  experiences feed durable memory and training labels.
- Make memory recall a dual-output subsystem:
  symbolic recollection sensations and numeric memory-sense features.
- Use `voice_stream` / `mouth` gating ideas for speech only; do not generalize XML tags to all robot actions.

Is Mortar-Sea’s cognitive vocabulary cleaner than Daringsby’s:

- Yes, clearly.

Are “faculties” and “wits” worth preserving in pete:

- As historical inspiration, yes.
- As primary public/runtime terminology, no.

Are there useful model boundaries for PETE’s small neural organs:

- Yes.
- `RealTimeExperienceWit` suggests a small fast interpretation organ.
- The clean separation between sensation, impression, and experience is exactly the right place to hang tiny specialist models.

Is there a better abstraction for sensory streams or impressions:

- Yes. `mortar-sea`'s typed `Sensation` plus `Provenance` is better than `daringsby`'s open `Any` sensation payloads.

## Tongues findings

### Reuse

- `Cargo.toml`
  Good workspace dependency baseline for Burn 0.21 with `ndarray` and `autodiff`.
- `crates/tongues-neural/src/lib.rs`
  Best reusable artifact manifest and recorder helper.
- `crates/tongues-common-phone/src/lib.rs`
  Best minimal viable Burn training loop in the repo set.
- `crates/tongues-cli/src/main.rs`
  Good family-first CLI structure pattern.
- `xtask/src/main.rs`
  Good place for scaffolding and bench/demo helpers, though keep `xtask` smaller in `pete`.

### Rewrite

- The overall model-family sprawl. `pete` should not copy the number of crates or command namespaces.
- Any task-specific seq2seq assumptions in `tongues-g2p2g`.

### Avoid

- Copying `tongues`' huge multi-family CLI wholesale.
- Pulling in CUDA backend complexity on day one unless a real training need exists.

### Specific files/types/functions

- Reuse or port:
  `Cargo.toml` workspace dependency pattern
- Reuse or port:
  `crates/tongues-neural/src/lib.rs::ModelArtifactManifest`
- Reuse or port:
  `crates/tongues-neural/src/lib.rs::make_recorder`
- Reuse or port:
  `crates/tongues-common-phone/src/lib.rs::train_with_progress`
- Reuse or port:
  `crates/tongues-common-phone/src/lib.rs::write_train_state`
- Reuse or port:
  `crates/tongues-common-phone/src/lib.rs::write_jsonl_atomic`
- Reuse or port:
  `crates/tongues-cli/src/main.rs` family-first CLI shape
- Reuse or port:
  `xtask/src/main.rs` lightweight helper-command pattern

Burn setup recommendation:

- Start with Burn 0.21 exactly as `tongues` does:
  `burn = { version = "0.21", features = ["ndarray", "autodiff"] }`
- Keep CPU-first training and inference for the first milestone.
- Add GPU backend only when one of the organ models actually needs it.

Training/checkpointing recommendation:

- Copy the artifact discipline:
  `manifest.json`,
  `model_config.json`,
  `train_config.json`,
  `train_state.json`,
  `model-latest.bin`,
  `model-epoch-N.bin`,
  `model.bin`.
- Prefer `model-latest.bin` during training, epoch checkpoints at validation boundaries, and `model.bin` as best-known checkpoint.
- Keep recorder and manifest helpers in one shared `pete-model-artifacts` crate.

CLI/xtask recommendation:

- Use a family-first CLI:
  `pete train ...`,
  `pete replay ...`,
  `pete sim ...`,
  `pete inspect ...`,
  or model-family subcommands only where necessary.
- Use `xtask` for scaffold, fixture generation, replay conversion, and benchmark helpers.

Data format recommendation for ledger/replay:

- Use append-only `jsonl` for event/ledger/replay records.
- Use compact binary blobs only for large sensor tensors or audio/video payloads.
- Copy `tongues-common-phone`'s atomic JSONL writing discipline and `listenbury`'s trace/replay payload style.

How should pete structure Burn models:

- One crate for shared tensor/artifact helpers.
- One crate per small neural organ family:
  latent encoder/decoder,
  future predictor,
  memory-sense retriever,
  behavior replacement models.

How should pete checkpoint models:

- `model-latest.bin`
- `model-epoch-N.bin`
- `model.bin`
- `manifest.json`
- `train_state.json`

Is there a good training CLI pattern to reuse:

- Yes. `tongues-cli`'s family-first structure is the right shape, but `pete` should keep fewer families and fewer commands.

Is there a good data/frames pattern for the ExperienceLedger:

- Partly.
- `tongues-common-phone`'s explicit metadata + external feature files is useful.
- `listenbury`'s trace JSONL plus viewer payloads are a better direct reference for replayable organism experience.

What dependencies and feature flags should be copied:

- Copy:
  Burn CPU/autodiff baseline
- Copy:
  `serde`, `serde_json`, `clap`, `anyhow`, `thiserror`
- Copy later:
  optional CUDA features only when justified

## Recommended pete crate layout

- `crates/pete-core`
  Core ids, timestamps, provenance, typed events, shared errors.
- `crates/pete-sense`
  `Sensation`, sensor frame types, memory-sense types, normalization.
- `crates/pete-cognition`
  `Impression`, `Experience`, context frame, recall policies, conductor inputs.
- `crates/pete-memory`
  ledger indexing, graph/vector storage, recall services, recollection sensations.
- `crates/pete-audio`
  `AudioFrame`, capture, VAD, suppression, timed-word streams.
- `crates/pete-mouth`
  speech planner, mouth gate, TTS boundary, playback events.
- `crates/pete-body`
  body interfaces, action primitives, embodiment state, simulator/real adapters.
- `crates/pete-autonomic`
  safety vetoes, reflexes, watchdogs, action arbitration.
- `crates/pete-model-artifacts`
  Burn recorder, manifests, checkpoint metadata.
- `crates/pete-models`
  latent encoder/decoder, future predictor, memory-sense retriever, replacement models.
- `crates/pete-runtime`
  simulator-first orchestration loop.
- `crates/pete-server`
  dashboard/WebSocket/API.
- `crates/pete-tools`
  replay/inspect/train helpers.
- `xtask`
  scaffolds, fixture generation, benchmark helpers.

## Recommended dependency list

- `serde`
- `serde_json`
- `anyhow`
- `thiserror`
- `chrono` or a strict internal timestamp type
- `uuid`
- `tokio`
- `tracing`
- `clap`
- `burn`
- `cpal`
- `rtrb`
- `crossbeam-channel`
- `reqwest`
- `axum`
- `tokio-tungstenite` or axum ws support
- optional:
  `webrtc-vad`
- optional:
  Qdrant and Neo4j client dependencies

## Recommended terminology

- Preserve:
  `Sensation`
- Preserve:
  `Impression`
- Preserve:
  `Experience`
- Preserve:
  `Provenance`
- Preserve:
  `Memory`
- Preserve:
  `Recollection`
- Introduce:
  `MemorySense`
- Introduce:
  `ActionPrimitive`
- Introduce:
  `BehaviorWrapper`
- Introduce:
  `BehaviorMode`

- Discard:
  `Wit`
- Discard:
  `Faculty`
- Discard:
  `Combobulator` as a production name

`Combobulator` is a charming prototype word, but `pete` should use plainer names like `AwarenessSynthesizer` or `SituationIntegrator`.

## Recommended memory/recall design

- Durable semantic memory should store `Experience` records as the primary unit.
- Raw/auxiliary sensor memory should store linked artifacts:
  face vectors,
  voice vectors,
  place vectors,
  image embeddings,
  audio spans,
  replay segments.
- Retrieval should return three surfaces:
  `Recollection` sensations,
  LLM brief text,
  numeric `MemorySense` tensors/features.
- Qdrant-like vector retrieval should always retain a graph/node back-reference.
- Graph memory should preserve provenance chains and source artifact references.
- Recall policy should support:
  similarity recall,
  recency recall,
  episodic neighborhood recall,
  identity recall,
  and self-generated training slice recall.

## Recommended LLM command design

- Do not copy `daringsby`'s open XML instruction style for body actions.
- Use typed `ActionPrimitive` outputs:
  `Speak`,
  `Look`,
  `Turn`,
  `Drive`,
  `Pause`,
  `QueryMemory`,
  `SetAffect`,
  `Teach`,
  `CritiqueBehavior`.
- If an LLM needs tools, expose typed tool calls and typed results.
- Speech should still pass through a mouth gate.
- Body actions should still pass through autonomic safety and behavior wrappers.

## Recommended audio/voice design

- Port `listenbury`'s `AudioFrame`, VAD seam, self-hearing suppression, ring buffering, and timed-word traces.
- Use a mouth gate inspired by `mortar-sea` for spoken-only regions and chunk commitments.
- Represent speech output as:
  intent text,
  chunk plan,
  synthesized frames,
  playback events,
  self-heard events.
- Keep trace export and live viewer support from the start.

## Recommended model/training design

- Keep models small and organ-specific.
- Start with CPU-friendly Burn models and replay-based offline training.
- Use append-only ledger/replay data as the canonical training source.
- Separate online runtime inference from offline training preparation.
- Give every model a stable artifact manifest and explicit checkpoint lineage.

## Recommended hardcoded-to-model replacement design

- Every replaceable behavior should implement one shared wrapper contract with modes:
  direct,
  shadow_train,
  shadow_infer,
  compare,
  model,
  model_with_fallback.
- The wrapper should emit:
  direct output,
  model output if available,
  comparison metrics,
  training examples,
  veto/fallback reason if any.
- This wrapper belongs near the action/runtime boundary, not deep inside the model crate.
- Speech planners, reflexes, navigation primitives, and memory-query heuristics should all fit this wrapper.

## Cleanup tasks before implementation

- Collapse current `pete` ontology docs around `mortar-sea`'s `Sensation`/`Impression`/`Experience` split.
- Rename any planned `Combobulator`-like component to a plainer integrator name.
- Define `MemorySense` now, before building recall.
- Decide one timestamp/provenance contract and apply it everywhere.
- Split runtime crates by hot-path responsibility before adding code.
- Decide whether graph memory is MVP-critical or phase-two; if MVP, scope it narrowly.
- Extract a minimal live-audio service boundary from `listenbury` instead of copying its command files.
- Define the typed `ActionPrimitive` enum before LLM integration.

## First implementation milestone

Build one simulator-first embodied loop with:

- typed `Sensation`, `Impression`, `Experience`, and `Provenance`
- a replayable `ExperienceLedger` JSONL stream
- `AudioFrame` capture/replay types and `MemorySense` placeholder fields in `Now`
- one `AwarenessSynthesizer` that emits an `Experience`
- one memory service that stores `Experience` plus embeddings and can emit a recollection sensation
- one `Speak` action primitive through a mouth gate
- one `BehaviorWrapper` implementation around a hardcoded speech-selection behavior
- one axum dashboard with:
  ledger tail,
  recollection events,
  LLM commands,
  safety vetoes,
  and actual vs predicted traces

That milestone is enough to validate the core doctrine:

- hardcoded behavior runs directly,
- training examples are emitted,
- model shadow output can be compared,
- recall re-enters state as sensed material,
- and the runtime remains simulator-first and inspectable.
