# 006 Models

Burn-backed models live in `netherwick-models`, while the rest of the workspace depends only on stable traits and schemas. The simulator should compile lightly without forcing GPU features.

Embodied vectorizers live on the stable schema side in `netherwick-experience`. They can consume model-backed vectors already surfaced by sensors (`VectorArtifact`) without pulling heavyweight runtimes into the experience crate. If those artifacts are absent, a configured local model path is missing, or a vectorizer is disabled in `configs/models.toml`, deterministic placeholder vectorizers remain available for repeatable tests and demos.

Vector metadata is part of the stable contract. Each `VectorEmbedding` records the vectorizer id, model id, model label, output dimension, source sensation or experience id, source kind, purpose, collection, input summary, provenance, and fallback status. The purpose taxonomy is intentionally search-oriented:

- `visual_similarity`: whole image or visual artifact similarity.
- `scene_similarity`: scene layout/frame similarity.
- `face_identity`: face crop identity matching.
- `transcript_semantic`: speech/transcript text recall, currently backed by `netherwick.text.hashing.v1` unless an upstream artifact is present.
- `impression_semantic`: generated impression text recall, currently backed by `netherwick.text.hashing.v1`.
- `voice_identity`: speaker or voice-window matching.
- `experience_semantic`: fused experience and experience-summary recall.

This mirrors Daringsby's model-labeled image, face, and CLIP/OpenCLIP scene embedding path while keeping Netherwick's stable crates free of heavyweight model runtimes. Listenbury informs the voice-signature and pluggable embedding seams. It also adopts Mortar-Sea's architectural rule that sensations and experiences may have multiple vectors, and each vector must declare what it represents, what produced it, and which purpose-specific collection it belongs to.

Prediction behaviors now have an embodied input path: `Experience.fused_vector` can be adapted into `FutureInput`, `DangerInput`, `ChargeInput`, `ActionValueInput`, and the next-eye/next-ear inputs. Runtime uses the current embodied fused vector when it is available, while keeping the older `ExperienceLatent` route as a fallback and adapting vector width for legacy checkpoints. The selected prediction signals are written back onto the current embodied `Experience.predictions` before ledger and memory persistence.

The online `TinyNowVector` discovery path is tracked in [experience-forge.md](experience-forge.md). Replay and checkpoint artifacts from that forge are the evolved-filter baseline for the later latent round-trip work in #25.
