# 006 Models

Burn-backed models live in `pete-models`, while the rest of the workspace depends only on stable traits and schemas. The simulator should compile lightly without forcing GPU features.

Embodied vectorizers live on the stable schema side in `pete-experience`. They can consume model-backed vectors already surfaced by sensors (`VectorArtifact`) without pulling heavyweight runtimes into the experience crate. If those artifacts are absent, a configured local model path is missing, or a vectorizer is disabled in `configs/models.toml`, deterministic placeholder vectorizers remain available for repeatable tests and demos.

Vector metadata is part of the stable contract. Each `VectorEmbedding` records the vectorizer id, model id, model label, output dimension, source sensation or experience id, source kind, purpose, collection, input summary, provenance, and fallback status. The purpose taxonomy is intentionally search-oriented:

- `visual_similarity`: whole image or visual artifact similarity.
- `scene_similarity`: scene layout/frame similarity.
- `face_identity`: face crop identity matching.
- `transcript_semantic`: speech/transcript text recall, currently backed by `pete.text.hashing.v1` unless an upstream artifact is present.
- `impression_semantic`: generated impression text recall, currently backed by `pete.text.hashing.v1`.
- `voice_identity`: speaker or voice-window matching.
- `experience_semantic`: experience summary and recall text vectors.

This mirrors Daringsby's model-labeled image, face, and CLIP/OpenCLIP scene embedding path while keeping Pete's stable crates free of heavyweight model runtimes. Listenbury informs the voice-signature and pluggable embedding seams. It also adopts Mortar-Sea's architectural rule that sensations and experiences may have multiple vectors, and each vector must declare what it represents, what produced it, and which purpose-specific collection it belongs to.

Prediction behaviors now use the learned `ExperienceLatent` produced from `ExperienceInstant`. Runtime should fail with "no latent yet" when a trained encoder is unavailable instead of adapting arbitrary present-state vectors into latent inputs. Width mismatches are checkpoint errors that require retraining.

The unified #37 trainer is tracked in [unified-experience.md](unified-experience.md). It assembles multimodal teacher-vector slots plus explicit missingness masks into a `UnifiedExperienceInstant`, trains a learned `ExperienceLatent`, predicts `ExperienceLatent_t+1`, computes `ExperienceSurprise`, and reports decode/predict/compare evidence against copy-current, random projection, and mechanical Instant research baselines. The broader canonical runtime/replay DTO is [ExperienceInstant](instant.md), which replaces the old Sensorium concept for #38.
