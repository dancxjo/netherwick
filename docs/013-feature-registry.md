# 013 Universal Feature Registry

Everything Pete observes should enter the learning architecture as a **Feature**.

The Feature Registry is the canonical observation layer. It replaces the idea that each subsystem owns its own kind of downstream object: voxels, point cloud points, face vectors, voice vectors, body events, image labels, memory recalls, prediction errors, and LLM critiques should all be representable as Features before clustering, binding, memory, graph storage, or prediction consumes them.

```text
sensor / recall / prediction / critique
  -> Feature
  -> FeatureRegistry
  -> clustering, bindings, constellations, memory, Qdrant, Neo4j, prediction
```

Pete should not remember "Kinect depth sample #41892." Pete should remember "I observed a feature." Everything else in the architecture begins there.

## Why

Pete already contains many feature-like concepts, but they are shaped by their source subsystem:

- voxels and point cloud points,
- object observations and image regions,
- face, voice, scene, and teacher vectors,
- odometry, IMU, and body state events,
- image captions, transcripts, labels, and human corrections,
- memory recalls and remembered places,
- predictions, surprise, counterfactuals, and LLM critiques.

Those differences are useful at the capture boundary, but harmful downstream. A constellation search should not need to understand raw Kinect structures. A binding gate should not care whether an embedding came from vision, audio, language, body state, memory, or prediction. Neo4j and Qdrant should receive stable observation references, not a fresh custom payload for every modality.

The registry gives Pete one common vocabulary for observed evidence while preserving enough provenance to debug and replay where the evidence came from.

## What

A `Feature` is a single observed or generated evidence item. It may be raw-ish, derived, recalled, predicted, or critiqued, but it must still be an observation-layer fact rather than a learned relationship.

The minimal shape should support:

```rust
struct Feature {
    id: FeatureId,
    feature_type: FeatureType,
    modality: Modality,
    created_at: Timestamp,
    confidence: f32,
    provenance: Provenance,
    source_frame: Option<FrameId>,
    source_sensor: Option<SensorId>,
    vector_refs: Vec<VectorRef>,
    world_pose: Option<Pose3>,
    local_pose: Option<Pose3>,
    metadata: FeatureMetadata,
}
```

The field names are illustrative, not a demand for this exact Rust API. The important contract is that every feature has stable identity, modality, time, confidence, provenance, vector references when applicable, pose when applicable, and a structured metadata payload for modality-specific details.

### Identity

`FeatureId` must be globally unique and stable enough for cross-system references. Registry consumers should pass around feature ids rather than copying whole source payloads.

### Type and modality

`feature_type` says what kind of observation this is: voxel, plane, speech segment, bumper state, remembered place, surprise event, action proposal, and so on.

`modality` says where the evidence belongs at the broad architecture level: vision, geometry, motion, audio, language, body, memory, prediction, or another registered modality.

### Provenance

Provenance must answer "why do we believe this feature exists?" It should capture the producing subsystem, model or algorithm version when relevant, capture/session id when available, and links to source records that can be replayed or inspected.

### Vectors

Features should refer to vectors rather than inline every vector payload. A feature may have zero, one, or many vector refs: for example an image region descriptor, face embedding, voice embedding, scene vector, teacher vector, or fused latent reference.

### Pose

`world_pose` is for observations that have a map/world pose. `local_pose` is for observations meaningful in a source-local frame, such as camera coordinates, robot base coordinates, image region coordinates, or short-horizon motion frames.

Features without meaningful pose should leave pose absent. Do not fake geometry to make every feature look spatial.

### Metadata

Metadata is where modality-specific details live: image boxes, segmentation masks, occupancy values, point counts, text snippets, battery levels, audio direction estimates, body mode flags, prediction horizon, surprise magnitude, or LLM critique text.

Metadata must not become a hiding place for learned relationships. If a payload says this feature is the same entity as another feature, belongs to a constellation, or is a causal explanation, that relationship belongs in a downstream layer.

## Feature Categories

The registry must be able to represent at least these feature categories.

### Vision

- object detections,
- image regions,
- semantic segmentation regions,
- RGB patches,
- image descriptors,
- face observations.

### Geometry

- voxels,
- point cloud features,
- planes,
- corners,
- blobs,
- occupancy cells.

### Motion

- odometry events,
- IMU events,
- optical flow,
- motion vectors.

### Audio

- speech segments,
- voice embeddings,
- sound events,
- direction estimates.

### Language

- transcripts,
- image descriptions,
- LLM labels,
- human corrections.

### Body

- battery,
- bumper,
- cliff,
- dock state,
- wheel drop,
- safety mode,
- robot mode.

### Memory

- recalled experiences,
- remembered places,
- remembered entities,
- remembered actions.

### Prediction

- future predictions,
- surprise,
- counterfactuals,
- action proposals.

## Registry Responsibilities

The Feature Registry is a lookup and indexing service. It should support:

- inserting or upserting features from perception, body, memory, language, and prediction subsystems,
- retrieving a feature by id,
- retrieving features by modality,
- retrieving features by time window,
- retrieving features by source frame,
- retrieving features by source sensor,
- retrieving features by vector id,
- retrieving features by provenance,
- returning stable ids and lightweight references suitable for downstream systems.

The registry should be boring on purpose. Its value is reliability, not interpretation.

## Non-Responsibilities

The registry must not:

- perform clustering,
- perform binding,
- perform constellation discovery,
- merge identities,
- infer semantics,
- decide that two features describe the same entity,
- decide that a feature is an object, place, affordance, or memory anchor.

Those jobs belong to downstream systems. The registry can make evidence easy to find, but it must not decide what the evidence means.

## Conversion Plan

Existing subsystem structures should be convertible to Features without breaking current APIs.

A safe migration path:

1. Define the core `Feature`, ids, modality/type enums, provenance, vector refs, pose refs, and metadata shape.
2. Add an in-memory registry implementation with the required indexes.
3. Add adapters from existing perception structures into Features, beginning with voxels, occupancy/map cells, body events, and language labels.
4. Let current APIs continue returning their existing structures while also emitting Features.
5. Move new learning systems to consume Feature ids or Feature queries first.
6. Gradually replace direct raw-structure dependencies in clustering, constellation search, memory, Qdrant, and Neo4j ingestion.

The migration should be additive at first. Do not pause perception progress to force a one-shot rewrite of every subsystem.

## Acceptance Criteria

- Every perception subsystem has a clear path to emit Features.
- Existing structures can be converted to Features without breaking existing APIs.
- Features can be queried by id, modality, time window, provenance, vector id, source sensor, and source frame.
- Features can represent observations from vision, geometry, motion, audio, language, body, memory, and prediction.
- No Feature stores learned relationships such as identity merges, constellation membership, semantic bindings, or causal explanations.
- Existing functionality continues working during migration.
- New clustering, binding, constellation, memory, prediction, Qdrant, Neo4j, active learning, and diagnostic paths prefer Feature Registry inputs over raw Kinect or subsystem-specific structures.

## What Not To Do

- Do not build a graph database inside the registry.
- Do not use the registry as the place where object identity is solved.
- Do not make each modality define an incompatible feature schema.
- Do not inline large vectors when a vector reference is enough.
- Do not invent spatial poses for non-spatial evidence.
- Do not discard source-specific payloads before debugging and replay needs are understood.
- Do not require all existing APIs to change before Features can be emitted.

## Future Consumers

The registry should become the common input for:

- HDBSCAN and other clustering paths,
- vector storage and retrieval through Qdrant,
- graph persistence through Neo4j,
- binding admission,
- cross-modal constellation discovery,
- memory recall and memory write paths,
- prediction and surprise tracking,
- active learning,
- diagnostics and replay tooling.

The desired end state is simple:

```text
sensor
  -> Feature
  -> registry
  -> everything else
```

