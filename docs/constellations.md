# Cross-Modal Constellations

A **constellation** is a repeatable pattern of experience. It is not limited to vision and it is not yet an object label.

The working definition:

```text
constellation = a recurrent arrangement of signals that persists across time,
viewpoint, context, or action strongly enough to be recognized again
```

For the current Pete milestone, the easiest constellations to see are visual-spatial: nearby colored voxels, surfaces, depth discontinuities, planes, corners, and blobs. That is only the first species of the idea. The abstraction should generalize across all modalities.

## Modalities that may contribute

A constellation may draw evidence from:

- **geometry:** point clouds, voxels, planes, normals, occupancy, relative distance, relative angle,
- **color and image:** dominant RGB, texture-ish image features, segmentation hints, image descriptions,
- **motion:** odometry, IMU, optical flow, before/after transforms, commanded motion,
- **body state:** bumpers, cliffs, dock state, battery, wheel reports, robot mode, serial/base health,
- **audio:** speech, noise events, direction estimates, sound identity, temporal rhythm,
- **language:** labels, captions, ASR text, LLM descriptions, user corrections,
- **memory:** prior matches, remembered places, remembered objects, prior failures,
- **affect and drive:** salience, risk, curiosity, frustration, fatigue, task pressure,
- **prediction:** expected next sensory state, surprise, counterfactual outcomes,
- **LLM critique:** training-data complaints, hypothesis critique, suggested tests, suggested motions.

## What constellations are for

Constellations give Pete a layer between raw sensation and named objects.

Do not begin with:

```text
Find the chair.
```

Begin with:

```text
Have I seen this arrangement before?
```

A constellation can later be promoted into:

- an object candidate,
- a place candidate,
- a landmark,
- an affordance,
- a risk pattern,
- a training specimen,
- a memory anchor,
- a counterfactual test target.

This matters because the world rarely arrives as clean labels. A wall edge, a table leg, a patch of floor, a repeated sound, a body refusal, and a user phrase may all be part of the same useful pattern before the system has a word for it.

## Minimal data shape

```rust
struct Constellation {
    id: ConstellationId,
    kind_hint: Option<String>,
    first_seen: Timestamp,
    last_seen: Timestamp,
    confidence: f32,
    modality_evidence: Vec<ModalityEvidence>,
    spatial_signature: Option<SpatialSignature>,
    temporal_signature: Option<TemporalSignature>,
    language_notes: Vec<String>,
    llm_critiques: Vec<String>,
    counterfactuals: Vec<Counterfactual>,
    suggested_actions: Vec<ActionIntent>,
    promoted_as: Option<Promotion>,
}
```

The important property is that the signature should allow partial matches. A constellation should be able to survive missing modalities. If the depth camera sees the shape but the captioner fails, the constellation can still match. If the body state repeats without a reliable visual anchor, that can also be a candidate.

## Search process

```text
current multimodal frame
  -> extract modality-local features
  -> form local candidate signatures
  -> compare against recent and long-term constellation memory
  -> score partial matches
  -> ask for critique when confidence is unstable
  -> suggest information-gathering actions
  -> promote, split, merge, or discard
```

A first implementation can be deliberately plain:

1. Extract visual-spatial clusters from the voxel world.
2. Attach current body state and motion context.
3. Attach captions, labels, or LLM notes if available.
4. Compare candidate signatures by relative geometry and temporal recurrence.
5. Store repeated candidates as provisional constellations.
6. Ask the LLM to critique ambiguous candidates.
7. Let the LLM suggest a motion that would test the candidate.
8. Record whether the motion happened, was vetoed, or failed downstream.

## Role of the LLM

The LLM should not be treated as the owner of truth. It is a critic and experiment designer.

Useful LLM outputs include:

- "This looks like one object but may be two fused clusters."
- "This training row is suspicious because the label does not match the geometry."
- "If this is a table edge, moving left should reveal a parallel support."
- "If this is a wall, forward motion should not change its relative plane much."
- "Do not move yet; the safety state says the body is docked or uncertain."

The LLM can suggest motion intents, but the body path must still pass through controller state and safety. Current movement is not responding, so motion suggestions are presently diagnostic artifacts until the command-to-base path is restored.

## Promotion gates

A candidate constellation becomes more real when it:

- appears in multiple frames,
- survives small viewpoint changes,
- survives lighting or color noise,
- remains coherent after robot motion,
- agrees with memory,
- produces useful predictions,
- survives LLM critique,
- supports a successful action or refusal explanation.

A constellation should be demoted, split, or discarded when:

- it only appears in one noisy frame,
- geometry and captions strongly disagree,
- movement disproves the predicted arrangement,
- safety/body state shows the relevant action could not have occurred,
- two separate patterns were fused by a weak segmentation step.

## Near-term target

The next useful implementation target is not object recognition. It is recurrence recognition:

```text
"I have seen this arrangement before, and I know what evidence would make me more sure."
```

That is the bridge from aligned colored voxels to object permanence, mapping, affordances, and self-training data.
