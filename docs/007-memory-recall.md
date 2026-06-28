# 007 Memory Recall

Recall is not just LLM context. It returns numeric `MemorySense` fields that feed `Now`, alongside summaries and similar past situations.

## Place Recognition

Place recognition asks whether the current canonical moment resembles a previously stored place. It uses the embodied stack rather than a parallel scene embedding path:

```text
Sensations
  -> teacher vectors
  -> ExperienceInstant
  -> fused Experience latent
  -> PlaceRecognitionInput
  -> candidates in RecallBundle / SemanticMapOverlay
```

`PlaceRecognitionInput` carries the fused or learned Experience latent, teacher vector refs, compact range/depth summaries, object/person/voice labels, action and odometry context, time window, and provenance. Stored candidates link back to the source Experience id, Instant/frame id, source vector ids/model ids, rough map cell, confidence, and a short reason. Low-confidence matches are rejected in `PlaceRecognitionOutput`; empty queries return an explicit not-enough-evidence reason.

This is a memory hint, not live pose correction. Candidate `SamePlace` and `SimilarPlace` results can later seed #11 pose-graph loop-closure proposals, while their labels and linked objects/people/voices provide the entity anchors for #23 entity-constellation SLAM. Safety and conductor decisions continue to own behavior; place recognition only supplies recall/map evidence.
