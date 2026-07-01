# Canonical Instant

#38 replaces the old Sensorium idea with the embodied observation spine already used by runtime, replay, and training:

```text
raw sensor input
  -> primary Sensations
  -> descendant Sensations
  -> teacher / modality vectors
  -> mechanically assembled ExperienceInstant
  -> learned ExperienceLatent
  -> decoding, prediction, memory, behavior
```

`ExperienceInstant` is the canonical DTO for one moment. It is not a policy object and it is not a separate world model. It bundles the current embodied batch: primary and descendant sensations, impressions, summary impression, teacher vectors with metadata and values, body/action context, time window, lineage, provenance, and explicit missing-modality records. The only compressed experience representation is the learned `ExperienceLatent` produced from this Instant.

Creation paths:

- live `Now`: `ExperienceInstant::from_now`
- embodied batch: `ExperienceInstant::from_embodied_now`
- replay ledger frame: `ExperienceFrame::experience_instant`

Encoding uses `ExperienceEncodeInput::from_instant`, so learned Experience models can consume the Instant instead of bespoke per-command glue. Missing modalities are represented in `missing_modalities` and the Instant modality mask; they are not silently erased.
