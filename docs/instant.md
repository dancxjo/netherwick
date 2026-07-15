# Canonical Experience Record

#38 replaces the old Sensorium idea with the embodied observation spine already used by runtime, replay, and training:

```text
raw sensor input
  -> primary Sensations
  -> descendant Sensations
  -> teacher / modality vectors
  -> belief integration -> structured Now
  -> mechanically assembled ExperienceInstant
  -> learned ExperienceLatent
  -> decoding, prediction, memory, behavior
```

`ExperienceInstant` is the canonical durable observation/audit DTO for one
moment. It is not a policy object and it is not a separate world model. The
canonical current policy state is `Now.world`. The Instant bundles the embodied
evidence batch and the `Now` used at the tick: primary and descendant
sensations, impressions, summary impression, teacher vectors, body/action
context, time window, lineage, provenance, and explicit missing-modality
records.

`ExperienceLatent` is the compressed learned representation produced from this
record. It supports similarity, prediction, retrieval, and learned goal
components. It may re-enter the next update as provenance-marked prediction
evidence, but it cannot overwrite safety-relevant structured beliefs or become
the sole source of current truth.

Creation paths:

- live `Now`: `ExperienceInstant::from_now`
- embodied batch: `ExperienceInstant::from_embodied_now`
- replay ledger frame: `ExperienceFrame::experience_instant`

Encoding uses `ExperienceEncodeInput::from_instant`, so learned Experience models can consume the Instant instead of bespoke per-command glue. Missing modalities are represented in `missing_modalities` and the Instant modality mask; they are not silently erased.
