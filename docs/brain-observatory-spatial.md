# Brain Observatory synchronized spatial view

The spatial view is selected by the same retained snapshot as the timeline,
provenance graph, authority flow, and `Now` inspector. It reuses the canonical
detection, calibration, point-cloud, and map state. It does not recompute a
second browser-side world model.

The endpoint returns lightweight metadata and bounded depth samples. Full RGB
frames load only after an explicit click, and only when the selected snapshot
is still the exact current live frame. Historical selections never substitute
the latest image or map. Missing frames/crops and unavailable historical map
state are labeled explicitly.

Detection cards retain source frame and snapshot, pixel bounds, label
hypotheses and confidence, track ID, model identity, calibration epoch,
geometry trust, robot/world position with uncertainty, position rejection
reasons, and downstream BrainEvent identities. Track IDs remain short-term
tracks, not entity identity.

Depth retains original resolution/count plus visible sample stride. RGB-depth
registration is trusted only with measured validation and a trusted live mount
transform; mismatched alignment timestamps and remount invalidation are shown
as rejection reasons. No world position or registration is fabricated.

For the exact current snapshot, the view embeds the existing live map response:
raw pose trail, corrected pose-graph summary, occupancy/free cells, semantic
cells, projected 3-D voxels, stable/below-floor diagnostics, loop corrections,
and the existing navigation-trust reasons. Raw/corrected comparison is enabled
only when an actual pose-graph correction is recorded. Lidar remains optional
corroboration.
