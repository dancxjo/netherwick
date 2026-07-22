# Brain Observatory diagnostic bundles and comparisons

The Observatory can mark two selected timeline times and export the closed
interval as a portable JSON diagnostic bundle. The bundle contains canonical
sequenced `BrainEvent` envelopes, retained lightweight `Now` snapshots,
artifact identities, asset manifests, component health, transport counters,
typed gaps, schema identities, and an overall SHA-256 checksum. Event inline
payloads retain the canonical size limit. Export pagination continues through
the bounded live history; a 50,000-event safety cap is reported as an explicit
partial bundle rather than silently truncating it.
Snapshots carry both their organism time and server-observed time. Export
selects them directly from the requested observed-time interval, so a quiet
interval can still contain its retained state even when it has no events.

Bundle schema v2 binds a checksummed session identity manifest into that
overall checksum. The manifest records a stable per-server session UUID,
host-boot correlation, source kind, software/build identity, active scene and
safety configuration digest, event clock epochs, observed producer inventory,
model and calibration artifacts, schema versions, creation time, and both the
requested and actually exported interval. `PETE_ROBOT_ID` and
`PETE_HARDWARE_REVISION` provide installation identity when configured;
brainstem boot and firmware identity are included when the live session has
observed them. Every absent value is listed in `unavailable_fields` instead of
being guessed.

The default `redact_sensitive` policy replaces image, audio, and vector asset
locators with redacted identities while retaining media type, byte length,
content checksum, and a hash of the original locator. `manifest_only` preserves
references without embedding bytes. `omit_heavy` marks large assets as omitted.
Large bytes are never pulled into the Observatory merely because an interval is
exported.

Under `redact_sensitive`, robot and boot values are omitted while stable
SHA-256 correlation identifiers remain. This permits same-robot/same-boot
comparison without disclosing the original identifiers. The UI shows source
and session identity for export, replay, and timeline comparison. Verifying a
second bundle in the same page visibly warns if robot, hardware revision, boot,
session, software/build, brainstem firmware, active configuration, or schema
identity differs.

`POST /api/observatory/diagnostic-verify` accepts a bundle without changing
server state. It reports four separate claims: `integrity_valid` for the bundle
and embedded checksums, `structurally_valid` for schema/count/event/sequence and
snapshot-clock consistency, `replayable` for mechanically loadable evidence,
and `evidence_complete` for a replay with no unavailable gaps, partial marker,
or missing payload/snapshot references. Intentional telemetry replacement is
declared but does not make a bundle partial. Missing optional heavy bytes do not
prevent event and snapshot replay; invalid integrity or structure does.
Verification also recomputes the identity checksum and rejects identity/event
clock, artifact, schema, source, or interval mismatches even if an outer bundle
checksum was recomputed. Schema-v1 bundles remain replayable for migration but
are marked `legacy_identity=true` and explicitly warn that source correlation
is unbound.

The comparison controls select two timeline times. The server resolves the
nearest retained snapshot at or before each time and compares flattened
canonical state plus the latest recorded event state. Ordinary value changes
are separate from source, provenance, trust, confidence, uncertainty,
freshness, and calibration-epoch changes. The response also compares event
categories, first-class calibration epochs, distinct calibration artifacts,
baseline/candidate model identities,
recorded/reprocessed lanes, and raw/corrected pose or map paths. Added, removed,
and changed beliefs, entities, tracks, parameters, evidence, proposals, and
outcomes remain visible rather than being reduced to a single diff count.

Endpoints:

- `GET /api/observatory/diagnostic-export?from_ms=...&to_ms=...&asset_policy=redact_sensitive`
- `POST /api/observatory/diagnostic-verify`
- `GET /api/observatory/compare?left_ms=...&right_ms=...`

Serve a verified exported bundle through the same read-only Observatory UI:

```bash
just observatory-replay pete-diagnostic-START-END.json
```

Then open `http://127.0.0.1:8788/view/observatory`. The loader refuses an
invalid overall or embedded-asset checksum before binding the server. Original
event IDs, sequences, clock epochs, typed sequence gaps, snapshots, trust, and
artifact identities populate the same API and page; replay mode exposes no
Reign/control router.
