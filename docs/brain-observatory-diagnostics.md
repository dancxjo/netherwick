# Brain Observatory diagnostic bundles and comparisons

The Observatory can mark two selected timeline times and export the closed
interval as a portable JSON diagnostic bundle. The bundle contains canonical
sequenced `BrainEvent` envelopes, retained lightweight `Now` snapshots,
artifact identities, asset manifests, component health, transport counters,
typed gaps, schema identities, and an overall SHA-256 checksum. Event inline
payloads retain the canonical size limit. Export pagination continues through
the bounded live history; a 50,000-event safety cap is reported as an explicit
partial bundle rather than silently truncating it.

The default `redact_sensitive` policy replaces image, audio, and vector asset
locators with redacted identities while retaining media type, byte length,
content checksum, and a hash of the original locator. `manifest_only` preserves
references without embedding bytes. `omit_heavy` marks large assets as omitted.
Large bytes are never pulled into the Observatory merely because an interval is
exported.

`POST /api/observatory/diagnostic-verify` accepts a bundle without changing
server state. It verifies the overall checksum, verifies checksummed embedded
assets when present, reports references missing from the asset manifest, and
retains partial/gap status. Missing optional heavy bytes do not prevent event
and snapshot replay; invalid bundle or embedded-content checksums do.

The comparison controls select two timeline times. The server resolves the
nearest retained snapshot at or before each time and compares flattened
canonical state plus the latest recorded event state. Ordinary value changes
are separate from source, provenance, trust, confidence, uncertainty,
freshness, and calibration-epoch changes. The response also compares event
categories, calibration artifacts, baseline/candidate model identities,
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
