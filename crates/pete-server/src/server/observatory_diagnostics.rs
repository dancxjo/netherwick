const DIAGNOSTIC_BUNDLE_SCHEMA_VERSION: u32 = 1;
const MAX_DIAGNOSTIC_EVENTS: usize = 50_000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticAssetPolicy {
    ManifestOnly,
    #[default]
    RedactSensitive,
    OmitHeavy,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticBundleManifest {
    pub schema_version: u32,
    pub bundle_id: String,
    pub source_id: String,
    pub from_ms: u64,
    pub to_ms: u64,
    pub asset_policy: DiagnosticAssetPolicy,
    pub event_count: usize,
    pub snapshot_count: usize,
    pub asset_count: usize,
    pub gap_count: usize,
    pub partial: bool,
    pub warnings: Vec<String>,
    pub schemas: BTreeMap<String, String>,
    pub bundle_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticAssetEntry {
    pub reference: PayloadReference,
    pub disposition: String,
    pub locator_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedded_base64: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticDropMetadata {
    pub gaps: Vec<BrainEventSequenceGap>,
    pub transport: BrainEventTransportHealth,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticBundle {
    pub manifest: DiagnosticBundleManifest,
    pub events: Vec<SequencedBrainEvent>,
    pub snapshots: Vec<ObservatoryNowSnapshot>,
    pub artifacts: Vec<ArtifactIdentity>,
    pub assets: Vec<DiagnosticAssetEntry>,
    pub component_health: ComponentHealthResponse,
    pub drops: DiagnosticDropMetadata,
}

struct DiagnosticBundleBuild<'a> {
    events: Vec<SequencedBrainEvent>,
    snapshots: Vec<ObservatoryNowSnapshot>,
    gaps: Vec<BrainEventSequenceGap>,
    transport: BrainEventTransportHealth,
    training: &'a LiveTrainingStatus,
    from_ms: u64,
    to_ms: u64,
    policy: DiagnosticAssetPolicy,
    partial: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticVerificationReport {
    pub bundle_checksum_valid: bool,
    pub invalid_asset_checksums: Vec<String>,
    pub missing_references: Vec<String>,
    pub declared_gaps: usize,
    pub partial: bool,
    pub replayable: bool,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct DiagnosticExportQuery {
    from_ms: u64,
    to_ms: u64,
    #[serde(default)]
    asset_policy: DiagnosticAssetPolicy,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct DiagnosticCompareQuery {
    left_ms: u64,
    right_ms: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticChangeKind {
    Value,
    ProvenanceTrust,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticFieldChange {
    pub path: String,
    pub kind: DiagnosticChangeKind,
    pub before: Option<serde_json::Value>,
    pub after: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticSetComparison {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub changed: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticComparisonPoint {
    pub snapshot_id: String,
    pub t_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticComparisonResponse {
    pub left: DiagnosticComparisonPoint,
    pub right: DiagnosticComparisonPoint,
    pub fields: Vec<DiagnosticFieldChange>,
    pub event_categories: BTreeMap<String, DiagnosticSetComparison>,
    pub calibration_epochs: DiagnosticSetComparison,
    pub model_artifacts: DiagnosticSetComparison,
    pub recorded_reprocessed: DiagnosticSetComparison,
    pub raw_corrected_pose_paths: Vec<String>,
    pub partial: bool,
    pub warnings: Vec<String>,
}

fn diagnostic_sha256(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn diagnostic_canonical_json(value: &serde_json::Value, output: &mut String) {
    match value {
        serde_json::Value::Null => output.push_str("null"),
        serde_json::Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        serde_json::Value::Number(value) => output.push_str(&value.to_string()),
        serde_json::Value::String(value) => {
            output.push_str(&serde_json::to_string(value).expect("JSON string serializes"));
        }
        serde_json::Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                diagnostic_canonical_json(value, output);
            }
            output.push(']');
        }
        serde_json::Value::Object(values) => {
            output.push('{');
            let mut entries: Vec<_> = values.iter().collect();
            entries.sort_by_key(|(key, _)| *key);
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                output.push_str(&serde_json::to_string(key).expect("JSON object key serializes"));
                output.push(':');
                diagnostic_canonical_json(value, output);
            }
            output.push('}');
        }
    }
}

fn diagnostic_bundle_checksum(bundle: &DiagnosticBundle) -> String {
    let mut content = bundle.clone();
    content.manifest.bundle_sha256.clear();
    // Hash the same float representation that crosses the JSON wire. The
    // `Value` serializer widens f32 values before formatting, while the normal
    // JSON serializer emits the shortest f32 round-trip representation.
    let wire = serde_json::to_vec(&content).expect("diagnostic bundle serializes");
    let value: serde_json::Value =
        serde_json::from_slice(&wire).expect("serialized diagnostic bundle is JSON");
    let mut canonical = String::new();
    diagnostic_canonical_json(&value, &mut canonical);
    diagnostic_sha256(canonical.as_bytes())
}

fn finalize_diagnostic_bundle(mut bundle: DiagnosticBundle) -> DiagnosticBundle {
    bundle.manifest.bundle_sha256 = diagnostic_bundle_checksum(&bundle);
    bundle
}

fn diagnostic_query_events(
    hub: &BrainEventHub,
    from_ms: u64,
    to_ms: u64,
) -> Result<(Vec<SequencedBrainEvent>, Vec<BrainEventSequenceGap>, bool), BrainEventQueryError> {
    let mut after_sequence = 0;
    let mut events = Vec::new();
    let mut gaps = Vec::new();
    let mut partial = false;
    loop {
        let page = hub.query(&BrainEventQuery {
            after_sequence: Some(after_sequence),
            observed_from_ms: Some(from_ms),
            observed_to_ms: Some(to_ms),
            limit: Some(MAX_OBSERVATORY_QUERY_LIMIT),
            ..Default::default()
        })?;
        let next = page.next_cursor;
        for record in page.records {
            match record {
                BrainEventStreamRecord::Event { envelope } => events.push(envelope),
                BrainEventStreamRecord::Gap { gap } => gaps.push(gap),
            }
        }
        if events.len() >= MAX_DIAGNOSTIC_EVENTS {
            events.truncate(MAX_DIAGNOSTIC_EVENTS);
            partial = true;
            break;
        }
        if next <= after_sequence
            || page
                .health
                .newest_sequence
                .is_none_or(|newest| next >= newest)
        {
            break;
        }
        after_sequence = next;
    }
    gaps.sort_by_key(|gap| (gap.from_sequence, gap.to_sequence));
    gaps.dedup_by_key(|gap| (gap.from_sequence, gap.to_sequence));
    Ok((events, gaps, partial))
}

fn diagnostic_asset_entries(
    events: &[SequencedBrainEvent],
    policy: DiagnosticAssetPolicy,
) -> Vec<DiagnosticAssetEntry> {
    let mut references = BTreeMap::<String, PayloadReference>::new();
    for envelope in events {
        if let BrainEventPayload::Reference { reference } = &envelope.event.payload {
            references.insert(reference.id.clone(), reference.clone());
        }
    }
    references
        .into_values()
        .map(|mut reference| {
            let locator_sha256 = diagnostic_sha256(reference.locator.as_bytes());
            let sensitive = reference.media_type.starts_with("image/")
                || reference.media_type.starts_with("audio/")
                || reference.media_type.contains("vector");
            let heavy = sensitive
                || reference
                    .byte_len
                    .is_some_and(|byte_len| byte_len > 64 * 1024);
            let disposition = match policy {
                DiagnosticAssetPolicy::ManifestOnly => "manifest_only",
                DiagnosticAssetPolicy::RedactSensitive if sensitive => {
                    reference.redacted = true;
                    reference.locator = format!("redacted://{}", reference.id);
                    "redacted"
                }
                DiagnosticAssetPolicy::OmitHeavy if heavy => "omitted_heavy",
                _ => "manifest_only",
            };
            DiagnosticAssetEntry {
                reference,
                disposition: disposition.into(),
                locator_sha256,
                embedded_base64: None,
            }
        })
        .collect()
}

fn diagnostic_artifacts(events: &[SequencedBrainEvent]) -> Vec<ArtifactIdentity> {
    let mut artifacts = BTreeMap::new();
    for artifact in events.iter().flat_map(|event| &event.event.artifacts) {
        let key = format!("{:?}:{}:{:?}", artifact.kind, artifact.id, artifact.version);
        artifacts.insert(key, artifact.clone());
    }
    artifacts.into_values().collect()
}

fn diagnostic_select_snapshots(
    snapshots: &[ObservatoryNowSnapshot],
    events: &[SequencedBrainEvent],
) -> Vec<ObservatoryNowSnapshot> {
    let snapshot_ids: BTreeSet<String> = events
        .iter()
        .filter_map(|event| event.event.references.snapshot_id.clone())
        .collect();
    let occurred_from_ms = events
        .iter()
        .map(|event| event.event.times.occurred.t_ms)
        .min();
    let occurred_to_ms = events
        .iter()
        .map(|event| event.event.times.occurred.t_ms)
        .max();
    snapshots
        .iter()
        .filter(|snapshot| {
            snapshot_ids.contains(&snapshot.snapshot_id)
                || matches!(
                    (occurred_from_ms, occurred_to_ms),
                    (Some(from_ms), Some(to_ms))
                        if snapshot.now.t_ms >= from_ms && snapshot.now.t_ms <= to_ms
                )
        })
        .cloned()
        .collect()
}

fn build_diagnostic_bundle(input: DiagnosticBundleBuild<'_>) -> DiagnosticBundle {
    let DiagnosticBundleBuild {
        events,
        snapshots,
        gaps,
        transport,
        training,
        from_ms,
        to_ms,
        policy,
        mut partial,
    } = input;
    let plain_events: Vec<BrainEvent> = events.iter().map(|event| event.event.clone()).collect();
    let assets = diagnostic_asset_entries(&events, policy);
    let artifacts = diagnostic_artifacts(&events);
    let mut warnings = Vec::new();
    if !gaps.is_empty() {
        warnings.push(format!("{} declared transport/capture gaps", gaps.len()));
    }
    if snapshots.is_empty() {
        warnings.push("no retained snapshots overlap the interval".into());
        partial = true;
    }
    if partial {
        warnings.push("bundle is an explicitly partial capture".into());
    }
    let now = snapshots.last().map(|snapshot| &snapshot.now);
    let component_health =
        build_component_health(&plain_events, to_ms, transport.clone(), training, now);
    let mut schemas = BTreeMap::new();
    schemas.insert(
        "brain_event".into(),
        format!("v{}", BRAIN_EVENT_SCHEMA_VERSION),
    );
    schemas.insert(
        "diagnostic_bundle".into(),
        format!("v{DIAGNOSTIC_BUNDLE_SCHEMA_VERSION}"),
    );
    let manifest = DiagnosticBundleManifest {
        schema_version: DIAGNOSTIC_BUNDLE_SCHEMA_VERSION,
        bundle_id: format!("diagnostic:{}", Uuid::new_v4()),
        source_id: "live".into(),
        from_ms,
        to_ms,
        asset_policy: policy,
        event_count: events.len(),
        snapshot_count: snapshots.len(),
        asset_count: assets.len(),
        gap_count: gaps.len(),
        partial,
        warnings,
        schemas,
        bundle_sha256: String::new(),
    };
    finalize_diagnostic_bundle(DiagnosticBundle {
        manifest,
        events,
        snapshots,
        artifacts,
        assets,
        component_health,
        drops: DiagnosticDropMetadata { gaps, transport },
    })
}

pub fn verify_diagnostic_bundle(bundle: &DiagnosticBundle) -> DiagnosticVerificationReport {
    let bundle_checksum_valid = diagnostic_bundle_checksum(bundle) == bundle.manifest.bundle_sha256;
    let mut invalid_asset_checksums = Vec::new();
    let mut manifest_ids = BTreeSet::new();
    for asset in &bundle.assets {
        manifest_ids.insert(asset.reference.id.clone());
        if let (Some(encoded), Some(expected)) =
            (&asset.embedded_base64, asset.reference.checksum.as_deref())
        {
            match base64::engine::general_purpose::STANDARD.decode(encoded) {
                Ok(bytes) if diagnostic_sha256(&bytes) == expected => {}
                _ => invalid_asset_checksums.push(asset.reference.id.clone()),
            }
        }
    }
    let mut missing_references = BTreeSet::new();
    for envelope in &bundle.events {
        if let BrainEventPayload::Reference { reference } = &envelope.event.payload {
            if !manifest_ids.contains(&reference.id) {
                missing_references.insert(reference.id.clone());
            }
        }
    }
    DiagnosticVerificationReport {
        bundle_checksum_valid,
        replayable: bundle_checksum_valid && invalid_asset_checksums.is_empty(),
        invalid_asset_checksums,
        missing_references: missing_references.into_iter().collect(),
        declared_gaps: bundle.drops.gaps.len(),
        partial: bundle.manifest.partial || !bundle.drops.gaps.is_empty(),
    }
}

impl LiveViewState {
    pub fn from_diagnostic_bundle(bundle: DiagnosticBundle) -> Result<Self, String> {
        let verification = verify_diagnostic_bundle(&bundle);
        if !verification.replayable {
            return Err(format!(
                "diagnostic bundle failed verification: checksum_valid={} invalid_assets={}",
                verification.bundle_checksum_valid,
                verification.invalid_asset_checksums.join(",")
            ));
        }
        let history_capacity = bundle.events.len().max(1);
        let hub = BrainEventHub::new(BrainEventHubConfig {
            history_capacity,
            max_query_limit: history_capacity.max(MAX_OBSERVATORY_QUERY_LIMIT),
            ..Default::default()
        });
        let newest_sequence = bundle
            .events
            .iter()
            .map(|event| event.sequence)
            .max()
            .unwrap_or(0);
        hub.shared
            .counters
            .next_sequence
            .store(newest_sequence, Ordering::Release);
        *hub.shared
            .history
            .lock()
            .map_err(|_| "diagnostic replay history lock poisoned".to_string())? = HistoryState {
            events: bundle.events.into(),
            replacements: BTreeMap::new(),
        };
        let state = Self {
            observatory: hub,
            observatory_now: Arc::new(Mutex::new(bundle.snapshots.into())),
            ..Self::default()
        };
        state.observatory_snapshot_sequence.store(
            state
                .observatory_now
                .lock()
                .map_err(|_| "diagnostic replay snapshot lock poisoned".to_string())?
                .len() as u64,
            Ordering::Release,
        );
        state.update_session(SceneSession {
            mode: "diagnostic-replay".into(),
            scenario: None,
            seed: None,
            source: bundle.manifest.source_id,
            tick_ms: None,
            control_state: Some("read-only".into()),
            control_detail: Some("verified diagnostic bundle; no control authority".into()),
            safety_class: None,
            independent_watchdog: None,
            motion_surface: None,
        });
        Ok(state)
    }
}

fn diagnostic_flatten(
    value: &serde_json::Value,
    path: &str,
    out: &mut BTreeMap<String, serde_json::Value>,
) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, child) in map {
                diagnostic_flatten(child, &format!("{path}.{key}"), out);
            }
        }
        serde_json::Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                diagnostic_flatten(child, &format!("{path}[{index}]"), out);
            }
            if items.is_empty() {
                out.insert(path.into(), value.clone());
            }
        }
        _ => {
            out.insert(path.into(), value.clone());
        }
    }
}

fn diagnostic_change_kind(path: &str) -> DiagnosticChangeKind {
    if [
        ".meta",
        ".trust",
        ".provenance",
        ".source",
        ".confidence",
        ".uncertainty",
        ".freshness",
        ".calibration_epoch",
    ]
    .iter()
    .any(|marker| path.contains(marker))
    {
        DiagnosticChangeKind::ProvenanceTrust
    } else {
        DiagnosticChangeKind::Value
    }
}

fn diagnostic_field_changes(
    left: &pete_now::Now,
    right: &pete_now::Now,
) -> Vec<DiagnosticFieldChange> {
    let mut left_fields = BTreeMap::new();
    let mut right_fields = BTreeMap::new();
    diagnostic_flatten(
        &serde_json::to_value(left).unwrap(),
        "now",
        &mut left_fields,
    );
    diagnostic_flatten(
        &serde_json::to_value(right).unwrap(),
        "now",
        &mut right_fields,
    );
    let paths: BTreeSet<String> = left_fields
        .keys()
        .chain(right_fields.keys())
        .cloned()
        .collect();
    paths
        .into_iter()
        .filter_map(|path| {
            let before = left_fields.get(&path).cloned();
            let after = right_fields.get(&path).cloned();
            (before != after).then(|| DiagnosticFieldChange {
                kind: diagnostic_change_kind(&path),
                path,
                before,
                after,
            })
        })
        .collect()
}

fn diagnostic_event_key(event: &BrainEvent) -> String {
    format!(
        "{}:{}:{}",
        event.event_type.as_str(),
        event.kind,
        event.producer.component
    )
}

fn diagnostic_event_state(events: &[BrainEvent], at_ms: u64) -> BTreeMap<String, &BrainEvent> {
    let mut state = BTreeMap::new();
    for event in events
        .iter()
        .filter(|event| event.times.observed.t_ms <= at_ms)
    {
        state.insert(diagnostic_event_key(event), event);
    }
    state
}

fn diagnostic_compare_sets(
    left: &BTreeMap<String, String>,
    right: &BTreeMap<String, String>,
) -> DiagnosticSetComparison {
    let mut comparison = DiagnosticSetComparison::default();
    for (key, value) in right {
        match left.get(key) {
            None => comparison.added.push(key.clone()),
            Some(previous) if previous != value => comparison.changed.push(key.clone()),
            _ => {}
        }
    }
    for key in left.keys().filter(|key| !right.contains_key(*key)) {
        comparison.removed.push(key.clone());
    }
    comparison
}

fn diagnostic_identity_sets(
    events: &[BrainEvent],
    at_ms: u64,
    kind: ArtifactKind,
) -> BTreeMap<String, String> {
    events
        .iter()
        .filter(|event| event.times.observed.t_ms <= at_ms)
        .flat_map(|event| &event.artifacts)
        .filter(|artifact| artifact.kind == kind)
        .map(|artifact| {
            (
                artifact.id.clone(),
                format!("{:?}:{:?}", artifact.version, artifact.checksum),
            )
        })
        .collect()
}

fn build_diagnostic_comparison(
    left: &ObservatoryNowSnapshot,
    right: &ObservatoryNowSnapshot,
    events: &[BrainEvent],
    partial: bool,
    warnings: Vec<String>,
) -> DiagnosticComparisonResponse {
    let left_state = diagnostic_event_state(events, left.now.t_ms);
    let right_state = diagnostic_event_state(events, right.now.t_ms);
    type CategoryStates = (BTreeMap<String, String>, BTreeMap<String, String>);
    let mut categories: BTreeMap<String, CategoryStates> = BTreeMap::new();
    for (key, event) in &left_state {
        categories
            .entry(event.event_type.as_str().into())
            .or_default()
            .0
            .insert(key.clone(), serde_json::to_string(event).unwrap());
    }
    for (key, event) in &right_state {
        categories
            .entry(event.event_type.as_str().into())
            .or_default()
            .1
            .insert(key.clone(), serde_json::to_string(event).unwrap());
    }
    let event_categories = categories
        .into_iter()
        .map(|(category, (before, after))| (category, diagnostic_compare_sets(&before, &after)))
        .collect();
    let left_models = diagnostic_identity_sets(events, left.now.t_ms, ArtifactKind::Model);
    let right_models = diagnostic_identity_sets(events, right.now.t_ms, ArtifactKind::Model);
    let left_epochs = diagnostic_identity_sets(events, left.now.t_ms, ArtifactKind::Calibration);
    let right_epochs = diagnostic_identity_sets(events, right.now.t_ms, ArtifactKind::Calibration);
    let lane = |at_ms| {
        events
            .iter()
            .filter(|event| event.times.observed.t_ms <= at_ms)
            .filter(|event| {
                event.kind.contains("recorded")
                    || event.kind.contains("reprocessed")
                    || event.kind.contains("baseline")
                    || event.kind.contains("candidate")
            })
            .map(|event| (diagnostic_event_key(event), event.kind.clone()))
            .collect::<BTreeMap<_, _>>()
    };
    let fields = diagnostic_field_changes(&left.now, &right.now);
    let raw_corrected_pose_paths = fields
        .iter()
        .filter(|change| {
            change.path.contains("pose")
                || change.path.contains("raw")
                || change.path.contains("corrected")
                || change.path.contains("map")
        })
        .map(|change| change.path.clone())
        .collect();
    DiagnosticComparisonResponse {
        left: DiagnosticComparisonPoint {
            snapshot_id: left.snapshot_id.clone(),
            t_ms: left.now.t_ms,
        },
        right: DiagnosticComparisonPoint {
            snapshot_id: right.snapshot_id.clone(),
            t_ms: right.now.t_ms,
        },
        fields,
        event_categories,
        calibration_epochs: diagnostic_compare_sets(&left_epochs, &right_epochs),
        model_artifacts: diagnostic_compare_sets(&left_models, &right_models),
        recorded_reprocessed: diagnostic_compare_sets(&lane(left.now.t_ms), &lane(right.now.t_ms)),
        raw_corrected_pose_paths,
        partial,
        warnings,
    }
}

async fn get_observatory_diagnostic_export(
    State(state): State<LiveViewState>,
    Query(query): Query<DiagnosticExportQuery>,
) -> Result<Json<DiagnosticBundle>, ObservatoryHttpError> {
    if query.from_ms > query.to_ms {
        return Err(ObservatoryHttpError::bad_request(
            "from_ms must not exceed to_ms",
        ));
    }
    let hub = state.observatory();
    let (events, gaps, partial) = diagnostic_query_events(&hub, query.from_ms, query.to_ms)
        .map_err(|error| ObservatoryHttpError::bad_request(error.to_string()))?;
    let retained_snapshots: Vec<ObservatoryNowSnapshot> = state
        .observatory_now
        .lock()
        .expect("observatory Now history mutex poisoned")
        .iter()
        .cloned()
        .collect();
    let snapshots = diagnostic_select_snapshots(&retained_snapshots, &events);
    Ok(Json(build_diagnostic_bundle(DiagnosticBundleBuild {
        events,
        snapshots,
        gaps,
        transport: hub.health(),
        training: &state.training_status(),
        from_ms: query.from_ms,
        to_ms: query.to_ms,
        policy: query.asset_policy,
        partial,
    })))
}

async fn post_observatory_diagnostic_verify(
    Json(bundle): Json<DiagnosticBundle>,
) -> Json<DiagnosticVerificationReport> {
    Json(verify_diagnostic_bundle(&bundle))
}

async fn get_observatory_compare(
    State(state): State<LiveViewState>,
    Query(query): Query<DiagnosticCompareQuery>,
) -> Result<Json<DiagnosticComparisonResponse>, ObservatoryHttpError> {
    let left = state
        .observatory_now_at_or_before(query.left_ms)
        .map(|selection| selection.selected)
        .ok_or_else(|| ObservatoryHttpError::unavailable("left snapshot is not retained"))?;
    let right = state
        .observatory_now_at_or_before(query.right_ms)
        .map(|selection| selection.selected)
        .ok_or_else(|| ObservatoryHttpError::unavailable("right snapshot is not retained"))?;
    let (events, gaps, partial) =
        diagnostic_query_events(&state.observatory(), 0, left.now.t_ms.max(right.now.t_ms))
            .map_err(|error| ObservatoryHttpError::bad_request(error.to_string()))?;
    let warnings = (!gaps.is_empty())
        .then(|| format!("comparison crosses {} declared gaps", gaps.len()))
        .into_iter()
        .collect();
    Ok(Json(build_diagnostic_comparison(
        &left,
        &right,
        &events
            .into_iter()
            .map(|event| event.event)
            .collect::<Vec<_>>(),
        partial || !gaps.is_empty(),
        warnings,
    )))
}
