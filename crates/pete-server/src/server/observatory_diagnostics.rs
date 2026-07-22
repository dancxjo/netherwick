const DIAGNOSTIC_BUNDLE_SCHEMA_VERSION: u32 = 2;
const DIAGNOSTIC_IDENTITY_SCHEMA_VERSION: u32 = 1;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticIdentityAvailability {
    Available,
    Unavailable,
    Redacted,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticIdentityClaim {
    pub availability: DiagnosticIdentityAvailability,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticSessionIdentity {
    pub schema_version: u32,
    pub robot_identity: DiagnosticIdentityClaim,
    pub hardware_revision: DiagnosticIdentityClaim,
    pub boot_identity: DiagnosticIdentityClaim,
    pub session_identity: DiagnosticIdentityClaim,
    pub software_revision: DiagnosticIdentityClaim,
    pub build_identity: DiagnosticIdentityClaim,
    pub brainstem_firmware_identity: DiagnosticIdentityClaim,
    pub brainstem_boot_identity: DiagnosticIdentityClaim,
    pub configuration_sha256: String,
    pub clock_epochs: Vec<String>,
    pub sensor_providers: Vec<String>,
    pub artifacts: Vec<ArtifactIdentity>,
    pub source_kind: String,
    pub source_identity: DiagnosticIdentityClaim,
    pub created_at_ms: u64,
    pub requested_from_ms: u64,
    pub requested_to_ms: u64,
    pub exported_from_ms: u64,
    pub exported_to_ms: u64,
    pub schemas: BTreeMap<String, String>,
    pub unavailable_fields: Vec<String>,
    pub identity_sha256: String,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_identity: Option<DiagnosticSessionIdentity>,
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
    session: Option<&'a SceneSession>,
    session_uuid: &'a str,
    session_created_at_ms: u64,
    identity_override: Option<DiagnosticSessionIdentity>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticVerificationReport {
    pub bundle_checksum_valid: bool,
    pub integrity_valid: bool,
    pub structurally_valid: bool,
    pub invalid_asset_checksums: Vec<String>,
    pub missing_references: Vec<String>,
    pub declared_gaps: usize,
    pub partial: bool,
    pub replayable: bool,
    pub evidence_complete: bool,
    pub structural_errors: Vec<String>,
    pub identity_valid: bool,
    pub legacy_identity: bool,
    pub identity_warnings: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_identity: Option<DiagnosticSessionIdentity>,
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
    pub calibration_artifacts: DiagnosticSetComparison,
    pub model_artifacts: DiagnosticSetComparison,
    pub recorded_reprocessed: DiagnosticSetComparison,
    pub raw_corrected_pose_paths: Vec<String>,
    pub partial: bool,
    pub warnings: Vec<String>,
    pub session_identity: DiagnosticSessionIdentity,
}

fn diagnostic_sha256(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

impl DiagnosticIdentityClaim {
    fn available(value: impl Into<String>) -> Self {
        let value = value.into();
        Self {
            availability: DiagnosticIdentityAvailability::Available,
            correlation_id: Some(diagnostic_sha256(value.as_bytes())),
            value: Some(value),
            reason: None,
        }
    }

    fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            availability: DiagnosticIdentityAvailability::Unavailable,
            value: None,
            correlation_id: None,
            reason: Some(reason.into()),
        }
    }

    fn sensitive(value: Option<String>, policy: DiagnosticAssetPolicy, reason: &str) -> Self {
        let Some(value) = value.filter(|value| !value.trim().is_empty()) else {
            return Self::unavailable(reason);
        };
        if policy == DiagnosticAssetPolicy::RedactSensitive {
            Self {
                availability: DiagnosticIdentityAvailability::Redacted,
                correlation_id: Some(diagnostic_sha256(value.as_bytes())),
                value: None,
                reason: Some("value redacted; correlation hash retained".into()),
            }
        } else {
            Self::available(value)
        }
    }

    fn comparison_key(&self) -> Option<&str> {
        self.correlation_id.as_deref().or(self.value.as_deref())
    }
}

fn diagnostic_identity_checksum(identity: &DiagnosticSessionIdentity) -> String {
    let mut content = identity.clone();
    content.identity_sha256.clear();
    let value = serde_json::to_value(content).expect("diagnostic identity serializes");
    let mut canonical = String::new();
    diagnostic_canonical_json(&value, &mut canonical);
    diagnostic_sha256(canonical.as_bytes())
}

fn diagnostic_event_clock_epochs(events: &[SequencedBrainEvent]) -> Vec<String> {
    let mut epochs = BTreeSet::new();
    for event in events {
        if let Some(epoch) = &event.event.times.occurred.clock_epoch {
            epochs.insert(format!("occurred:{epoch}"));
        }
        if let Some(epoch) = &event.event.times.observed.clock_epoch {
            epochs.insert(format!("observed:{epoch}"));
        }
    }
    epochs.into_iter().collect()
}

fn build_diagnostic_session_identity(
    events: &[SequencedBrainEvent],
    artifacts: &[ArtifactIdentity],
    schemas: &BTreeMap<String, String>,
    session: Option<&SceneSession>,
    session_uuid: &str,
    session_created_at_ms: u64,
    requested_from_ms: u64,
    requested_to_ms: u64,
    policy: DiagnosticAssetPolicy,
) -> DiagnosticSessionIdentity {
    let robot_identity = DiagnosticIdentityClaim::sensitive(
        std::env::var("PETE_ROBOT_ID").ok(),
        policy,
        "PETE_ROBOT_ID is not configured",
    );
    let hardware_revision = std::env::var("PETE_HARDWARE_REVISION")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(DiagnosticIdentityClaim::available)
        .unwrap_or_else(|| {
            DiagnosticIdentityClaim::unavailable("PETE_HARDWARE_REVISION is not configured")
        });
    let boot_identity = DiagnosticIdentityClaim::sensitive(
        fs::read_to_string("/proc/sys/kernel/random/boot_id")
            .ok()
            .map(|value| value.trim().to_string()),
        policy,
        "host boot identity is unavailable",
    );
    let software_revision = option_env!("PETE_GIT_COMMIT")
        .map(DiagnosticIdentityClaim::available)
        .unwrap_or_else(|| DiagnosticIdentityClaim::unavailable("git revision not embedded"));
    let build_identity = DiagnosticIdentityClaim::available(format!(
        "pete-server@{}+{}",
        env!("CARGO_PKG_VERSION"),
        option_env!("PETE_BUILD_ID").unwrap_or("development")
    ));
    let brainstem_firmware_identity = session
        .and_then(|session| session.brainstem_firmware_identity.as_ref())
        .and_then(|identity| serde_json::to_string(identity).ok())
        .map(DiagnosticIdentityClaim::available)
        .unwrap_or_else(|| {
            DiagnosticIdentityClaim::unavailable("brainstem firmware identity was not observed")
        });
    let brainstem_boot_identity = DiagnosticIdentityClaim::sensitive(
        session.and_then(|session| session.brainstem_boot_id.clone()),
        policy,
        "brainstem boot identity was not observed",
    );
    let session_value = serde_json::to_value(session).unwrap_or(serde_json::Value::Null);
    let mut session_canonical = String::new();
    diagnostic_canonical_json(&session_value, &mut session_canonical);
    let configuration_sha256 = diagnostic_sha256(session_canonical.as_bytes());
    let clock_epochs = diagnostic_event_clock_epochs(events);
    let sensor_providers = events
        .iter()
        .map(|event| {
            format!(
                "{}:{}",
                format!("{:?}", event.event.producer.brain).to_lowercase(),
                event.event.producer.component
            )
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let source_kind = session.map_or("live", |session| session.source.as_str()).to_string();
    let source_identity = DiagnosticIdentityClaim::available(source_kind.clone());
    let exported_from_ms = events
        .iter()
        .map(|event| event.event.times.observed.t_ms)
        .min()
        .unwrap_or(requested_from_ms);
    let exported_to_ms = events
        .iter()
        .map(|event| event.event.times.observed.t_ms)
        .max()
        .unwrap_or(requested_to_ms);
    let claims = [
        ("robot_identity", &robot_identity),
        ("hardware_revision", &hardware_revision),
        ("boot_identity", &boot_identity),
        ("software_revision", &software_revision),
        ("brainstem_firmware_identity", &brainstem_firmware_identity),
        ("brainstem_boot_identity", &brainstem_boot_identity),
    ];
    let mut unavailable_fields = claims
        .into_iter()
        .filter(|(_, claim)| {
            claim.availability == DiagnosticIdentityAvailability::Unavailable
        })
        .map(|(name, _)| name.to_string())
        .collect::<Vec<_>>();
    if clock_epochs.is_empty() {
        unavailable_fields.push("clock_epochs".into());
    }
    if sensor_providers.is_empty() {
        unavailable_fields.push("sensor_providers".into());
    }
    let mut identity = DiagnosticSessionIdentity {
        schema_version: DIAGNOSTIC_IDENTITY_SCHEMA_VERSION,
        robot_identity,
        hardware_revision,
        boot_identity,
        session_identity: DiagnosticIdentityClaim::available(session_uuid),
        software_revision,
        build_identity,
        brainstem_firmware_identity,
        brainstem_boot_identity,
        configuration_sha256,
        clock_epochs,
        sensor_providers,
        artifacts: artifacts.to_vec(),
        source_kind,
        source_identity,
        created_at_ms: session_created_at_ms,
        requested_from_ms,
        requested_to_ms,
        exported_from_ms,
        exported_to_ms,
        schemas: schemas.clone(),
        unavailable_fields,
        identity_sha256: String::new(),
    };
    identity.identity_sha256 = diagnostic_identity_checksum(&identity);
    identity
}

pub fn compare_diagnostic_session_identities(
    left: &DiagnosticSessionIdentity,
    right: &DiagnosticSessionIdentity,
) -> Vec<String> {
    let mut warnings = Vec::new();
    for (label, left, right) in [
        ("robot", &left.robot_identity, &right.robot_identity),
        (
            "hardware revision",
            &left.hardware_revision,
            &right.hardware_revision,
        ),
        ("boot", &left.boot_identity, &right.boot_identity),
        ("session", &left.session_identity, &right.session_identity),
        ("software revision", &left.software_revision, &right.software_revision),
        ("build", &left.build_identity, &right.build_identity),
        (
            "brainstem firmware",
            &left.brainstem_firmware_identity,
            &right.brainstem_firmware_identity,
        ),
    ] {
        if left.comparison_key() != right.comparison_key() {
            warnings.push(format!("{label} identity differs"));
        }
    }
    if left.configuration_sha256 != right.configuration_sha256 {
        warnings.push("configuration digest differs".into());
    }
    if left.schemas != right.schemas {
        warnings.push("schema versions differ".into());
    }
    warnings
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
    from_ms: u64,
    to_ms: u64,
) -> Vec<ObservatoryNowSnapshot> {
    let snapshot_ids: BTreeSet<String> = events
        .iter()
        .filter_map(|event| event.event.references.snapshot_id.clone())
        .collect();
    snapshots
        .iter()
        .filter(|snapshot| {
            snapshot_ids.contains(&snapshot.snapshot_id)
                || (snapshot.observed_at_ms >= from_ms && snapshot.observed_at_ms <= to_ms)
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
        session,
        session_uuid,
        session_created_at_ms,
        identity_override,
    } = input;
    let plain_events: Vec<BrainEvent> = events.iter().map(|event| event.event.clone()).collect();
    let assets = diagnostic_asset_entries(&events, policy);
    let artifacts = diagnostic_artifacts(&events);
    let mut warnings = Vec::new();
    let replacements = gaps
        .iter()
        .filter(|gap| gap.reason == SequenceGapReason::Coalesced)
        .count();
    let unavailable_gaps = gaps.len().saturating_sub(replacements);
    if unavailable_gaps > 0 {
        warnings.push(format!(
            "{unavailable_gaps} declared unavailable transport/capture gaps"
        ));
    }
    if replacements > 0 {
        warnings.push(format!(
            "{replacements} intentional telemetry replacements"
        ));
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
    schemas.insert(
        "diagnostic_identity".into(),
        format!("v{DIAGNOSTIC_IDENTITY_SCHEMA_VERSION}"),
    );
    let session_identity = if let Some(mut identity) = identity_override {
        identity.requested_from_ms = from_ms;
        identity.requested_to_ms = to_ms;
        identity.exported_from_ms = events
            .iter()
            .map(|event| event.event.times.observed.t_ms)
            .min()
            .unwrap_or(from_ms);
        identity.exported_to_ms = events
            .iter()
            .map(|event| event.event.times.observed.t_ms)
            .max()
            .unwrap_or(to_ms);
        identity.clock_epochs = diagnostic_event_clock_epochs(&events);
        identity.artifacts.clone_from(&artifacts);
        identity.schemas.clone_from(&schemas);
        identity.identity_sha256 = diagnostic_identity_checksum(&identity);
        identity
    } else {
        build_diagnostic_session_identity(
            &events,
            &artifacts,
            &schemas,
            session,
            session_uuid,
            session_created_at_ms,
            from_ms,
            to_ms,
            policy,
        )
    };
    let source_id = session_identity.source_kind.clone();
    let manifest = DiagnosticBundleManifest {
        schema_version: DIAGNOSTIC_BUNDLE_SCHEMA_VERSION,
        bundle_id: format!("diagnostic:{}", Uuid::new_v4()),
        source_id,
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
        session_identity: Some(session_identity),
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
    let snapshot_ids: BTreeSet<_> = bundle
        .snapshots
        .iter()
        .map(|snapshot| snapshot.snapshot_id.as_str())
        .collect();
    for envelope in &bundle.events {
        if let BrainEventPayload::Reference { reference } = &envelope.event.payload {
            if !manifest_ids.contains(&reference.id) {
                missing_references.insert(reference.id.clone());
            }
        }
        if let Some(snapshot_id) = envelope.event.references.snapshot_id.as_deref() {
            if !snapshot_ids.contains(snapshot_id) {
                missing_references.insert(format!("snapshot:{snapshot_id}"));
            }
        }
    }
    let mut structural_errors = Vec::new();
    if !matches!(bundle.manifest.schema_version, 1 | DIAGNOSTIC_BUNDLE_SCHEMA_VERSION) {
        structural_errors.push(format!(
            "unsupported diagnostic schema {}",
            bundle.manifest.schema_version
        ));
    }
    for (name, declared, actual) in [
        ("events", bundle.manifest.event_count, bundle.events.len()),
        (
            "snapshots",
            bundle.manifest.snapshot_count,
            bundle.snapshots.len(),
        ),
        ("assets", bundle.manifest.asset_count, bundle.assets.len()),
        ("gaps", bundle.manifest.gap_count, bundle.drops.gaps.len()),
    ] {
        if declared != actual {
            structural_errors.push(format!(
                "manifest {name} count {declared} does not match {actual}"
            ));
        }
    }
    let mut prior_sequence = None;
    for envelope in &bundle.events {
        if prior_sequence.is_some_and(|prior| envelope.sequence <= prior) {
            structural_errors.push(format!(
                "event sequence {} is duplicate or out of order",
                envelope.sequence
            ));
        }
        prior_sequence = Some(envelope.sequence);
        if let Err(error) = envelope.event.validate() {
            structural_errors.push(format!(
                "event {} is invalid: {error}",
                envelope.event.event_id.0
            ));
        }
        if envelope.event.event_type == BrainEventType::Snapshot {
            if let Some(snapshot_id) = envelope.event.references.snapshot_id.as_deref() {
                if let Some(snapshot) = bundle
                    .snapshots
                    .iter()
                    .find(|snapshot| snapshot.snapshot_id == snapshot_id)
                {
                    if envelope.event.times.occurred.t_ms != snapshot.now.t_ms {
                        structural_errors.push(format!(
                            "snapshot {snapshot_id} event time does not match retained Now"
                        ));
                    }
                }
            }
        }
    }
    let legacy_identity = bundle.manifest.schema_version == 1 && bundle.session_identity.is_none();
    let mut identity_warnings = Vec::new();
    let identity_valid = if legacy_identity {
        identity_warnings.push(
            "legacy v1 bundle has no bound session identity; source correlation is unavailable"
                .into(),
        );
        true
    } else if let Some(identity) = bundle.session_identity.as_ref() {
        let mut valid = true;
        if identity.schema_version != DIAGNOSTIC_IDENTITY_SCHEMA_VERSION {
            structural_errors.push(format!(
                "unsupported diagnostic identity schema {}",
                identity.schema_version
            ));
            valid = false;
        }
        if diagnostic_identity_checksum(identity) != identity.identity_sha256 {
            structural_errors.push("session identity checksum mismatch".into());
            valid = false;
        }
        if identity.source_kind != bundle.manifest.source_id
            || identity.source_identity.value.as_deref() != Some(bundle.manifest.source_id.as_str())
        {
            structural_errors.push("manifest source does not match session identity".into());
            valid = false;
        }
        if identity.requested_from_ms != bundle.manifest.from_ms
            || identity.requested_to_ms != bundle.manifest.to_ms
        {
            structural_errors.push("manifest interval does not match session identity".into());
            valid = false;
        }
        if identity.schemas != bundle.manifest.schemas {
            structural_errors.push("manifest schemas do not match session identity".into());
            valid = false;
        }
        if identity.artifacts != bundle.artifacts {
            structural_errors.push("bundle artifacts do not match session identity".into());
            valid = false;
        }
        if identity.clock_epochs != diagnostic_event_clock_epochs(&bundle.events) {
            structural_errors.push("event clock epochs do not match session identity".into());
            valid = false;
        }
        for field in &identity.unavailable_fields {
            identity_warnings.push(format!("identity field unavailable: {field}"));
        }
        valid
    } else {
        structural_errors.push("diagnostic v2 bundle is missing session identity".into());
        false
    };
    let integrity_valid =
        bundle_checksum_valid && invalid_asset_checksums.is_empty() && identity_valid;
    let structurally_valid = structural_errors.is_empty();
    let partial = bundle.manifest.partial
        || bundle
            .drops
            .gaps
            .iter()
            .any(|gap| gap.reason != SequenceGapReason::Coalesced);
    let replayable = integrity_valid && structurally_valid;
    let evidence_complete = replayable && !partial && missing_references.is_empty();
    DiagnosticVerificationReport {
        bundle_checksum_valid,
        integrity_valid,
        structurally_valid,
        invalid_asset_checksums,
        missing_references: missing_references.into_iter().collect(),
        declared_gaps: bundle.drops.gaps.len(),
        partial,
        replayable,
        evidence_complete,
        structural_errors,
        identity_valid,
        legacy_identity,
        identity_warnings,
        session_identity: bundle.session_identity.clone(),
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
        let replay_identity = bundle.session_identity.clone();
        let replay_session_uuid = replay_identity
            .as_ref()
            .and_then(|identity| identity.session_identity.value.clone())
            .unwrap_or_else(|| format!("legacy-replay:{}", Uuid::new_v4()));
        let replay_created_at_ms = replay_identity
            .as_ref()
            .map_or_else(wall_now_ms, |identity| identity.created_at_ms);
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
            diagnostic_session_uuid: Arc::new(replay_session_uuid),
            diagnostic_session_created_at_ms: replay_created_at_ms,
            diagnostic_replay_identity: Arc::new(Mutex::new(replay_identity)),
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
            brainstem_boot_id: None,
            brainstem_firmware_identity: None,
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

fn diagnostic_calibration_epoch_sets(
    events: &[BrainEvent],
    at_ms: u64,
) -> BTreeMap<String, String> {
    events
        .iter()
        .filter(|event| event.times.observed.t_ms <= at_ms)
        .flat_map(|event| {
            event
                .calibration_epochs
                .iter()
                .map(move |epoch| (epoch.clone(), event.producer.component.clone()))
        })
        .collect()
}

fn build_diagnostic_comparison(
    left: &ObservatoryNowSnapshot,
    right: &ObservatoryNowSnapshot,
    events: &[BrainEvent],
    partial: bool,
    warnings: Vec<String>,
    session_identity: DiagnosticSessionIdentity,
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
    let left_epochs = diagnostic_calibration_epoch_sets(events, left.now.t_ms);
    let right_epochs = diagnostic_calibration_epoch_sets(events, right.now.t_ms);
    let left_calibration_artifacts =
        diagnostic_identity_sets(events, left.now.t_ms, ArtifactKind::Calibration);
    let right_calibration_artifacts =
        diagnostic_identity_sets(events, right.now.t_ms, ArtifactKind::Calibration);
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
        calibration_artifacts: diagnostic_compare_sets(
            &left_calibration_artifacts,
            &right_calibration_artifacts,
        ),
        model_artifacts: diagnostic_compare_sets(&left_models, &right_models),
        recorded_reprocessed: diagnostic_compare_sets(&lane(left.now.t_ms), &lane(right.now.t_ms)),
        raw_corrected_pose_paths,
        partial,
        warnings,
        session_identity,
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
    let snapshots = diagnostic_select_snapshots(
        &retained_snapshots,
        &events,
        query.from_ms,
        query.to_ms,
    );
    let session = state.session();
    let identity_override = state
        .diagnostic_replay_identity
        .lock()
        .ok()
        .and_then(|identity| identity.clone());
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
        session: session.as_ref(),
        session_uuid: &state.diagnostic_session_uuid,
        session_created_at_ms: state.diagnostic_session_created_at_ms,
        identity_override,
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
    let mut schemas = BTreeMap::new();
    schemas.insert(
        "brain_event".into(),
        format!("v{}", BRAIN_EVENT_SCHEMA_VERSION),
    );
    schemas.insert(
        "diagnostic_bundle".into(),
        format!("v{DIAGNOSTIC_BUNDLE_SCHEMA_VERSION}"),
    );
    schemas.insert(
        "diagnostic_identity".into(),
        format!("v{DIAGNOSTIC_IDENTITY_SCHEMA_VERSION}"),
    );
    let session = state.session();
    let session_identity = state
        .diagnostic_replay_identity
        .lock()
        .ok()
        .and_then(|identity| identity.clone())
        .unwrap_or_else(|| {
            build_diagnostic_session_identity(
                &events,
                &diagnostic_artifacts(&events),
                &schemas,
                session.as_ref(),
                &state.diagnostic_session_uuid,
                state.diagnostic_session_created_at_ms,
                left.now.t_ms.min(right.now.t_ms),
                left.now.t_ms.max(right.now.t_ms),
                DiagnosticAssetPolicy::RedactSensitive,
            )
        });
    Ok(Json(build_diagnostic_comparison(
        &left,
        &right,
        &events
            .into_iter()
            .map(|event| event.event)
            .collect::<Vec<_>>(),
        partial || !gaps.is_empty(),
        warnings,
        session_identity,
    )))
}
