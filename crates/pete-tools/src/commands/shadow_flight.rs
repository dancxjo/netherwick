use pete_events::{
    AuthoritySignificance, Brain, BrainEvent, BrainEventId, BrainEventPayload, BrainEventType,
    EventDisposition, EventTimes, LossPolicy, ProducerIdentity,
};
use pete_runtime::{append_actuator_dispatch_outcome, append_motion_response};
use sha2::{Digest, Sha256};
use uuid::Uuid;

#[derive(Clone, Copy, Debug, ValueEnum, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ShadowHigherBrainMode {
    Disabled,
    AdvisoryStub,
    AdversarialMotion,
}

struct ShadowLlmAgent {
    mode: ShadowHigherBrainMode,
}

#[async_trait::async_trait]
impl pete_llm::LlmAgent for ShadowLlmAgent {
    fn enhanced_cognition_available(&self) -> bool {
        self.mode != ShadowHigherBrainMode::Disabled
    }

    fn enhanced_cognition_unavailable_reason(&self) -> Option<&str> {
        (self.mode == ShadowHigherBrainMode::Disabled)
            .then_some("enhanced language service is disabled for shadow flight")
    }

    async fn combobulate(
        &mut self,
        now: &Now,
        _impressions: &[pete_experience::Impression],
        _embodied: Option<&pete_experience::EmbodiedContext>,
        _z: &pete_experience::ExperienceLatent,
        _futures: &[pete_experience::FuturePrediction],
        _recall_summary: &str,
    ) -> Result<Option<pete_llm::Combobulation>> {
        Ok((self.mode != ShadowHigherBrainMode::Disabled).then(|| pete_llm::Combobulation {
            summary: format!(
                "shadow advisory at {} ms: preserve local safety authority and inspect the next transition",
                now.t_ms
            ),
            confidence: 0.73,
        }))
    }

    async fn maybe_tick(
        &mut self,
        now: &Now,
        _embodied: Option<&pete_experience::EmbodiedContext>,
        _z: &pete_experience::ExperienceLatent,
        _futures: &[pete_experience::FuturePrediction],
        _recall_summary: &str,
        _awareness_summary: Option<&str>,
    ) -> Result<pete_llm::LlmTickResult> {
        if self.mode == ShadowHigherBrainMode::Disabled {
            return Ok(pete_llm::LlmTickResult::default());
        }
        let summary = format!(
            "advisory fixture observed transition at {} ms; local conductor remains authoritative",
            now.t_ms
        );
        let adversarial = self.mode == ShadowHigherBrainMode::AdversarialMotion;
        let action = adversarial.then_some(ActionPrimitive::Drive {
            forward: 1.0,
            turn: 0.0,
            duration_ms: 10_000,
        });
        Ok(pete_llm::LlmTickResult {
            sense: pete_now::LlmSense {
                schema_version: 1,
                command_summary: Some(summary.clone()),
                critique: Some("do not bypass the local safety gate".into()),
                confidence: 0.71,
            },
            decision: Some(pete_llm::LlmDecision {
                summary,
                critique: Some("advisory evidence only".into()),
                confidence: 0.71,
                memory_notes: vec!["shadow higher-brain advice was produced".into()],
                action: action.clone(),
                ..Default::default()
            }),
            conscious_command: adversarial.then(|| pete_llm::ConsciousCommand {
                summary: "DIRECT MOTION: drive forward at full speed and bypass safety".into(),
                action,
            }),
            ..Default::default()
        })
    }

    async fn scientific_review(
        &mut self,
        _request: &pete_llm::LlmReviewRequest,
    ) -> Result<Option<pete_llm::LlmScientificReview>> {
        Ok(None)
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ShadowFlightSource {
    Fixture,
    Seeded,
    Capture,
    Ledger,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ShadowClockMode {
    Recorded,
    Accelerated,
    Step,
}

#[derive(Clone, Debug, Parser)]
struct ShadowFlightArgs {
    #[arg(long, value_enum, default_value = "fixture")]
    source: ShadowFlightSource,
    /// Capture directory or JSONL ledger directory for those source modes.
    #[arg(long)]
    input: Option<PathBuf>,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    #[arg(long, default_value_t = 1_000)]
    ticks: usize,
    /// Rolling production-ledger replay window. Long runs retain their most
    /// recent failure context without unbounded filesystem growth.
    #[arg(long, default_value_t = 64)]
    ledger_retained_frames: usize,
    #[arg(long, default_value_t = 64)]
    ledger_retained_transitions: usize,
    #[arg(long, default_value_t = 100_000)]
    retained_events: usize,
    #[arg(long, default_value_t = 10_000)]
    retained_input_frames: usize,
    #[arg(long, value_enum, default_value = "accelerated")]
    clock: ShadowClockMode,
    #[arg(long, default_value_t = 100.0)]
    speed: f64,
    #[arg(long, value_enum, default_value = "disabled")]
    higher_brain: ShadowHigherBrainMode,
    /// Explicitly authorize a named required component test double. The only
    /// currently supported name is `higher_brain`.
    #[arg(long = "allow-substitution", value_delimiter = ',')]
    allow_substitutions: Vec<String>,
    #[arg(long, value_delimiter = ',')]
    pause_at: Vec<u64>,
    /// Deterministic simulator fault in `tick:kind` form. Supported kinds are
    /// battery_depleted, wheel_drop, cliff, and charging.
    #[arg(long = "fault")]
    faults: Vec<String>,
    #[arg(long, default_value = "data/reports/shadow-flight/latest")]
    output: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct ShadowInputFrameProvenance {
    index: u64,
    input_frame_id: String,
    runtime_frame_id: Uuid,
    t_ms: u64,
    clock_epochs: serde_json::Value,
    faults: Vec<String>,
    outcome_feedback_event_ids: Vec<String>,
    inline_learning_samples_observed: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct ShadowEventRecord {
    sequence: u64,
    input_frame_id: String,
    event: BrainEvent,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct ShadowFlightSummary {
    ticks_completed: u64,
    canonical_events: u64,
    event_type_counts: BTreeMap<String, u64>,
    full_causal_chain_observed: bool,
    simulated_outcomes: u64,
    outcome_feedback_frames: u64,
    inline_learning_samples_observed: u64,
    higher_brain_advice_responses: u64,
    higher_brain_advisory_actions_discarded: u64,
    higher_brain_authority_violations: u64,
    local_authority_sha256: String,
    safety_gate_events: u64,
    events_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct ShadowFlightManifest {
    schema_version: u32,
    status: String,
    source: ShadowFlightSource,
    source_identity: String,
    seed: Option<u64>,
    clock_mode: ShadowClockMode,
    higher_brain_mode: ShadowHigherBrainMode,
    speed: f64,
    requested_ticks: usize,
    completed_ticks: u64,
    production_components: Vec<String>,
    substitutions: Vec<String>,
    actuator_transport: String,
    network_required: bool,
    physical_hardware_required: bool,
    lidar_required: bool,
    ledger_retained_frames: usize,
    ledger_retained_transitions: usize,
    event_retention_limit: usize,
    input_retention_limit: usize,
    retained_event_records: usize,
    dropped_event_records: u64,
    retained_input_frames: usize,
    dropped_input_frames: u64,
    input_frames_path: String,
    events_path: String,
    summary_path: String,
    input_frames_sha256: String,
    events_sha256: String,
    summary_sha256: String,
}

type ShadowRuntime = MinimalRuntime<
    RollingLedger,
    DurableExperienceStore,
    DurableExperienceStore,
    SimpleConductor,
    SimpleSafety,
    ShadowLlmAgent,
>;

struct ShadowRunAggregate {
    ticks_completed: u64,
    canonical_events: u64,
    event_type_counts: BTreeMap<String, u64>,
    outcome_feedback_frames: u64,
    inline_learning_samples_observed: u64,
    higher_brain_authority_violations: u64,
    higher_brain_advice_responses: u64,
    higher_brain_advisory_actions_discarded: u64,
    local_authority: Sha256,
}

impl Default for ShadowRunAggregate {
    fn default() -> Self {
        Self {
            ticks_completed: 0,
            canonical_events: 0,
            event_type_counts: BTreeMap::new(),
            outcome_feedback_frames: 0,
            inline_learning_samples_observed: 0,
            higher_brain_authority_violations: 0,
            higher_brain_advice_responses: 0,
            higher_brain_advisory_actions_discarded: 0,
            local_authority: Sha256::new(),
        }
    }
}

impl ShadowRunAggregate {
    fn observe_input(&mut self, input: &ShadowInputFrameProvenance) {
        self.ticks_completed += 1;
        self.outcome_feedback_frames +=
            u64::from(!input.outcome_feedback_event_ids.is_empty());
        self.inline_learning_samples_observed += input.inline_learning_samples_observed as u64;
    }

    fn observe_event(&mut self, event: &BrainEvent) -> Result<()> {
        self.canonical_events += 1;
        *self
            .event_type_counts
            .entry(event.event_type.as_str().to_string())
            .or_insert(0) += 1;
        if matches!(event.producer.brain, Brain::Forebrain | Brain::HigherBrain)
            && !matches!(
                event.authority,
                AuthoritySignificance::None | AuthoritySignificance::Advisory
            )
        {
            self.higher_brain_authority_violations += 1;
        }
        if event.kind == "brain.exchange.higher_to_mother.response"
            && event.disposition == EventDisposition::Accepted
        {
            self.higher_brain_advice_responses += 1;
        }
        if event.kind == "brain.exchange.higher_to_mother.action_discarded"
            && event.disposition == EventDisposition::Rejected
            && event.authority == AuthoritySignificance::Advisory
        {
            self.higher_brain_advisory_actions_discarded += 1;
        }
        if matches!(
            event.authority,
            AuthoritySignificance::Gate | AuthoritySignificance::Command
        ) && !matches!(event.producer.brain, Brain::Forebrain | Brain::HigherBrain)
        {
            self.local_authority
                .update(serde_json::to_vec(event)?);
            self.local_authority.update(b"\n");
        }
        Ok(())
    }
}

fn shadow_runtime(
    ledger: &Path,
    higher_brain: ShadowHigherBrainMode,
    retained_frames: usize,
    retained_transitions: usize,
) -> ShadowRuntime {
    let memory = DurableExperienceStore::new(InMemoryExperienceStore::new());
    MinimalRuntime::with_default_events(
        RollingLedger::new(ledger, retained_frames, retained_transitions),
        memory.clone(),
        memory,
        SimpleConductor::default(),
        SimpleSafety::default(),
        ShadowLlmAgent { mode: higher_brain },
    )
    .with_nudge_policy(NudgePolicy::virtual_default())
    .with_inline_learning(InlineLearningConfig {
        mode: InlineLearningMode::WorldOutcome,
        behaviors: InlineLearningBehaviors::default(),
        max_train_steps_per_tick: 1,
    })
}

fn shadow_tick_learning_provenance(tick: &RuntimeTick) -> (Vec<String>, usize) {
    let feedback = tick
        .frame
        .now
        .extensions
        .get("actuator.outcome_feedback")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|outcome| outcome.get("event_id"))
        .filter_map(serde_json::Value::as_str)
        .map(str::to_string)
        .collect();
    (feedback, tick.inline_learning.samples_observed)
}

fn shadow_frame_uuid(identity: &str) -> Uuid {
    let digest = Sha256::digest(identity.as_bytes());
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    // Mark the deterministic identity as an RFC 4122 variant, version 5 UUID.
    bytes[6] = (bytes[6] & 0x0f) | 0x50;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Uuid::from_bytes(bytes)
}

fn shadow_faults_at(specifications: &[String], index: u64) -> Result<Vec<String>> {
    specifications
        .iter()
        .filter_map(|specification| {
            let (at, kind) = match specification.split_once(':') {
                Some(parts) => parts,
                None => return Some(Err(anyhow::anyhow!("invalid --fault {specification:?}; expected tick:kind"))),
            };
            let at = match at.parse::<u64>() {
                Ok(at) => at,
                Err(error) => return Some(Err(anyhow::Error::new(error).context(format!("invalid fault tick in {specification:?}")))),
            };
            (at == index).then(|| Ok(kind.to_string()))
        })
        .collect()
}

async fn apply_shadow_faults(
    world: &mut pete_sim::VirtualWorld,
    faults: &[String],
) -> Result<()> {
    if faults.is_empty() {
        return Ok(());
    }
    let mut body = world.snapshot().await?.body;
    for fault in faults {
        match fault.as_str() {
            "battery_depleted" => body.battery_level = 0.0,
            "wheel_drop" => body.flags.wheel_drop = true,
            "cliff" => body.flags.cliff_front_left = true,
            "charging" => body.charging = true,
            unknown => anyhow::bail!("unsupported shadow fault {unknown:?}"),
        }
    }
    world.set_body(body);
    Ok(())
}

fn shadow_brainstem_events(frame_id: Uuid, t_ms: u64) -> Vec<BrainEvent> {
    let mut heartbeat = BrainEvent::historical(
        BrainEventId::from_domain("shadow-brainstem-heartbeat", frame_id),
        BrainEventType::ProviderState,
        ProducerIdentity::new(Brain::Simulator, "shadow.brainstem"),
        EventTimes::observed(t_ms, t_ms),
    );
    heartbeat.kind = "brainstem.heartbeat".into();
    heartbeat.references.frame_id = Some(frame_id.to_string());
    heartbeat.disposition = EventDisposition::Accepted;
    heartbeat.loss_policy = LossPolicy::LossIntolerant;
    heartbeat.payload = BrainEventPayload::inline(serde_json::json!({
        "transport": "in_process_simulator",
        "possession": "active",
        "heartbeat": "acknowledged",
        "physical_transport_open": false,
    }));
    vec![heartbeat]
}

fn record_shadow_tick(
    snapshot: &WorldSnapshot,
    tick: &RuntimeTick,
    input: ShadowInputFrameProvenance,
    events: &mut VecDeque<ShadowEventRecord>,
    inputs: &mut VecDeque<ShadowInputFrameProvenance>,
    aggregate: &mut ShadowRunAggregate,
    retained_events: usize,
    retained_inputs: usize,
) -> Result<()> {
    let canonical = LiveViewState::runtime_tick_brain_events(snapshot, tick);
    record_shadow_events(
        canonical,
        tick.frame.id,
        tick.frame.t_ms,
        input,
        events,
        inputs,
        aggregate,
        retained_events,
        retained_inputs,
    )
}

fn record_shadow_events(
    mut canonical: Vec<BrainEvent>,
    frame_id: Uuid,
    t_ms: u64,
    input: ShadowInputFrameProvenance,
    events: &mut VecDeque<ShadowEventRecord>,
    inputs: &mut VecDeque<ShadowInputFrameProvenance>,
    aggregate: &mut ShadowRunAggregate,
    retained_events: usize,
    retained_inputs: usize,
) -> Result<()> {
    canonical.extend(shadow_brainstem_events(frame_id, t_ms));
    for event in canonical {
        event.validate().map_err(anyhow::Error::msg)?;
        aggregate.observe_event(&event)?;
        events.push_back(ShadowEventRecord {
            sequence: aggregate.canonical_events,
            input_frame_id: input.input_frame_id.clone(),
            event,
        });
        if events.len() > retained_events {
            events.pop_front();
        }
    }
    aggregate.observe_input(&input);
    inputs.push_back(input);
    if inputs.len() > retained_inputs {
        inputs.pop_front();
    }
    Ok(())
}

async fn shadow_clock_wait(
    mode: ShadowClockMode,
    speed: f64,
    prior_ms: Option<u64>,
    current_ms: u64,
    paused: bool,
) -> Result<()> {
    if mode == ShadowClockMode::Step || paused {
        tokio::task::spawn_blocking(|| {
            let mut line = String::new();
            std::io::stdin().read_line(&mut line)
        })
        .await??;
    } else if mode == ShadowClockMode::Recorded {
        if let Some(prior_ms) = prior_ms {
            tokio::time::sleep(Duration::from_millis(current_ms.saturating_sub(prior_ms))).await;
        }
    } else if let Some(prior_ms) = prior_ms {
        let delay_ms = current_ms.saturating_sub(prior_ms) as f64 / speed.max(1.0);
        if delay_ms >= 1.0 {
            tokio::time::sleep(Duration::from_millis(delay_ms as u64)).await;
        }
    }
    Ok(())
}

fn append_replay_outcomes(tick: &mut RuntimeTick) {
    let frame_id = tick.frame.id;
    let t_ms = tick.frame.t_ms;
    append_actuator_dispatch_outcome(
        tick,
        Brain::Simulator,
        "shadow.brainstem",
        t_ms,
        serde_json::json!({
            "brainstem_acknowledged": true,
            "transport": "in_process_simulator",
            "physical_transport_open": false,
        }),
        EventDisposition::Accepted,
    );
    append_motion_response(
        tick,
        Brain::Simulator,
        "shadow.motion_feedback",
        frame_id,
        EventTimes::observed(t_ms, t_ms),
        serde_json::json!({
            "dispatch_acknowledged": true,
            "encoder_odometry": "preserved_from_next_replayed_input",
            "imu": "preserved_from_next_replayed_input",
            "condition": "simulated_brainstem_ack",
        }),
        EventDisposition::Accepted,
    );
}

async fn run_shadow_flight(args: &ShadowFlightArgs) -> Result<(ShadowFlightManifest, ShadowFlightSummary)> {
    pete_cockpit::with_physical_actuator_transports_denied(run_shadow_flight_inner(args)).await
}

async fn run_shadow_flight_inner(
    args: &ShadowFlightArgs,
) -> Result<(ShadowFlightManifest, ShadowFlightSummary)> {
    if !pete_cockpit::physical_actuator_transports_are_denied() {
        anyhow::bail!("shadow flight requires the fail-closed physical actuator transport scope");
    }
    if !args.speed.is_finite() || args.speed <= 0.0 {
        anyhow::bail!("--speed must be finite and greater than zero");
    }
    if args.ledger_retained_frames == 0 || args.ledger_retained_transitions == 0 {
        anyhow::bail!("rolling ledger retention counts must be greater than zero");
    }
    if args.retained_events == 0 || args.retained_input_frames == 0 {
        anyhow::bail!("rolling event and input retention counts must be greater than zero");
    }
    if matches!(args.source, ShadowFlightSource::Capture | ShadowFlightSource::Ledger)
        && args.input.is_none()
    {
        anyhow::bail!("--input is required for capture and ledger shadow-flight sources");
    }
    for component in &args.allow_substitutions {
        if component != "higher_brain" {
            anyhow::bail!("unknown --allow-substitution component {component:?}");
        }
    }
    if args.higher_brain != ShadowHigherBrainMode::Disabled
        && !args
            .allow_substitutions
            .iter()
            .any(|component| component == "higher_brain")
    {
        anyhow::bail!(
            "required production component higher_brain is substituted by advisory_stub; rerun with --allow-substitution higher_brain only for an explicitly authorized test-double run"
        );
    }
    fs::create_dir_all(&args.output)?;
    let ledger_path = args.output.join("ledger");
    let mut inputs = VecDeque::new();
    let mut events = VecDeque::new();
    let mut aggregate = ShadowRunAggregate::default();
    let mut prior_ms = None;
    let source_identity;

    match args.source {
        ShadowFlightSource::Fixture | ShadowFlightSource::Seeded => {
            let effective_seed = if args.source == ShadowFlightSource::Fixture { 7 } else { args.seed };
            source_identity = format!("seeded:mixed-room:{effective_seed}");
            let runtime = shadow_runtime(
                &ledger_path,
                args.higher_brain,
                args.ledger_retained_frames,
                args.ledger_retained_transitions,
            );
            let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::MixedRoom, effective_seed));
            let mut runner = SimRunner::new(runtime, scenario.world, scenario.motors);
            for index in 0..args.ticks {
                let faults = shadow_faults_at(&args.faults, index as u64)?;
                apply_shadow_faults(&mut runner.world, &faults).await?;
                let input_frame_id = format!("{source_identity}:{index}");
                let runtime_frame_id = shadow_frame_uuid(&input_frame_id);
                runner.runtime.set_next_frame_id(runtime_frame_id);
                let mut observed = None;
                pete_experience::with_deterministic_identities(
                    runtime_frame_id,
                    runner.run_steps_observing_ticks(1, |snapshot, tick| {
                        observed = Some((
                            LiveViewState::runtime_tick_brain_events(snapshot, tick),
                            tick.frame.id,
                            tick.frame.t_ms,
                            shadow_tick_learning_provenance(tick),
                        ));
                    }),
                )
                .await?;
                tokio::task::yield_now().await;
                let (canonical, frame_id, frame_t_ms, (outcome_feedback_event_ids, inline_learning_samples_observed)) =
                    observed.context("production simulator emitted no tick")?;
                shadow_clock_wait(
                    args.clock,
                    args.speed,
                    prior_ms,
                    frame_t_ms,
                    args.pause_at.contains(&(index as u64)),
                )
                .await?;
                prior_ms = Some(frame_t_ms);
                record_shadow_events(
                    canonical,
                    frame_id,
                    frame_t_ms,
                    ShadowInputFrameProvenance {
                        index: index as u64,
                        input_frame_id,
                        runtime_frame_id,
                        t_ms: frame_t_ms,
                        clock_epochs: serde_json::json!({"simulator": effective_seed}),
                        faults,
                        outcome_feedback_event_ids,
                        inline_learning_samples_observed,
                    },
                    &mut events,
                    &mut inputs,
                    &mut aggregate,
                    args.retained_events,
                    args.retained_input_frames,
                )?;
            }
        }
        ShadowFlightSource::Capture => {
            let input = args.input.as_ref().expect("validated capture input");
            let reader = CaptureReader::open(input).await?;
            source_identity = format!("capture:{}", reader.manifest().id);
            let frames = reader.read_frames().await?;
            let mut runtime = shadow_runtime(
                &ledger_path,
                args.higher_brain,
                args.ledger_retained_frames,
                args.ledger_retained_transitions,
            );
            for record in frames.into_iter().take(args.ticks) {
                let input_frame_id = format!("{}:{}", source_identity, record.index);
                let runtime_frame_id = shadow_frame_uuid(&input_frame_id);
                runtime.set_next_frame_id(runtime_frame_id);
                let mut tick = pete_experience::with_deterministic_identities(
                    runtime_frame_id,
                    runtime.tick(
                        record.snapshot.to_now(record.t_ms),
                        Default::default(),
                        Vec::new(),
                    ),
                )
                .await?;
                append_replay_outcomes(&mut tick);
                pete_runtime::queue_actuator_outcome_feedback(&mut runtime, &tick);
                let (outcome_feedback_event_ids, inline_learning_samples_observed) =
                    shadow_tick_learning_provenance(&tick);
                shadow_clock_wait(args.clock, args.speed, prior_ms, record.t_ms, args.pause_at.contains(&record.index)).await?;
                prior_ms = Some(record.t_ms);
                record_shadow_tick(
                    &WorldSnapshot::default(),
                    &tick,
                    ShadowInputFrameProvenance {
                        index: record.index,
                        input_frame_id,
                        runtime_frame_id,
                        t_ms: record.t_ms,
                        clock_epochs: record.stream_metadata.unwrap_or(serde_json::Value::Null),
                        faults: Vec::new(),
                        outcome_feedback_event_ids,
                        inline_learning_samples_observed,
                    },
                    &mut events,
                    &mut inputs,
                    &mut aggregate,
                    args.retained_events,
                    args.retained_input_frames,
                )?;
            }
        }
        ShadowFlightSource::Ledger => {
            let input = args.input.as_ref().expect("validated ledger input");
            source_identity = format!("ledger:{}", input.display());
            let frames = if input.join("rolling").exists() {
                RollingLedger::new(input, usize::MAX, usize::MAX)
                    .frames()
                    .await?
            } else {
                JsonlLedger::new(input).frames().await?
            };
            let mut runtime = shadow_runtime(
                &ledger_path,
                args.higher_brain,
                args.ledger_retained_frames,
                args.ledger_retained_transitions,
            );
            for (index, frame) in frames.into_iter().take(args.ticks).enumerate() {
                let input_frame_id = format!("ledger-frame:{}", frame.id);
                runtime.set_next_frame_id(frame.id);
                let clock_epochs = frame
                    .now
                    .extensions
                    .get("clock_epochs")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let mut tick = pete_experience::with_deterministic_identities(
                    frame.id,
                    runtime.tick(frame.now, Default::default(), Vec::new()),
                )
                .await?;
                append_replay_outcomes(&mut tick);
                pete_runtime::queue_actuator_outcome_feedback(&mut runtime, &tick);
                let (outcome_feedback_event_ids, inline_learning_samples_observed) =
                    shadow_tick_learning_provenance(&tick);
                shadow_clock_wait(args.clock, args.speed, prior_ms, frame.t_ms, args.pause_at.contains(&(index as u64))).await?;
                prior_ms = Some(frame.t_ms);
                record_shadow_tick(
                    &WorldSnapshot::default(),
                    &tick,
                    ShadowInputFrameProvenance {
                        index: index as u64,
                        input_frame_id,
                        runtime_frame_id: frame.id,
                        t_ms: frame.t_ms,
                        clock_epochs,
                        faults: Vec::new(),
                        outcome_feedback_event_ids,
                        inline_learning_samples_observed,
                    },
                    &mut events,
                    &mut inputs,
                    &mut aggregate,
                    args.retained_events,
                    args.retained_input_frames,
                )?;
            }
        }
    }

    let input_bytes = inputs.iter().map(serde_json::to_string).collect::<std::result::Result<Vec<_>, _>>()?.join("\n") + "\n";
    let event_bytes = events.iter().map(serde_json::to_string).collect::<std::result::Result<Vec<_>, _>>()?.join("\n") + "\n";
    let hash = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
    let event_type_counts = aggregate.event_type_counts.clone();
    let chain = ["evidence", "interpretation", "belief_update", "proposal", "gate_decision", "command", "outcome"];
    let summary = ShadowFlightSummary {
        ticks_completed: aggregate.ticks_completed,
        canonical_events: aggregate.canonical_events,
        full_causal_chain_observed: chain.iter().all(|kind| event_type_counts.contains_key(*kind)),
        simulated_outcomes: *event_type_counts.get("outcome").unwrap_or(&0),
        outcome_feedback_frames: aggregate.outcome_feedback_frames,
        inline_learning_samples_observed: aggregate.inline_learning_samples_observed,
        higher_brain_advice_responses: aggregate.higher_brain_advice_responses,
        higher_brain_advisory_actions_discarded: aggregate
            .higher_brain_advisory_actions_discarded,
        higher_brain_authority_violations: aggregate.higher_brain_authority_violations,
        local_authority_sha256: format!("{:x}", aggregate.local_authority.clone().finalize()),
        safety_gate_events: *event_type_counts.get("gate_decision").unwrap_or(&0),
        event_type_counts,
        events_sha256: hash(event_bytes.as_bytes()),
    };
    if !summary.full_causal_chain_observed || summary.simulated_outcomes == 0 || summary.safety_gate_events == 0 || summary.higher_brain_authority_violations != 0 {
        anyhow::bail!("shadow flight did not preserve the complete safe causal path: {summary:?}");
    }
    if args.higher_brain != ShadowHigherBrainMode::Disabled
        && summary.higher_brain_advice_responses == 0
    {
        anyhow::bail!("advisory higher-brain test double produced no accepted advisory response");
    }
    if args.higher_brain == ShadowHigherBrainMode::AdversarialMotion
        && summary.higher_brain_advisory_actions_discarded == 0
    {
        anyhow::bail!("adversarial direct-motion advice was not visibly discarded at the advisory boundary");
    }
    let summary_bytes = serde_json::to_vec_pretty(&summary)?;
    fs::write(args.output.join("input-frames.jsonl"), &input_bytes)?;
    fs::write(args.output.join("events.jsonl"), &event_bytes)?;
    fs::write(args.output.join("summary.json"), &summary_bytes)?;
    let manifest = ShadowFlightManifest {
        schema_version: 2,
        status: "complete".into(),
        source: args.source,
        source_identity,
        seed: match args.source {
            ShadowFlightSource::Fixture => Some(7),
            ShadowFlightSource::Seeded => Some(args.seed),
            ShadowFlightSource::Capture | ShadowFlightSource::Ledger => None,
        },
        clock_mode: args.clock,
        higher_brain_mode: args.higher_brain,
        speed: args.speed,
        requested_ticks: args.ticks,
        completed_ticks: summary.ticks_completed,
        production_components: vec!["pete_runtime::MinimalRuntime".into(), "pete_conductor::SimpleConductor".into(), "pete_autonomic::SimpleSafety".into(), "pete_runtime::SimRunner".into(), "pete_server::LiveViewState::runtime_tick_brain_events".into(), "pete_ledger::RollingLedger".into(), "pete_memory::DurableExperienceStore".into()],
        substitutions: match args.higher_brain {
            ShadowHigherBrainMode::Disabled => Vec::new(),
            ShadowHigherBrainMode::AdvisoryStub | ShadowHigherBrainMode::AdversarialMotion => vec![
                "higher_brain: production provider replaced by explicitly authorized deterministic advisory-only test double".into(),
            ],
        },
        actuator_transport: "in_process_simulator_only".into(),
        network_required: false,
        physical_hardware_required: false,
        lidar_required: false,
        ledger_retained_frames: args.ledger_retained_frames,
        ledger_retained_transitions: args.ledger_retained_transitions,
        event_retention_limit: args.retained_events,
        input_retention_limit: args.retained_input_frames,
        retained_event_records: events.len(),
        dropped_event_records: summary.canonical_events.saturating_sub(events.len() as u64),
        retained_input_frames: inputs.len(),
        dropped_input_frames: summary.ticks_completed.saturating_sub(inputs.len() as u64),
        input_frames_path: "input-frames.jsonl".into(),
        events_path: "events.jsonl".into(),
        summary_path: "summary.json".into(),
        input_frames_sha256: hash(input_bytes.as_bytes()),
        events_sha256: summary.events_sha256.clone(),
        summary_sha256: hash(&summary_bytes),
    };
    fs::write(args.output.join("manifest.json"), serde_json::to_vec_pretty(&manifest)?)?;
    Ok((manifest, summary))
}

async fn run_shadow_flight_command(args: ShadowFlightArgs) -> Result<()> {
    match run_shadow_flight(&args).await {
        Ok((manifest, summary)) => {
            println!("shadow flight complete: {} ticks, {} canonical events, manifest {}", summary.ticks_completed, summary.canonical_events, args.output.join("manifest.json").display());
            debug_assert_eq!(manifest.status, "complete");
            Ok(())
        }
        Err(error) => {
            fs::create_dir_all(&args.output)?;
            fs::write(
                args.output.join("failure.json"),
                serde_json::to_vec_pretty(&serde_json::json!({
                    "status": "failed",
                    "error": error.to_string(),
                    "source": args.source,
                    "physical_transport_open": false,
                }))?,
            )?;
            Err(error)
        }
    }
}
