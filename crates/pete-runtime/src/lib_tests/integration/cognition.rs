#[test]
fn missing_experience_checkpoint_returns_no_latent_yet() {
    let config: BehaviorRegistryConfig = toml::from_str(
        r#"
            [behavior.experience]
            regime = "shadow_infer"
            hardcoded = "experience.no_latent_yet"
            model = "experience.autoencoder.v0"
            checkpoint = "/tmp/pete-missing-experience-checkpoint"
            fallback = "use_hardcoded"
            "#,
    )
    .unwrap();
    let mut stack = RuntimeModelStack::from_behavior_config(&config).unwrap();
    let now = Now::blank(100, test_body(1.0, 1.0, 0.8, 100));
    let run = stack
        .behaviors
        .experience
        .infer(&ExperienceBehaviorInput::from_now(&now), now.t_ms)
        .unwrap();

    assert_eq!(run.record.regime, BehaviorRegime::ShadowInfer);
    assert!(run.record.hardcoded_output.is_some());
    assert!(run.record.model_output.is_none());
    assert_eq!(run.chosen, run.record.hardcoded_output.unwrap());
    assert!(run.chosen.latent.z.is_empty());
    assert_eq!(run.chosen.confidence, 0.0);
}

#[tokio::test]
async fn shared_reign_queue_controls_next_sim_tick() {
    let root = test_ledger_root("sim-runner-shared-reign");
    let ledger = JsonlLedger::new(&root);
    let queue = Arc::new(Mutex::new(ReignQueue::default()));
    queue.lock().unwrap().push(test_reign_input(
        7,
        ReignMode::Direct,
        ReignCommand::Turn {
            direction: pete_actions::TurnDir::Left,
            intensity: 0.5,
            duration_ms: 500,
        },
        2_000,
    ));
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let runtime = MinimalRuntime::with_reign_queue(
        ledger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
        queue,
    );
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.8, 7));
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(1).await.unwrap();
    let snapshot = runner.world.snapshot().await.unwrap();
    let frames = JsonlLedger::new(&root).recent(5).await.unwrap();
    let frame = frames.last().unwrap();

    assert!(snapshot.body.odometry.heading_rad > 0.0);
    assert!(frame.now.reign.active);
    assert!(frame
        .sensations
        .iter()
        .any(|sensation| sensation.kind == "reign.command"));
    assert!(frame.reign_input.is_some());
    assert!(frame
        .reign_outcome
        .as_ref()
        .map(|outcome| outcome.accepted_by_conductor)
        .unwrap_or(false));
}

#[tokio::test]
async fn direct_reign_reverse_drives_sim_while_stuck_active() {
    let root = test_ledger_root("sim-runner-reign-reverse-interrupts-stuck");
    let ledger = JsonlLedger::new(&root);
    let queue = Arc::new(Mutex::new(ReignQueue::default()));
    queue.lock().unwrap().push(test_reign_input(
        7,
        ReignMode::Direct,
        ReignCommand::Reverse {
            intensity: 0.5,
            duration_ms: 500,
        },
        2_000,
    ));
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let runtime = MinimalRuntime::with_reign_queue(
        ledger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
        queue,
    );
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 1.0, 7));
    let mut runner = SimRunner::new(runtime, world, motors);
    runner.stuck.active = true;
    runner.stuck.phase = RecoveryPhase::Stop;
    runner.stuck.phase_ticks_remaining = 1;
    runner.stuck.turn_sign = 1.0;

    let mut observed_debug = None;
    runner
        .run_steps_observing(1, |snapshot| {
            observed_debug = snapshot.action_debug.clone();
        })
        .await
        .unwrap();
    let debug = observed_debug.unwrap();
    let motion = debug.get("motion_sent_to_sim").cloned().unwrap();

    let motion = serde_json::from_value::<MotionCommand>(motion.clone())
        .unwrap_or_else(|error| panic!("motion decode failed: {error}; debug={debug}"));
    assert_eq!(motion, MotionCommand::Forward { speed_m_s: -0.5 });
}

#[tokio::test]
async fn column_trap_scenario_recovers_within_budget() {
    let root = test_ledger_root("sim-runner-column-trap-recovery");
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger, SimpleConductor::default());
    let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ColumnTrap, 7));
    let start = (
        scenario.metadata.body.odometry.x_m,
        scenario.metadata.body.odometry.y_m,
    );
    let mut runner = SimRunner::new(runtime, scenario.world, scenario.motors);
    let mut saw_column = false;
    let mut recovered = false;
    let mut last_skill_status = None;

    runner
        .run_steps_observing_ticks(90, |snapshot, tick| {
            if tick.skill_status.is_some() {
                last_skill_status = tick.skill_status.clone();
            }
            if let Some(stuck) = snapshot
                .extensions
                .iter()
                .find(|extension| extension.name == "sim.stuck")
            {
                saw_column |= stuck.values.get(10).copied() == Some(3.0);
                recovered |= stuck.values.get(7).copied() == Some(1.0);
            }
        })
        .await
        .unwrap();
    let end = runner.world.body();
    let distance = distance_between_points(start, (end.odometry.x_m, end.odometry.y_m));

    assert!(saw_column);
    assert!(recovered, "last Lua skill status was {last_skill_status:?}");
    assert!(distance > 0.10, "distance after recovery was {distance}");
}

#[derive(Clone, Copy, Debug, Default)]
struct TrapRunMetrics {
    collision_frames: usize,
    stuck_frames: usize,
    recovered: bool,
    distance_m: f32,
}

async fn run_column_trap_metrics<C>(ledger_name: &str, conductor: C, steps: usize) -> TrapRunMetrics
where
    C: Conductor + Send + 'static,
{
    let root = test_ledger_root(ledger_name);
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger, conductor);
    let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ColumnTrap, 7));
    let start = (
        scenario.metadata.body.odometry.x_m,
        scenario.metadata.body.odometry.y_m,
    );
    let mut runner = SimRunner::new(runtime, scenario.world, scenario.motors);
    let mut metrics = TrapRunMetrics::default();

    runner
        .run_steps_observing(steps, |snapshot| {
            let flags = &snapshot.body.flags;
            if flags.wall
                || flags.bump_left
                || flags.bump_right
                || flags.cliff_front_left
                || flags.cliff_front_right
            {
                metrics.collision_frames += 1;
            }
            if let Some(stuck) = snapshot
                .extensions
                .iter()
                .find(|extension| extension.name == "sim.stuck")
            {
                metrics.stuck_frames +=
                    (stuck.values.first().copied().unwrap_or_default() > 0.0) as usize;
                metrics.recovered |= stuck.values.get(7).copied() == Some(1.0);
            }
        })
        .await
        .unwrap();
    let end = runner.world.body();
    metrics.distance_m = distance_between_points(start, (end.odometry.x_m, end.odometry.y_m));
    metrics
}

#[tokio::test]
async fn column_trap_recovery_beats_plain_explore_baseline() {
    let plain = run_column_trap_metrics(
        "sim-runner-column-trap-plain-explore",
        FixedConductor::new(ActionPrimitive::Explore {
            style: ExploreStyle::RandomWalk,
            duration_ms: 1_000,
        }),
        120,
    )
    .await;
    let recovered = run_column_trap_metrics(
        "sim-runner-column-trap-simple-recovery-comparison",
        SimpleConductor::default(),
        120,
    )
    .await;

    assert!(
        recovered.recovered,
        "expected recovery event, got {recovered:?}"
    );
    assert!(
        recovered.collision_frames < plain.collision_frames / 2,
        "recovery should reduce repeated collision frames; plain={plain:?} recovered={recovered:?}"
    );
    assert!(
            recovered.distance_m > plain.distance_m,
            "recovery should make more progress than plain explore; plain={plain:?} recovered={recovered:?}"
        );
    assert!(
        recovered.stuck_frames < plain.stuck_frames,
        "recovery should reduce repeated stuck frames; plain={plain:?} recovered={recovered:?}"
    );
}

#[derive(Clone, Debug)]
struct FixedConductor {
    action: ActionPrimitive,
}

#[derive(Clone, Debug, Default)]
struct FixedRecall {
    bundle: RecallBundle,
}

#[async_trait::async_trait]
impl Recall for FixedRecall {
    async fn recall(&self, _query: RecallQuery) -> Result<RecallBundle> {
        Ok(self.bundle.clone())
    }
}

impl FixedConductor {
    fn new(action: ActionPrimitive) -> Self {
        Self { action }
    }
}

impl Conductor for FixedConductor {
    fn choose(&mut self, _input: ConductorInput) -> Result<ActionPrimitive> {
        Ok(self.action.clone())
    }
}

#[derive(Clone, Debug)]
struct FixedLlmAgent {
    action: ActionPrimitive,
}

#[async_trait::async_trait]
impl LlmAgent for FixedLlmAgent {
    async fn combobulate(
        &mut self,
        _now: &Now,
        _impressions: &[Impression],
        _embodied: Option<&EmbodiedContext>,
        _z: &ExperienceLatent,
        _futures: &[FuturePrediction],
        _recall_summary: &str,
    ) -> Result<Option<Combobulation>> {
        Ok(None)
    }

    async fn maybe_tick(
        &mut self,
        _now: &Now,
        _embodied: Option<&EmbodiedContext>,
        _z: &ExperienceLatent,
        _futures: &[FuturePrediction],
        _recall_summary: &str,
        _awareness_summary: Option<&str>,
    ) -> Result<LlmTickResult> {
        Ok(LlmTickResult {
            sense: pete_now::LlmSense {
                schema_version: 1,
                command_summary: Some("test command".to_string()),
                critique: None,
                confidence: 1.0,
            },
            conscious_command: Some(ConsciousCommand {
                summary: "test command".to_string(),
                action: Some(self.action.clone()),
            }),
            decision: Some(LlmDecision {
                summary: "test command".to_string(),
                action: Some(self.action.clone()),
                confidence: 1.0,
                ..LlmDecision::default()
            }),
            teaching: Vec::new(),
        })
    }

    async fn scientific_review(
        &mut self,
        _request: &LlmReviewRequest,
    ) -> Result<Option<LlmScientificReview>> {
        Ok(None)
    }
}

#[derive(Debug, Default)]
struct SlowAdvisoryAgent;

#[async_trait::async_trait]
impl LlmAgent for SlowAdvisoryAgent {
    async fn combobulate(
        &mut self,
        _now: &Now,
        _impressions: &[Impression],
        _embodied: Option<&EmbodiedContext>,
        _z: &ExperienceLatent,
        _futures: &[FuturePrediction],
        _recall_summary: &str,
    ) -> Result<Option<Combobulation>> {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        Ok(Some(Combobulation {
            summary: "historical doorway hypothesis".to_string(),
            confidence: 0.8,
        }))
    }

    async fn maybe_tick(
        &mut self,
        _now: &Now,
        _embodied: Option<&EmbodiedContext>,
        _z: &ExperienceLatent,
        _futures: &[FuturePrediction],
        _recall_summary: &str,
        _awareness_summary: Option<&str>,
    ) -> Result<LlmTickResult> {
        Ok(LlmTickResult {
            sense: pete_now::LlmSense {
                schema_version: 1,
                critique: Some("test the doorway hypothesis".to_string()),
                confidence: 0.8,
                ..pete_now::LlmSense::default()
            },
            decision: Some(LlmDecision {
                action: Some(ActionPrimitive::Go {
                    intensity: 1.0,
                    duration_ms: 5_000,
                }),
                ..LlmDecision::default()
            }),
            ..LlmTickResult::default()
        })
    }

    async fn scientific_review(
        &mut self,
        _request: &LlmReviewRequest,
    ) -> Result<Option<LlmScientificReview>> {
        Ok(None)
    }
}

fn test_runtime<C>(
    ledger: JsonlLedger,
    conductor: C,
) -> MinimalRuntime<
    JsonlLedger,
    InMemoryExperienceStore,
    InMemoryExperienceStore,
    C,
    SimpleSafety,
    pete_llm::NoopLlmAgent,
>
where
    C: Conductor,
{
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    MinimalRuntime::new(
        ledger,
        memory,
        recall,
        conductor,
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
    )
}

async fn finished_cognition_task() -> JoinHandle<Result<(Option<Combobulation>, LlmTickResult)>> {
    let task = tokio::spawn(async {
        Ok((
            None,
            LlmTickResult {
                sense: pete_now::LlmSense {
                    schema_version: 1,
                    command_summary: Some("completed cognition".to_string()),
                    confidence: 1.0,
                    ..pete_now::LlmSense::default()
                },
                ..LlmTickResult::default()
            },
        ))
    });
    tokio::task::yield_now().await;
    assert!(task.is_finished(), "fixture cognition task should be ready");
    task
}

fn cognition_test_inputs() -> (
    EmbodiedContext,
    ExperienceLatent,
    Vec<FuturePrediction>,
    Vec<String>,
) {
    (
        EmbodiedContext::default(),
        ExperienceLatent::default(),
        Vec::new(),
        Vec::new(),
    )
}
