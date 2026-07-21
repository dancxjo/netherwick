async fn capture_sim(args: CaptureSimArgs) -> Result<()> {
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let runtime = MinimalRuntime::with_default_events(
        NoopLedger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        configured_llm_agent(&args.llm)?,
    );
    let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::MixedRoom, args.seed));
    let world = scenario.world;
    let motors = scenario.motors;
    let mut runner = SimRunner::new(runtime, world, motors);
    let mut snapshots = Vec::new();
    runner
        .run_steps_observing(args.steps, |snapshot| snapshots.push(snapshot.clone()))
        .await?;

    let mut writer =
        CaptureWriter::create(&args.out, CaptureSource::Sim, Some(args.tick_ms)).await?;
    writer.manifest_mut().scenario = Some(scenario.metadata);
    for snapshot in snapshots {
        let t_ms = snapshot.body.last_update_ms;
        writer.append_snapshot(t_ms, snapshot, Vec::new()).await?;
    }
    let manifest = writer.finish().await?;

    println!(
        "capture complete: {} frames, seed {}, out {}, tick_ms {:?}",
        manifest.frame_count, args.seed, args.out, manifest.tick_ms
    );
    Ok(())
}

async fn replay_capture(args: ReplayCaptureArgs) -> Result<()> {
    let reader = CaptureReader::open(&args.capture).await?;
    let ledger = JsonlLedger::new(&args.ledger);
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let runtime = MinimalRuntime::with_default_events(
        ledger.clone(),
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        configured_llm_agent(&args.llm)?,
    );
    let mut runner = CaptureReplayRunner::new(runtime, reader);
    let summary = runner.replay().await?;
    let transitions = ledger.transitions().await?;

    println!(
        "replay complete: {} frames replayed, {} runtime ticks, ledger {}, {} transitions written",
        summary.frames_replayed,
        summary.runtime_ticks,
        args.ledger,
        transitions.len()
    );
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
enum CounterfactualEdit {
    MoveObject {
        kind: CounterfactualObjectKind,
        id: Option<String>,
        x_m: f32,
        y_m: f32,
    },
    RemoveObstacle {
        id: Option<String>,
    },
    AddObstacle {
        x_m: f32,
        y_m: f32,
        radius_m: f32,
    },
    SetBattery {
        value: f32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CounterfactualObjectKind {
    Charger,
    Person,
    Speaker,
}

#[derive(Clone, Debug, PartialEq)]
enum CounterfactualPolicy {
    Baseline,
    Stop,
    TurnLeftOnDanger,
    TurnRightOnDanger,
    SeekCharge,
    RandomWalk { seed: u64 },
    Scripted(Vec<ActionPrimitive>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct CounterfactualReport {
    schema_version: u32,
    source_capture: String,
    reconstructable: bool,
    edits: Vec<String>,
    policy: String,
    steps: usize,
    summary: CounterfactualSummary,
    warnings: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct CounterfactualSummary {
    collisions: usize,
    charging_ticks: usize,
    battery_delta: f32,
    distance_traveled: f32,
    final_distance_to_charger_m: Option<f32>,
}

#[derive(Clone, Debug)]
struct CounterfactualConductor {
    policy: CounterfactualPolicy,
    baseline: SimpleConductor,
    rng: StdRng,
    scripted_index: usize,
}

impl CounterfactualConductor {
    fn new(policy: CounterfactualPolicy) -> Self {
        let seed = match policy {
            CounterfactualPolicy::RandomWalk { seed } => seed,
            _ => 0,
        };
        Self {
            policy,
            baseline: SimpleConductor::default(),
            rng: StdRng::seed_from_u64(seed),
            scripted_index: 0,
        }
    }
}

impl Conductor for CounterfactualConductor {
    fn choose(&mut self, input: ConductorInput) -> Result<ActionPrimitive> {
        match &self.policy {
            CounterfactualPolicy::Baseline => self.baseline.choose(input),
            CounterfactualPolicy::Stop => Ok(ActionPrimitive::Stop),
            CounterfactualPolicy::TurnLeftOnDanger => {
                if danger_present(&input) {
                    Ok(turn_action(TurnDir::Left))
                } else {
                    self.baseline.choose(input)
                }
            }
            CounterfactualPolicy::TurnRightOnDanger => {
                if danger_present(&input) {
                    Ok(turn_action(TurnDir::Right))
                } else {
                    self.baseline.choose(input)
                }
            }
            CounterfactualPolicy::SeekCharge => Ok(ActionPrimitive::Approach {
                target: ApproachTarget::Charger,
            }),
            CounterfactualPolicy::RandomWalk { .. } => {
                if self.rng.gen_bool(0.25) || danger_present(&input) {
                    let direction = if self.rng.gen_bool(0.5) {
                        TurnDir::Left
                    } else {
                        TurnDir::Right
                    };
                    Ok(turn_action(direction))
                } else {
                    Ok(ActionPrimitive::Go {
                        intensity: 0.25,
                        duration_ms: 1_000,
                    })
                }
            }
            CounterfactualPolicy::Scripted(actions) => {
                let action = actions
                    .get(self.scripted_index)
                    .cloned()
                    .unwrap_or(ActionPrimitive::Stop);
                self.scripted_index = self.scripted_index.saturating_add(1);
                Ok(action)
            }
        }
    }
}

async fn replay_counterfactual(args: ReplayCounterfactualArgs) -> Result<()> {
    let reader = CaptureReader::open(&args.capture).await?;
    let manifest = reader.manifest().clone();
    let Some(mut metadata) = manifest.scenario.clone() else {
        anyhow::bail!(
            "passive captures without reconstructable sim metadata cannot yet be counterfactually replayed"
        );
    };
    let frames = reader.read_frames().await?;
    let steps = args.steps.unwrap_or(frames.len()).max(1);
    let edits = args
        .edit
        .iter()
        .map(|edit| parse_counterfactual_edit(edit))
        .collect::<Result<Vec<_>>>()?;
    let mut warnings = Vec::new();
    apply_counterfactual_edits(&mut metadata, &edits, &mut warnings)?;
    let policy = parse_counterfactual_policy(&args.policy, args.actions.as_deref())?;

    let (mut world, motors) =
        pete_sim::VirtualWorld::new_with_cockpit(metadata.seed, metadata.arena);
    world.set_body(metadata.body.clone());
    world.set_objects(metadata.objects.clone());

    if let Some(ledger_path) = &args.out_ledger {
        let runtime =
            counterfactual_runtime(JsonlLedger::new(ledger_path), policy.clone(), &args.llm)?;
        let report = run_counterfactual_sim(
            runtime, world, motors, &metadata, &manifest, &args, steps, policy, warnings,
        )
        .await?;
        write_or_print_counterfactual_report(&args, &report)?;
        let transitions = JsonlLedger::new(ledger_path).transitions().await?;
        println!(
            "counterfactual replay complete: {} steps, ledger {}, transitions {}, report {}",
            steps,
            ledger_path,
            transitions.len(),
            args.out_report.as_deref().unwrap_or("stdout")
        );
    } else {
        let runtime = counterfactual_runtime(NoopLedger, policy.clone(), &args.llm)?;
        let report = run_counterfactual_sim(
            runtime, world, motors, &metadata, &manifest, &args, steps, policy, warnings,
        )
        .await?;
        write_or_print_counterfactual_report(&args, &report)?;
        println!(
            "counterfactual replay complete: {} steps, report {}",
            steps,
            args.out_report.as_deref().unwrap_or("stdout")
        );
    }
    Ok(())
}

fn counterfactual_runtime<L>(
    ledger: L,
    policy: CounterfactualPolicy,
    llm: &LlmArgs,
) -> Result<
    MinimalRuntime<
        L,
        InMemoryExperienceStore,
        InMemoryExperienceStore,
        CounterfactualConductor,
        SimpleSafety,
        ConfiguredLlmAgent,
    >,
>
where
    L: LedgerWriter + Sync + Send,
{
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    Ok(MinimalRuntime::with_default_events(
        ledger,
        memory,
        recall,
        CounterfactualConductor::new(policy),
        SimpleSafety::default(),
        configured_llm_agent(llm)?,
    ))
}

async fn run_counterfactual_sim<R>(
    runtime: R,
    world: pete_sim::VirtualWorld,
    motors: pete_sim::SimCockpit,
    metadata: &pete_sim::ScenarioMetadata,
    manifest: &pete_worldlab::CaptureManifest,
    args: &ReplayCounterfactualArgs,
    steps: usize,
    policy: CounterfactualPolicy,
    warnings: Vec<String>,
) -> Result<CounterfactualReport>
where
    R: RuntimeLoop + Send,
{
    let mut metrics = EpisodeMetricBuilder::new(
        metadata.kind,
        metadata.clone(),
        0,
        metadata.seed,
        args.out_ledger.clone(),
        Some(args.capture.clone()),
    );
    let mut runner = SimRunner::new(runtime, world, motors);
    runner.tick_ms = manifest.tick_ms.unwrap_or(100);
    runner
        .run_steps_observing_ticks(steps, |snapshot, tick| metrics.observe(snapshot, tick))
        .await?;
    let episode = metrics.finish();
    Ok(CounterfactualReport {
        schema_version: 1,
        source_capture: args.capture.clone(),
        reconstructable: true,
        edits: args.edit.clone(),
        policy: counterfactual_policy_label(&policy),
        steps: episode.ticks,
        summary: CounterfactualSummary {
            collisions: episode.collisions,
            charging_ticks: episode.charging_ticks,
            battery_delta: episode.battery_delta,
            distance_traveled: episode.distance_traveled_m,
            final_distance_to_charger_m: episode.final_distance_to_charger_m,
        },
        warnings,
    })
}

fn write_or_print_counterfactual_report(
    args: &ReplayCounterfactualArgs,
    report: &CounterfactualReport,
) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(report)?;
    if let Some(out) = &args.out_report {
        if let Some(parent) = Path::new(out).parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(out, bytes)?;
    } else {
        println!("{}", String::from_utf8_lossy(&bytes));
    }
    Ok(())
}

fn parse_counterfactual_policy(
    policy: &str,
    actions: Option<&str>,
) -> Result<CounterfactualPolicy> {
    if let Some(actions) = actions {
        return Ok(CounterfactualPolicy::Scripted(parse_scripted_actions(
            actions,
        )?));
    }
    if policy == "baseline" {
        Ok(CounterfactualPolicy::Baseline)
    } else if policy == "stop" {
        Ok(CounterfactualPolicy::Stop)
    } else if policy == "turn-left-on-danger" {
        Ok(CounterfactualPolicy::TurnLeftOnDanger)
    } else if policy == "turn-right-on-danger" {
        Ok(CounterfactualPolicy::TurnRightOnDanger)
    } else if policy == "seek-charge" {
        Ok(CounterfactualPolicy::SeekCharge)
    } else if policy == "random-walk" {
        Ok(CounterfactualPolicy::RandomWalk { seed: 0 })
    } else if let Some(rest) = policy.strip_prefix("random-walk:seed=") {
        Ok(CounterfactualPolicy::RandomWalk {
            seed: rest.parse().context("invalid random-walk seed")?,
        })
    } else {
        anyhow::bail!("unknown counterfactual policy '{policy}'")
    }
}

fn counterfactual_policy_label(policy: &CounterfactualPolicy) -> String {
    match policy {
        CounterfactualPolicy::Baseline => "baseline".to_string(),
        CounterfactualPolicy::Stop => "stop".to_string(),
        CounterfactualPolicy::TurnLeftOnDanger => "turn-left-on-danger".to_string(),
        CounterfactualPolicy::TurnRightOnDanger => "turn-right-on-danger".to_string(),
        CounterfactualPolicy::SeekCharge => "seek-charge".to_string(),
        CounterfactualPolicy::RandomWalk { seed } => format!("random-walk:seed={seed}"),
        CounterfactualPolicy::Scripted(_) => "scripted".to_string(),
    }
}

fn parse_scripted_actions(actions: &str) -> Result<Vec<ActionPrimitive>> {
    actions
        .split(',')
        .map(|token| match token.trim() {
            "forward" | "go" => Ok(ActionPrimitive::Go {
                intensity: 0.25,
                duration_ms: 1_000,
            }),
            "left" => Ok(turn_action(TurnDir::Left)),
            "right" => Ok(turn_action(TurnDir::Right)),
            "stop" => Ok(ActionPrimitive::Stop),
            "dock" => Ok(ActionPrimitive::Dock),
            "wander" | "random-walk" => Ok(ActionPrimitive::Explore {
                style: ExploreStyle::RandomWalk,
                duration_ms: 1_000,
            }),
            other => anyhow::bail!("unknown scripted action '{other}'"),
        })
        .collect()
}

fn turn_action(direction: TurnDir) -> ActionPrimitive {
    ActionPrimitive::Turn {
        direction,
        intensity: 0.6,
        duration_ms: 1_000,
    }
}

fn danger_present(input: &ConductorInput) -> bool {
    input.body.flags.bump_left
        || input.body.flags.bump_right
        || input.body.flags.wall
        || input.body.flags.cliff_left
        || input.body.flags.cliff_front_left
        || input.body.flags.cliff_front_right
        || input.body.flags.cliff_right
        || input.drives.danger_avoidance >= 0.5
}

fn parse_counterfactual_edit(input: &str) -> Result<CounterfactualEdit> {
    let (name, rest) = input
        .split_once(':')
        .map(|(name, rest)| (name.trim(), rest.trim()))
        .unwrap_or((input.trim(), ""));
    let fields = parse_edit_fields(rest)?;
    match name {
        "move-charger" => parse_move_edit(CounterfactualObjectKind::Charger, fields),
        "move-person" => parse_move_edit(CounterfactualObjectKind::Person, fields),
        "move-speaker" => parse_move_edit(CounterfactualObjectKind::Speaker, fields),
        "remove-obstacle" => Ok(CounterfactualEdit::RemoveObstacle {
            id: fields.get("id").cloned(),
        }),
        "add-obstacle" => Ok(CounterfactualEdit::AddObstacle {
            x_m: required_f32(&fields, "x")?,
            y_m: required_f32(&fields, "y")?,
            radius_m: required_f32(&fields, "radius")?,
        }),
        "set-battery" => Ok(CounterfactualEdit::SetBattery {
            value: required_f32(&fields, "value")?.clamp(0.0, 1.0),
        }),
        _ => anyhow::bail!("unknown counterfactual edit '{name}'"),
    }
}

fn parse_move_edit(
    kind: CounterfactualObjectKind,
    fields: HashMap<String, String>,
) -> Result<CounterfactualEdit> {
    Ok(CounterfactualEdit::MoveObject {
        kind,
        id: fields.get("id").cloned(),
        x_m: required_f32(&fields, "x")?,
        y_m: required_f32(&fields, "y")?,
    })
}

fn parse_edit_fields(input: &str) -> Result<HashMap<String, String>> {
    let mut fields = HashMap::new();
    if input.trim().is_empty() {
        return Ok(fields);
    }
    for part in input.split(',') {
        let (key, value) = part
            .split_once('=')
            .with_context(|| format!("invalid edit field '{part}', expected key=value"))?;
        fields.insert(key.trim().to_string(), value.trim().to_string());
    }
    Ok(fields)
}

fn required_f32(fields: &HashMap<String, String>, key: &str) -> Result<f32> {
    fields
        .get(key)
        .with_context(|| format!("missing required edit field '{key}'"))?
        .parse()
        .with_context(|| format!("invalid float for edit field '{key}'"))
}

fn apply_counterfactual_edits(
    metadata: &mut pete_sim::ScenarioMetadata,
    edits: &[CounterfactualEdit],
    warnings: &mut Vec<String>,
) -> Result<()> {
    for edit in edits {
        match edit {
            CounterfactualEdit::MoveObject { kind, id, x_m, y_m } => {
                let object = find_counterfactual_object_mut(&mut metadata.objects, *kind, id)?;
                object.x_m = *x_m;
                object.y_m = *y_m;
                if id.is_none() {
                    warnings.push(format!(
                        "{} edit used first matching object because no id was provided",
                        object_kind_label(*kind)
                    ));
                }
            }
            CounterfactualEdit::RemoveObstacle { id } => {
                let index = metadata
                    .objects
                    .iter()
                    .position(|object| {
                        matches!(object.kind, pete_sim::SimObjectKind::Obstacle)
                            && id.as_ref().map(|id| id == &object.id).unwrap_or(true)
                    })
                    .with_context(|| {
                        if let Some(id) = id {
                            format!("obstacle '{id}' not found")
                        } else {
                            "no obstacle found to remove".to_string()
                        }
                    })?;
                metadata.objects.remove(index);
                if id.is_none() {
                    warnings.push(
                        "remove-obstacle edit used first obstacle because no id was provided"
                            .to_string(),
                    );
                }
            }
            CounterfactualEdit::AddObstacle { x_m, y_m, radius_m } => {
                let index = metadata
                    .objects
                    .iter()
                    .filter(|object| matches!(object.kind, pete_sim::SimObjectKind::Obstacle))
                    .count();
                metadata.objects.push(pete_sim::SimObject::obstacle(
                    format!("counterfactual-obstacle-{index}"),
                    format!("counterfactual obstacle {index}"),
                    *x_m,
                    *y_m,
                    *radius_m,
                ));
            }
            CounterfactualEdit::SetBattery { value } => {
                metadata.body.battery_level = *value;
                metadata.body.charging = false;
            }
        }
    }
    Ok(())
}

fn find_counterfactual_object_mut<'a>(
    objects: &'a mut [pete_sim::SimObject],
    kind: CounterfactualObjectKind,
    id: &Option<String>,
) -> Result<&'a mut pete_sim::SimObject> {
    objects
        .iter_mut()
        .find(|object| {
            object_matches_counterfactual_kind(&object.kind, kind)
                && id.as_ref().map(|id| id == &object.id).unwrap_or(true)
        })
        .with_context(|| {
            if let Some(id) = id {
                format!("{} '{id}' not found", object_kind_label(kind))
            } else {
                format!("no {} found", object_kind_label(kind))
            }
        })
}

fn object_matches_counterfactual_kind(
    kind: &pete_sim::SimObjectKind,
    edit_kind: CounterfactualObjectKind,
) -> bool {
    match edit_kind {
        CounterfactualObjectKind::Charger => matches!(kind, pete_sim::SimObjectKind::Charger),
        CounterfactualObjectKind::Person => {
            matches!(kind, pete_sim::SimObjectKind::Person { .. })
        }
        CounterfactualObjectKind::Speaker => {
            matches!(kind, pete_sim::SimObjectKind::SoundSource { .. })
        }
    }
}

fn object_kind_label(kind: CounterfactualObjectKind) -> &'static str {
    match kind {
        CounterfactualObjectKind::Charger => "charger",
        CounterfactualObjectKind::Person => "person",
        CounterfactualObjectKind::Speaker => "speaker",
    }
}

async fn hardware_env(args: HardwareEnvArgs) -> Result<()> {
    let report = collect_hardware_env_report().await;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!("hardware environment");
    println!("  os: {}", report["os"].as_str().unwrap_or("unknown"));
    println!(
        "  architecture: {}",
        report["architecture"].as_str().unwrap_or("unknown")
    );
    println!(
        "  cpu: {}",
        report["cpu_model"].as_str().unwrap_or("unknown")
    );
    println!(
        "  memory: {} kB",
        report["memory_total_kb"].as_u64().unwrap_or(0)
    );
    println!(
        "  raspberry-pi-like: {}",
        report["raspberry_pi_like"].as_bool().unwrap_or(false)
    );
    print_json_list("  create serial candidates", &report["serial_devices"]);
    print_json_list("  gps serial candidates", &report["gps_serial_candidates"]);
    println!(
        "  default gps: {}",
        report["default_gps"]["device"]
            .as_str()
            .unwrap_or("not detected")
    );
    print_json_list(
        "  lidar serial candidates",
        &report["lidar_serial_candidates"],
    );
    println!(
        "  default lidar: {}",
        report["default_lidar"]["device"]
            .as_str()
            .unwrap_or("not detected")
    );
    print_json_list("  i2c devices", &report["i2c_devices"]);
    println!(
        "  default imu: {}",
        report["default_imu"]["device"]
            .as_str()
            .unwrap_or("unknown")
    );
    print_json_list("  cameras", &report["camera_devices"]);
    print_json_list("  audio inputs", &report["audio_input_devices"]);
    println!(
        "  libfreenect/freenect: {}",
        report["kinect"]["freenect_available"]
            .as_bool()
            .unwrap_or(false)
    );
    println!("  data dirs writable:");
    if let Some(object) = report["data_dirs_writable"].as_object() {
        for (path, writable) in object {
            println!("    {path}: {}", writable.as_bool().unwrap_or(false));
        }
    }
    print_json_list("  warnings", &report["warnings"]);
    Ok(())
}

async fn capture_real(args: CaptureRealArgs) -> Result<()> {
    if args.duration_seconds == 0 {
        anyhow::bail!("--duration-seconds must be greater than zero");
    }

    let env_report = collect_hardware_env_report().await;
    let lidar_device = selected_lidar_device(args.lidar.as_deref(), args.mock, &env_report);
    let mut warnings = Vec::new();
    let mut device_availability = serde_json::json!({
        "mock": args.mock,
        "create": null,
        "camera": null,
        "microphone": null,
        "imu": null,
        "gps": null,
        "lidar": null,
        "kinect": env_report["kinect"].clone(),
    });

    let create_port = selected_cockpit_endpoint(
        args.cockpit,
        &args.create_port,
        &args.brainstem_host,
        args.brainstem_local,
        &env_report,
        lidar_device.as_deref(),
    );

    let cockpit: Box<dyn Cockpit + Send> = if args.mock || create_port.as_deref() == Some("mock") {
        device_availability["create"] =
            serde_json::json!({"present": true, "source": "sim-cockpit"});
        Box::new(LocalSimCockpit::new().with_unscoped_bench_mode())
    } else if let Some(create_port) = &create_port {
        let opened: pete_cockpit::Result<Box<dyn Cockpit + Send>> = match args.cockpit {
            CockpitBackendArg::Wifi => Ok(Box::new(HttpCockpit::connect(create_port))),
            CockpitBackendArg::Uart => UartCockpit::connect(create_port)
                .map(|cockpit| Box::new(cockpit) as Box<dyn Cockpit + Send>),
            CockpitBackendArg::Local => create_port
                .parse()
                .map_err(|error| {
                    CockpitError::BadResponse(format!("invalid local address: {error}"))
                })
                .and_then(UdpCockpit::connect)
                .map(|cockpit| Box::new(cockpit) as Box<dyn Cockpit + Send>),
            CockpitBackendArg::Sim => unreachable!("sim resolves to mock"),
        };
        match opened {
            Ok(cockpit) => {
                device_availability["create"] = serde_json::json!({
                    "present": true,
                    "endpoint": create_port,
                    "baud": args.create_baud,
                    "backend": match args.cockpit {
                        CockpitBackendArg::Wifi => "wifi-http-cockpit",
                        CockpitBackendArg::Uart => "uart-cockpit",
                        CockpitBackendArg::Local => "local-rpi5-brainstem",
                        CockpitBackendArg::Sim => "sim-cockpit",
                    }
                });
                cockpit
            }
            Err(error) => {
                anyhow::bail!("failed to open cockpit endpoint {create_port}: {error}");
            }
        }
    } else {
        warnings.push("cockpit UART device not found; using simulated cockpit status".to_string());
        device_availability["create"] =
            serde_json::json!({"present": false, "reason": "no cockpit serial candidate"});
        Box::new(LocalSimCockpit::new().with_unscoped_bench_mode())
    };

    let mut sensors: Vec<Box<dyn SenseProducer + Send>> = Vec::new();
    if args.mock {
        sensors.push(Box::new(MockEyeProducer::default()));
        sensors.push(Box::new(MockEarProducer::default()));
        sensors.push(Box::new(MockRangeProducer::default()));
        sensors.push(Box::new(MockKinectProducer::default()));
        device_availability["camera"] = serde_json::json!({"present": true, "source": "mock"});
        device_availability["microphone"] = serde_json::json!({"present": true, "source": "mock"});
        device_availability["kinect"] = serde_json::json!({"present": true, "source": "mock"});
    } else {
        add_optional_real_sensors(
            &args,
            &env_report,
            create_port.as_deref(),
            lidar_device.as_deref(),
            &mut sensors,
            &mut device_availability,
            &mut warnings,
        );
    }
    let no_real_create = device_availability["create"]["present"].as_bool() != Some(true);
    if !args.mock && no_real_create && sensors.is_empty() {
        anyhow::bail!(
            "no usable devices found: no Create serial device and no requested sensor initialized"
        );
    }

    let requested_frames = duration_to_steps(args.duration_seconds, args.tick_ms);
    if let Some(ledger_path) = &args.ledger {
        let runtime = durable_runtime(JsonlLedger::new(ledger_path), &args.llm)?;
        capture_real_with_runtime(
            args,
            runtime,
            cockpit,
            sensors,
            env_report,
            device_availability,
            warnings,
            requested_frames,
        )
        .await
    } else {
        let runtime = default_noop_runtime(&args.llm)?;
        capture_real_with_runtime(
            args,
            runtime,
            cockpit,
            sensors,
            env_report,
            device_availability,
            warnings,
            requested_frames,
        )
        .await
    }
}

#[allow(clippy::too_many_arguments)]
async fn capture_real_with_runtime<R>(
    args: CaptureRealArgs,
    runtime: R,
    cockpit: Box<dyn Cockpit + Send>,
    sensors: Vec<Box<dyn SenseProducer + Send>>,
    env_report: Value,
    device_availability: Value,
    mut warnings: Vec<String>,
    requested_frames: usize,
) -> Result<()>
where
    R: RuntimeLoop + Send,
{
    let mut runner = RealRobotRunner::new(RobotMode::ReadOnly, cockpit, sensors, runtime)
        .with_frame_processor(real_robot_frame_processor(&mut warnings).await);
    runner.tick_ms = args.tick_ms;
    let firmware_identity = brainstem_firmware_identity(runner.cockpit.client_mut().as_mut());
    let mut writer =
        CaptureWriter::create(&args.out, CaptureSource::RealRobot, Some(args.tick_ms)).await?;
    {
        let manifest = writer.manifest_mut();
        manifest.machine = Some(machine_info_from_env(&env_report));
        manifest.firmware_identity = firmware_identity;
        manifest.command_args = std::env::args().collect();
        manifest.device_availability = device_availability;
        manifest
            .notes
            .push("capture-real is capture-only; motors are not commanded".to_string());
        if args.export_rgb || args.export_depth || args.export_audio {
            manifest.notes.push(
                "asset export enabled; frame asset paths are relative to capture root".to_string(),
            );
        }
    }
    let mut stream_counts = StreamCounts::default();
    let mut events_written = 0usize;
    for _ in 0..requested_frames {
        let tick_result = runner.tick_read_only().await;
        let (snapshot, tick) = match tick_result {
            Ok(values) => values,
            Err(error) if is_transient_readonly_timeout(&error) => {
                eprintln!("read-only capture tick timed out; continuing");
                tokio::time::sleep(Duration::from_millis(args.tick_ms)).await;
                continue;
            }
            Err(error) => return Err(error),
        };
        stream_counts.observe(&snapshot);
        if tick
            .frame
            .notes
            .iter()
            .any(|note| note.contains("ReadOnlyActionSuppressed"))
        {
            events_written = events_written.saturating_add(1);
        }
        writer
            .append_snapshot_with_exported_assets_and_context(
                snapshot.body.last_update_ms,
                snapshot,
                Vec::new(),
                args.export_rgb,
                args.export_depth,
                args.export_audio,
                CaptureExportContext {
                    imu_selection: tick
                        .frame
                        .now
                        .extensions
                        .get("sensor.imu_selection")
                        .cloned(),
                },
            )
            .await?;
        tokio::time::sleep(Duration::from_millis(args.tick_ms)).await;
    }

    let streams = stream_counts.streams();
    warnings.extend(stream_counts.warnings());
    if stream_counts.useful_stream_count() == 0 {
        anyhow::bail!("no usable body or sensor streams were captured");
    }
    {
        let manifest = writer.manifest_mut();
        manifest.streams = streams;
        manifest.warnings = warnings.clone();
        if let Some(ledger) = &args.ledger {
            manifest.notes.push(format!("ledger: {ledger}"));
        }
    }
    let manifest = writer.finish().await?;
    println!(
        "capture-real complete: {} frames, out {}, streams {:?}, warnings {}, motor_applied false",
        manifest.frame_count,
        args.out,
        manifest.streams.present,
        manifest.warnings.len()
    );
    if events_written > 0 {
        println!("  read-only motor suppressions observed: {events_written}");
    }
    if args.export_pointcloud {
        capture_assets(CaptureAssetsArgs {
            capture: args.out.clone(),
            pointcloud: true,
            world_pointcloud: true,
            stride: args.pointcloud_stride,
            max_depth_m: 8.0,
        })
        .await?;
    }
    Ok(())
}

async fn capture_assets(args: CaptureAssetsArgs) -> Result<()> {
    if !args.pointcloud && !args.world_pointcloud {
        anyhow::bail!("no asset conversion requested; pass --pointcloud and/or --world-pointcloud");
    }
    let root = PathBuf::from(&args.capture);
    let reader = CaptureReader::open(&root).await?;
    let mut manifest = reader.manifest().clone();
    let mut frames = reader.read_frames().await?;
    let mut exported = 0usize;
    if args.pointcloud {
        for frame in &mut frames {
            if export_pointcloud_for_frame(&root, frame, args.max_depth_m, args.stride)?.is_some() {
                exported = exported.saturating_add(1);
            }
        }
        rewrite_frames(&root, &frames).await?;
    }
    let world_vertices = if args.world_pointcloud {
        Some(export_world_pointcloud_for_capture(&root, &frames)?)
    } else {
        None
    };
    if exported > 0
        && !manifest
            .warnings
            .iter()
            .any(|warning| warning.contains("uncalibrated point cloud"))
    {
        manifest
            .warnings
            .push("uncalibrated point cloud: using approximate placeholder intrinsics".to_string());
    }
    let world_vertices_count = world_vertices.unwrap_or(0);
    if world_vertices_count > 0 {
        manifest.asset_layout["world_pointcloud"] =
            serde_json::json!("assets/pointcloud/world-accumulated.ply");
        manifest.notes.push(format!(
            "world pointcloud: assets/pointcloud/world-accumulated.ply ({world_vertices_count} voxels)"
        ));
    }
    update_manifest(&root, &manifest).await?;
    println!(
        "capture-assets complete: pointcloud {} frames, world {}, capture {}, stride {}",
        exported, world_vertices_count, args.capture, args.stride
    );
    Ok(())
}

fn export_world_pointcloud_for_capture(
    root: &Path,
    frames: &[pete_worldlab::CaptureFrameRecord],
) -> Result<usize> {
    let mut cloud = VoxelPointCloud::default();
    for frame in frames {
        cloud.observe_snapshot(&frame.snapshot, frame.t_ms);
    }
    let points = cloud.points();
    let rel = Path::new("assets")
        .join("pointcloud")
        .join("world-accumulated.ply");
    write_world_pointcloud_ply(&root.join(rel), &points)?;
    Ok(points.len())
}

fn write_world_pointcloud_ply(path: &Path, points: &[pete_map::VoxelPoint]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut out = String::new();
    out.push_str("ply\nformat ascii 1.0\n");
    out.push_str(&format!("element vertex {}\n", points.len()));
    out.push_str("property float x\nproperty float y\nproperty float z\n");
    out.push_str("property uchar red\nproperty uchar green\nproperty uchar blue\n");
    out.push_str("property float confidence\nproperty uchar stable\nproperty uchar transient\n");
    out.push_str("end_header\n");
    for point in points {
        let [r, g, b] = point.color_rgb.unwrap_or([190, 194, 246]);
        out.push_str(&format!(
            "{:.6} {:.6} {:.6} {} {} {} {:.6} {} {}\n",
            point.position.x_m,
            point.position.y_m,
            point.position.z_m,
            r,
            g,
            b,
            point.confidence,
            u8::from(point.stable),
            u8::from(point.transient)
        ));
    }
    fs::write(path, out)?;
    Ok(())
}

async fn inspect_capture(args: InspectCaptureArgs) -> Result<()> {
    let report = inspect_capture_report(&args.path).await?;
    println!("capture: {}", report.path.display());
    println!("  frames: {}", report.frame_count);
    println!("  duration_ms: {}", report.duration_ms.unwrap_or(0));
    println!(
        "  streams present: {}",
        join_or_none(&report.streams_present)
    );
    println!(
        "  streams missing: {}",
        join_or_none(&report.streams_missing)
    );
    println!(
        "  first/last timestamps: {:?} / {:?}",
        report.first_timestamp_ms, report.last_timestamp_ms
    );
    println!("  events: {}", report.event_count);
    println!("  assets:");
    for (kind, count) in &report.asset_counts {
        println!("    {kind}: {count}");
    }
    for detail in &report.asset_details {
        println!("    {detail}");
    }
    println!("  asset stream health:");
    for stream in &report.asset_streams {
        println!(
            "    {}: count {}, producer {:?}..{:?}, bytes {}, missing_intervals {:?}, unavailable {}, late {}, partial {}, dropped {}, checksum_failures {}",
            stream.kind,
            stream.count,
            stream.first_producer_ms,
            stream.last_producer_ms,
            stream.bytes,
            stream.missing_intervals,
            stream.unavailable,
            stream.late,
            stream.partial,
            stream.dropped,
            stream.checksum_failures,
        );
    }
    println!("  warnings: {}", report.warnings.len());
    for warning in &report.warnings {
        println!("    - {warning}");
    }
    Ok(())
}

struct BackgroundSenseProducer {
    name: &'static str,
    state: std::sync::Arc<std::sync::Mutex<BackgroundSenseState>>,
}

#[derive(Clone, Debug, Default)]
struct BackgroundSenseState {
    latest: Option<SensePacket>,
    pending: VecDeque<SensePacket>,
    last_error: Option<String>,
}

impl BackgroundSenseState {
    fn record_packet(&mut self, name: &str, packet: SensePacket) {
        if name == "kinect-depth" && matches!(packet, SensePacket::EyeFrame(_)) {
            return;
        }
        if is_reliable_background_packet(&packet) {
            self.pending.push_back(packet);
            while self.pending.len() > 32 {
                self.pending.pop_front();
            }
        } else {
            self.latest = Some(packet);
        }
    }

    fn next_packet(&mut self) -> Option<SensePacket> {
        self.pending.pop_front().or_else(|| self.latest.clone())
    }
}

impl BackgroundSenseProducer {
    fn spawn_with_callback<T, F>(
        name: &'static str,
        mut producer: T,
        poll_interval: Duration,
        on_packet: F,
    ) -> Self
    where
        T: SenseProducer + Send + 'static,
        F: Fn(&SensePacket) + Send + 'static,
    {
        let state = std::sync::Arc::new(std::sync::Mutex::new(BackgroundSenseState::default()));
        let worker_state = std::sync::Arc::clone(&state);
        std::thread::spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(error) => {
                    let mut state = worker_state
                        .lock()
                        .expect("background sensor mutex poisoned");
                    state.last_error = Some(format!(
                        "failed to start background sensor runtime: {error}"
                    ));
                    return;
                }
            };
            let mut consecutive_failures = 0_u32;
            loop {
                match runtime.block_on(producer.poll()) {
                    Ok(packet) => {
                        consecutive_failures = 0;
                        on_packet(&packet);
                        let mut state = worker_state
                            .lock()
                            .expect("background sensor mutex poisoned");
                        state.record_packet(name, packet);
                        state.last_error = None;
                    }
                    Err(error) => {
                        consecutive_failures = consecutive_failures.saturating_add(1);
                        let mut state = worker_state
                            .lock()
                            .expect("background sensor mutex poisoned");
                        state.last_error = Some(error.to_string());
                    }
                }
                std::thread::sleep(background_sensor_retry_delay(
                    poll_interval,
                    consecutive_failures,
                ));
            }
        });
        Self { name, state }
    }
}

#[async_trait::async_trait]
impl SenseProducer for BackgroundSenseProducer {
    fn source_name(&self) -> &'static str {
        self.name
    }

    async fn poll(&mut self) -> Result<SensePacket> {
        let mut state = self.state.lock().expect("background sensor mutex poisoned");
        if let Some(packet) = state.next_packet() {
            Ok(packet)
        } else if let Some(error) = state.last_error.clone() {
            anyhow::bail!("{} sensor unavailable: {error}", self.name)
        } else {
            anyhow::bail!("{} sensor has no frame yet", self.name)
        }
    }
}

fn background_sensor_retry_delay(base: Duration, consecutive_failures: u32) -> Duration {
    if consecutive_failures == 0 {
        return base;
    }
    let exponent = consecutive_failures.saturating_sub(1).min(3);
    Duration::from_secs(1_u64 << exponent).min(Duration::from_secs(5))
}

#[cfg(test)]
mod background_sensor_retry_tests {
    use super::{background_sensor_retry_delay, slow_motion_note};
    use pete_sensors::WorldSnapshot;
    use std::time::Duration;

    #[test]
    fn persistent_optional_sensor_failure_backs_off_quickly() {
        let base = Duration::from_millis(33);
        assert_eq!(background_sensor_retry_delay(base, 0), base);
        assert_eq!(
            background_sensor_retry_delay(base, 1),
            Duration::from_secs(1)
        );
        assert_eq!(
            background_sensor_retry_delay(base, 2),
            Duration::from_secs(2)
        );
        assert_eq!(
            background_sensor_retry_delay(base, 3),
            Duration::from_secs(4)
        );
        assert_eq!(
            background_sensor_retry_delay(base, 4),
            Duration::from_secs(5)
        );
        assert_eq!(
            background_sensor_retry_delay(base, 20),
            Duration::from_secs(5)
        );
    }

    #[test]
    fn slow_trace_includes_observed_recovery_progress() {
        let recovery_action = pete_actions::ActionPrimitive::Go {
            intensity: -0.05,
            duration_ms: 500,
        };
        let mut snapshot = WorldSnapshot {
            action_debug: Some(serde_json::json!({
                "conductor_navigation_goal": {
                    "intent": "recover_from_contact",
                    "action": recovery_action,
                    "reason": "escape attempt 2 reversing: 31/120 mm observed odometry"
                }
            })),
            final_selected_action: Some(recovery_action.clone()),
            ..WorldSnapshot::default()
        };

        assert_eq!(
            slow_motion_note(&snapshot),
            ", recovery: escape attempt 2 reversing: 31/120 mm observed odometry"
        );

        snapshot.action_debug = Some(serde_json::json!({
            "why_not_moving": "operator E-stop is latched",
            "conductor_navigation_goal": {
                "intent": "recover_from_contact",
                "action": recovery_action,
                "reason": "should not obscure a motor gate"
            }
        }));
        assert_eq!(
            slow_motion_note(&snapshot),
            ", motion blocked: operator E-stop is latched"
        );
    }
}

fn is_reliable_background_packet(packet: &SensePacket) -> bool {
    matches!(packet, SensePacket::Ear(_))
}

fn publish_live_sensor_only_snapshot(live_state: &LiveViewState, packet: &SensePacket) {
    let now_ms = Utc::now().timestamp_millis().max(0) as u64;
    let mut snapshot = live_state.latest().unwrap_or_default();
    snapshot
        .extensions
        .retain(|extension| extension.name != "live/startup_sensor_only");
    snapshot.extensions.push(ExtensionSense {
        schema_version: 1,
        name: "live/startup_sensor_only".to_string(),
        values: vec![now_ms as f32],
    });

    match packet {
        SensePacket::Kinect(kinect) => {
            snapshot.kinect = kinect.clone();
        }
        SensePacket::EyeFrame(frame) => {
            snapshot.eye_frame = Some(frame.clone());
        }
        SensePacket::Range(range) => {
            snapshot.range = range.clone();
        }
        SensePacket::Imu(imu) => {
            snapshot.imu = imu.clone();
        }
        SensePacket::Gps(gps) => {
            snapshot.gps = Some(gps.clone());
        }
        SensePacket::Ear(ear) => {
            snapshot.ear = ear.clone();
        }
        SensePacket::EarPcm(frame) => {
            snapshot.ear_pcm = Some(frame.clone());
        }
        SensePacket::Eye(eye) => {
            snapshot.eye = eye.clone();
        }
        SensePacket::Face(face) => {
            snapshot.face = face.clone();
        }
        SensePacket::Voice(voice) => {
            snapshot.voice = voice.clone();
        }
        SensePacket::Objects(objects) => {
            snapshot.objects = objects.clone();
        }
        SensePacket::Extension(extension) => {
            snapshot.extensions.push(extension.clone());
        }
    }

    live_state.update(snapshot);
}
