fn danger_prediction(output: DangerOutput) -> DangerPrediction {
    DangerPrediction {
        bump_risk: output.bump_risk,
        cliff_risk: output.cliff_risk,
        wheel_drop_risk: output.wheel_drop_risk,
        stuck_risk: output.stuck_risk,
        confidence: output.confidence,
    }
}

fn charge_prediction(output: ChargeOutput) -> ChargePrediction {
    ChargePrediction {
        charge_probability: output.charge_probability,
        expected_battery_delta: output.expected_battery_delta,
        dock_likelihood: output.dock_likelihood,
        confidence: output.confidence,
    }
}

fn stuck_phase_label(code: f64) -> &'static str {
    match code.round() as i32 {
        1 => "stop",
        2 => "reverse",
        3 => "turn-away",
        _ => "none",
    }
}

fn action_value_prediction(
    action: ActionPrimitive,
    output: ActionValueOutput,
) -> ActionValuePrediction {
    ActionValuePrediction {
        action,
        value: output.value,
        confidence: output.confidence,
    }
}

fn eye_prediction(output: &EyeNextOutput) -> EyePrediction {
    EyePrediction {
        width: output.width,
        height: output.height,
        rgb: output.rgb.clone(),
        confidence: output.confidence,
    }
}

fn ear_prediction(output: &EarNextOutput) -> EarPrediction {
    EarPrediction {
        sample_rate_hz: output.sample_rate_hz,
        channels: output.channels,
        pcm: output.pcm.clone(),
        features: output.features.clone(),
        confidence: output.confidence,
    }
}

fn robot_initialized_event_input(now: &Now) -> Option<RobotInitializedEventInput> {
    let init = now.extensions.get("robot.initialization")?;
    Some(RobotInitializedEventInput {
        t_ms: now.t_ms,
        mode: init
            .get("mode")
            .and_then(|value| value.as_str())
            .unwrap_or("unknown")
            .to_string(),
        body: init
            .get("body")
            .and_then(|value| value.as_str())
            .unwrap_or("unknown body")
            .to_string(),
        battery_percent: init
            .get("battery_percent")
            .and_then(|value| value.as_u64())
            .and_then(|value| u32::try_from(value).ok()),
        charging: init.get("charging").and_then(|value| value.as_bool()),
        active_sensors: init
            .get("active_sensors")
            .and_then(|value| value.as_u64())
            .unwrap_or(0) as usize,
        requested_sensors: init
            .get("requested_sensors")
            .and_then(|value| value.as_u64())
            .unwrap_or(0) as usize,
        ledger: init
            .get("ledger")
            .and_then(|value| value.as_str())
            .unwrap_or("unknown ledger")
            .to_string(),
        tick_ms: init
            .get("tick_ms")
            .and_then(|value| value.as_u64())
            .unwrap_or(0),
        dashboard: init
            .get("dashboard")
            .and_then(|value| value.as_str())
            .map(str::to_string),
        capture: init
            .get("capture")
            .and_then(|value| value.as_str())
            .map(str::to_string),
    })
}

fn safety_trace_script_actions<S>(
    safety: &mut S,
    now: &Now,
    output: &EventScriptOutput,
) -> SafeScriptSequence
where
    S: SafetyLayer,
{
    SafeScriptSequence {
        actions: output
            .actions
            .iter()
            .map(|requested| {
                let action = script_action_to_primitive(requested);
                let desired_motor = action_to_motor_command(action.as_ref());
                let decision = safety.filter(now, desired_motor);
                SafeScriptAction {
                    requested: requested.clone(),
                    action,
                    desired_motor,
                    final_motor: decision.command,
                    vetoed: decision.vetoed,
                    safety_reason: decision
                        .reason
                        .map(|reason| describe_safety_reason(Some(reason)).to_string()),
                }
            })
            .collect(),
    }
}

fn first_motor_script_action(output: &EventScriptOutput) -> Option<ActionPrimitive> {
    output
        .actions
        .iter()
        .filter_map(script_action_to_primitive)
        .find(|action| {
            !matches!(
                action,
                ActionPrimitive::Speak { .. } | ActionPrimitive::Chirp { .. }
            )
        })
}

fn script_action_to_primitive(action: &EventScriptAction) -> Option<ActionPrimitive> {
    match action {
        EventScriptAction::Say { text } => Some(ActionPrimitive::Speak { text: text.clone() }),
        EventScriptAction::Chirp { pattern } => Some(ActionPrimitive::Chirp {
            pattern: pattern.clone(),
        }),
        EventScriptAction::Song { .. } => None,
        EventScriptAction::Stop => Some(ActionPrimitive::Stop),
        EventScriptAction::Rotate { deg } => Some(ActionPrimitive::Turn {
            direction: if *deg >= 0 {
                TurnDir::Left
            } else {
                TurnDir::Right
            },
            intensity: 0.5,
            duration_ms: ((*deg as i32).unsigned_abs() as u64 * 10).max(500),
        }),
        EventScriptAction::Go => Some(ActionPrimitive::Go {
            intensity: 0.15,
            duration_ms: 500,
        }),
    }
}

fn append_event_script_chirp(
    now: &mut Now,
    notes: &mut Vec<String>,
    event_name: &str,
    pattern: ChirpPattern,
) -> Result<()> {
    let sequence = SafeScriptSequence {
        actions: vec![SafeScriptAction {
            requested: EventScriptAction::Chirp {
                pattern: pattern.clone(),
            },
            action: Some(ActionPrimitive::Chirp {
                pattern: pattern.clone(),
            }),
            desired_motor: MotorCommand::stop(),
            final_motor: MotorCommand::stop(),
            vetoed: false,
            safety_reason: None,
        }],
    };
    let event_scripts = now
        .extensions
        .entry("event_scripts".to_string())
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    if !event_scripts.is_object() {
        *event_scripts = serde_json::Value::Object(serde_json::Map::new());
    }
    if let Some(object) = event_scripts.as_object_mut() {
        object.insert(event_name.to_string(), serde_json::to_value(sequence)?);
    }
    notes.push(format!(
        "EventScript:on({event_name}) emitted chirp({pattern:?})"
    ));
    Ok(())
}

fn charger_visible(now: &Now) -> bool {
    now.objects.observations.iter().any(|observation| {
        observation.class == ObjectClass::Charger && observation.confidence >= 0.4
    }) || charger_signal_scores(now).1 >= 0.5
}

fn execute_event_script_typescript<I>(script: &str, input: &I) -> Result<EventScriptOutput>
where
    I: Serialize,
{
    let input_json =
        serde_json::to_string(input).context("failed to serialize event script input")?;
    let random_value = rand::random::<f64>();
    let source = format!(
        r#"
const input = {input_json};
const __peteRandom = {random_value};
function random() {{
  return __peteRandom;
}}
function say(text) {{
  return {{ type: "say", text: String(text) }};
}}
function chirp(pattern) {{
  return {{ type: "chirp", pattern: String(pattern) }};
}}
function song(name) {{
  return {{ type: "song", name: String(name) }};
}}
function stop() {{
  return {{ type: "stop" }};
}}
function rotate(deg) {{
  return {{ type: "rotate", deg: Number(deg) }};
}}
function go() {{
  return {{ type: "go" }};
}}
{script}
"#
    );
    let mut interp = Interpreter::new();
    interp
        .prepare(
            &source,
            Some(tsrun::ModulePath::new("/pete-event-script.ts")),
        )
        .map_err(tsrun_error)?;
    let value = loop {
        match interp.step().map_err(tsrun_error)? {
            StepResult::Continue => continue,
            StepResult::Complete(value) => break value,
            StepResult::NeedImports(imports) => {
                let names = imports
                    .iter()
                    .map(|request| request.specifier.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                anyhow::bail!("unsupported TypeScript import(s): {names}");
            }
            StepResult::Suspended { .. } => {
                anyhow::bail!(
                    "TypeScript event script suspended; async host commands are not enabled"
                )
            }
            StepResult::Done => return Ok(EventScriptOutput::default()),
        }
    };
    let value = js_value_to_json(value.value()).map_err(tsrun_error)?;
    if value.get("actions").is_some() {
        serde_json::from_value(value).context("failed to parse TypeScript event script output")
    } else {
        let actions = serde_json::from_value(value)
            .context("failed to parse TypeScript event script action list")?;
        Ok(EventScriptOutput { actions })
    }
}

fn tsrun_error(err: JsError) -> anyhow::Error {
    anyhow::anyhow!("TypeScript event script failed: {err}")
}
