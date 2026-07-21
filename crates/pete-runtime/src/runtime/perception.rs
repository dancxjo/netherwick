fn apply_responses(
    now: &mut Now,
    responses: Vec<Response>,
    sensations: &mut Vec<Sensation>,
    impressions: &mut Vec<Impression>,
    experiences: &mut Vec<Experience>,
    teachings: &mut Vec<pete_llm::LlmTeaching>,
    notes: &mut Vec<String>,
    drive_impulses: &mut DriveSense,
) {
    for response in responses {
        match response {
            Response::Emit(_) => {}
            Response::AddSensation(sensation) => sensations.push(sensation),
            Response::AddImpression(impression) => impressions.push(impression),
            Response::AddExperience(experience) => experiences.push(experience),
            Response::AddDriveImpulse { name, value } => {
                add_drive_impulse(drive_impulses, &name, value)
            }
            Response::SetMemorySense(memory) => now.memory = memory,
            Response::Teach(teaching) => teachings.push(teaching),
            Response::AddMemoryNote(note) => notes.push(note),
        }
    }
}

fn apply_llm_tick(
    llm_tick: &LlmTickResult,
    occurred_at_ms: u64,
    observed_at_ms: u64,
    snapshot_ref: &str,
    sensations: &mut Vec<Sensation>,
    impressions: &mut Vec<Impression>,
    experiences: &mut Vec<Experience>,
    teachings: &mut Vec<pete_llm::LlmTeaching>,
) {
    if let Some(command) = &llm_tick.conscious_command {
        let sensation = Sensation::new(
            "llm.command",
            "llm",
            occurred_at_ms,
            observed_at_ms,
            serde_json::json!({
                "summary": command.summary,
                "action": command.action,
                "input_snapshot_ref": snapshot_ref,
            }),
        )
        .with_summary(command.summary.clone())
        .with_provenance(Provenance::direct().with_stage("llm"));
        let impression = Impression::new(
            "llm.command.observation",
            command.summary.clone(),
            vec![sensation.id],
            sensation.occurred_at_ms,
            sensation.observed_at_ms,
        )
        .with_confidence(llm_tick.sense.confidence);
        let experience = Experience::new(
            "llm.command",
            command.summary.clone(),
            vec![impression.id],
            vec![sensation.id],
            sensation.occurred_at_ms,
            sensation.observed_at_ms,
        );
        sensations.push(sensation);
        impressions.push(impression);
        experiences.push(experience);
    }

    if let Some(critique) = &llm_tick.sense.critique {
        let sensation = Sensation::new(
            "llm.critique",
            "llm",
            occurred_at_ms,
            observed_at_ms,
            serde_json::json!({
                "critique": critique,
                "input_snapshot_ref": snapshot_ref,
            }),
        )
        .with_summary(critique.clone())
        .with_provenance(Provenance::direct().with_stage("llm"));
        let impression = Impression::new(
            "llm.critique.observation",
            critique.clone(),
            vec![sensation.id],
            sensation.occurred_at_ms,
            sensation.observed_at_ms,
        )
        .with_confidence(llm_tick.sense.confidence);
        sensations.push(sensation);
        impressions.push(impression);
    }

    teachings.extend(llm_tick.teaching.clone());
}

fn append_combobulation(
    sensations: &mut Vec<Sensation>,
    impressions: &mut Vec<Impression>,
    experiences: &mut Vec<Experience>,
    occurred_at_ms: u64,
    observed_at_ms: u64,
    snapshot_ref: &str,
    combobulation: &Combobulation,
) {
    let sensation = Sensation::new(
        "llm.combobulation",
        "llm",
        occurred_at_ms,
        observed_at_ms,
        serde_json::json!({
            "summary": combobulation.summary,
            "confidence": combobulation.confidence,
            "input_snapshot_ref": snapshot_ref,
        }),
    )
    .with_summary(combobulation.summary.clone())
    .with_provenance(Provenance::direct().with_stage("combobulator"));
    let impression = Impression::new(
        "llm.combobulation.observation",
        combobulation.summary.clone(),
        vec![sensation.id],
        occurred_at_ms,
        observed_at_ms,
    )
    .with_confidence(combobulation.confidence);
    let experience = Experience::new(
        "llm.combobulation",
        combobulation.summary.clone(),
        vec![impression.id],
        vec![sensation.id],
        occurred_at_ms,
        observed_at_ms,
    );
    sensations.push(sensation);
    impressions.push(impression);
    experiences.push(experience);
}

fn embodied_recall_sensations_and_impressions(
    recall: &RecallBundle,
) -> (Vec<Sensation>, Vec<Impression>) {
    let mut sensations = Vec::new();
    let mut impressions = Vec::new();
    for recollection in &recall.recollections {
        let sensation = recollection.sensation.clone();
        if let Some(impression) = sensation.impression.clone() {
            impressions.push(impression);
        }
        sensations.push(sensation);
    }
    (sensations, impressions)
}

fn derive_direct_impressions_from_now(now: &Now) -> (Vec<Sensation>, Vec<Impression>) {
    let mut sensations = Vec::new();
    let mut impressions = Vec::new();
    let floor_feel = if now.body.flags.cliff_left
        || now.body.flags.cliff_front_left
        || now.body.flags.cliff_front_right
        || now.body.flags.cliff_right
    {
        "the floor feels like it falls away near me"
    } else if now.body.cliff_sensors.max() > 0.0 {
        "the floor feels steady, though my cliff IR signal is uncertain"
    } else {
        "the floor feels steady under me"
    };
    let contact_feel = if now.body.flags.bump_left || now.body.flags.bump_right {
        "my body feels blocked by contact"
    } else if now.body.flags.wall || now.body.flags.virtual_wall {
        "I feel a boundary close to me"
    } else {
        "my body feels unblocked"
    };
    let wheel_feel = if now.body.flags.wheel_drop {
        "one wheel feels unsupported"
    } else {
        "my wheels feel supported"
    };
    let charging_feel = if now.body.charging {
        "charging feels present"
    } else {
        "I do not feel charging contact"
    };
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "body.state",
        "body",
        format!(
            "My body feels {:.0}% full of power; {charging_feel}, {contact_feel}, {wheel_feel}, and {floor_feel}. I feel myself moving forward {:.2} m/s and turning {:.2} rad/s, with my body centered near ({:.2}, {:.2}) and facing {:.2} radians.",
            now.body.battery_level * 100.0,
            now.body.velocity.forward_m_s,
            now.body.velocity.turn_rad_s,
            now.body.odometry.x_m,
            now.body.odometry.y_m,
            now.body.odometry.heading_rad,
        ),
        0.9,
    );
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "eye.state",
        "eye",
        format!(
            "I am seeing through {} frame feature sets, with {} image vectors, {} image-description vectors, and {} scene vectors available.",
            now.eye.frames.len(),
            now.eye.image_vectors.len(),
            now.eye.image_description_vectors.len(),
            now.eye.scene_vectors.len(),
        ),
        0.6,
    );
    let transcript = now
        .ear
        .asr
        .transcript
        .as_deref()
        .or(now.ear.transcript.as_deref());
    if let Some(transcript) = transcript {
        let transcript = transcript.trim();
        if !transcript.is_empty() {
            push_now_input_impression(
                &mut sensations,
                &mut impressions,
                now.t_ms,
                "audio.transcript",
                "ear",
                asr_hearing_impression_text(
                    transcript,
                    now.ear.asr.is_final,
                    now.ear.asr.confidence,
                ),
                now.ear.asr.confidence.max(0.35),
            );
        }
    }
    if let Some(possible) = now
        .ear
        .asr
        .possible_transcript
        .as_deref()
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        push_now_input_impression(
            &mut sensations,
            &mut impressions,
            now.t_ms,
            "audio.possible_speech",
            "ear",
            asr_possible_speech_impression_text(possible, now.ear.asr.confidence),
            now.ear.asr.confidence.max(0.25),
        );
    }
    if let Some(committed) = now
        .ear
        .asr
        .committed_transcript
        .as_deref()
        .or_else(|| now.ear.asr.is_final.then_some(transcript).flatten())
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        push_now_input_impression(
            &mut sensations,
            &mut impressions,
            now.t_ms,
            "audio.committed_speech",
            "ear",
            asr_committed_speech_impression_text(committed, now.ear.asr.confidence),
            now.ear.asr.confidence.max(0.35),
        );
    }
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "ear.state",
        "ear",
        format!(
            "I am hearing through {} audio feature sets; my speech recognition final state is {}, confidence is {:.2}, word count is {:?}, and sequence is {:?}-{:?}.",
            now.ear.features.len(),
            now.ear.asr.is_final,
            now.ear.asr.confidence,
            now.ear.asr.word_count,
            now.ear.asr.sequence_start,
            now.ear.asr.sequence_end,
        ),
        0.6,
    );
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "range.state",
        "range",
        format!(
            "I sense the nearest obstacle at {:?} meters, from {} range beam samples.",
            now.range.nearest_m,
            now.range.beams.len(),
        ),
        0.7,
    );
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "imu.state",
        "imu",
        format!(
            "I feel my orientation through {} values, acceleration through {} values, and angular velocity through {} values.",
            now.imu.orientation.len(),
            now.imu.acceleration.len(),
            now.imu.angular_velocity.len(),
        ),
        0.5,
    );
    if let Some(gps) = &now.gps {
        push_now_input_impression(
            &mut sensations,
            &mut impressions,
            now.t_ms,
            "gps.state",
            "gps",
            format!(
                "I am located near latitude {:.6}, longitude {:.6}, altitude {:?} meters.",
                gps.lat, gps.lon, gps.altitude_m
            ),
            0.6,
        );
    }
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "identity.state",
        "identity",
        format!(
            "I have {} face vectors and {} voice vectors available for recognizing who may be present.",
            now.face.vectors.len(),
            now.voice.vectors.len(),
        ),
        0.5,
    );
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "kinect.state",
        "kinect",
        format!(
            "I sense the room with {} Kinect color feature sets, {} depth samples, {} IR samples, {} skeletons, and audio angle {:?} at confidence {:.2}.",
            now.kinect.color_features.len(),
            now.kinect.depth_m.len(),
            now.kinect.ir.len(),
            now.kinect.skeletons.len(),
            now.kinect.audio_angle_rad,
            now.kinect.audio_confidence,
        ),
        0.5,
    );
    if let Some(surface_graph) = now.extensions.get("surface.scene_graph") {
        push_now_input_impression(
            &mut sensations,
            &mut impressions,
            now.t_ms,
            "surface.scene_graph",
            "surface",
            summarize_surface_scene_graph(surface_graph),
            0.75,
        );
    }
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "memory.state",
        "memory",
        format!(
            "I remember this place with familiarity {:.2}, danger {:.2}, charge value {:.2}, social value {:.2}, novelty {:.2}, {} similar situations, warning {:?}, and graph summary {:?}.",
            now.memory.place_familiarity,
            now.memory.place_danger,
            now.memory.place_charge_value,
            now.memory.place_social_value,
            now.memory.place_novelty,
            now.memory.similar_situation_count,
            now.memory.remembered_warning,
            now.memory.graph_context_summary,
        ),
        0.7,
    );
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "prediction.state",
        "predictions",
        format!(
            "I expect events {:?} with uncertainty {:.2}; my danger model says {:?}, hardcoded danger says {:?}, charge model says {:?}, hardcoded charge says {:?}, and I have {} model action values plus {} hardcoded action values.",
            now.predictions.expected_events,
            now.predictions.uncertainty,
            now.predictions.danger_model,
            now.predictions.danger_hardcoded,
            now.predictions.charge_model,
            now.predictions.charge_hardcoded,
            now.predictions.action_values_model.len(),
            now.predictions.action_values_hardcoded.len(),
        ),
        0.7,
    );
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "surprise.state",
        "surprise",
        format!(
            "I feel surprise at {:.2}, with prediction error {:.2}.",
            now.surprise.total, now.surprise.prediction_error
        ),
        0.7,
    );
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "drive.state",
        "drives",
        format!(
            "I feel battery hunger {:.2}, danger avoidance {:.2}, curiosity {:.2}, social interest {:.2}, fatigue {:.2}, and uncertainty pressure {:.2}.",
            now.drives.battery_hunger,
            now.drives.danger_avoidance,
            now.drives.curiosity,
            now.drives.social_interest,
            now.drives.fatigue,
            now.drives.uncertainty_pressure,
        ),
        0.7,
    );
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "reign.state",
        "reign",
        format!(
            "Remote reign control active {}, mode {:?}, with {} pending commands, age {:?} ms, override pressure {:.2}, and latest command {}.",
            now.reign.active,
            now.reign.mode,
            now.reign.pending_count,
            now.reign.last_command_age_ms,
            now.reign.human_override_pressure,
            now.reign
                .latest
                .as_ref()
                .map(summarize_reign_command_for_runtime)
                .unwrap_or_else(|| "none".to_string()),
        ),
        0.7,
    );
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "self.state",
        "self",
        format!(
            "I am pursuing active goal {:?}, and my mode is {:?}.",
            now.self_sense.active_goal, now.self_sense.mode
        ),
        0.6,
    );
    if !now.extensions.is_empty() {
        push_now_input_impression(
            &mut sensations,
            &mut impressions,
            now.t_ms,
            "extension.state",
            "extensions",
            format!(
                "I have extension context from {}.",
                now.extensions
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            0.5,
        );
    }
    (sensations, impressions)
}

fn summarize_surface_scene_graph(value: &serde_json::Value) -> String {
    let floor_confidence = value
        .get("floor")
        .and_then(|floor| floor.get("confidence"))
        .and_then(|confidence| confidence.as_f64());
    let surfaces = value
        .get("surfaces")
        .and_then(|surfaces| surfaces.as_array())
        .map_or(0, Vec::len);
    let clusters = value
        .get("clusters")
        .and_then(|clusters| clusters.as_array())
        .map_or(0, Vec::len);
    let moving_clusters = value
        .get("clusters")
        .and_then(|clusters| clusters.as_array())
        .map(|clusters| {
            clusters
                .iter()
                .filter(|cluster| {
                    cluster
                        .get("moving")
                        .and_then(|moving| moving.as_bool())
                        .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0);
    let hints = value
        .get("clusters")
        .and_then(|clusters| clusters.as_array())
        .map(|clusters| {
            clusters
                .iter()
                .filter_map(|cluster| cluster.get("semantic_hint").and_then(|hint| hint.as_str()))
                .take(4)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let front_clear_m = value
        .get("navigation")
        .and_then(|navigation| navigation.get("front_clear_m"))
        .and_then(|clearance| clearance.as_f64());
    let left_clear_m = value
        .get("navigation")
        .and_then(|navigation| navigation.get("left_clear_m"))
        .and_then(|clearance| clearance.as_f64());
    let right_clear_m = value
        .get("navigation")
        .and_then(|navigation| navigation.get("right_clear_m"))
        .and_then(|clearance| clearance.as_f64());
    let calibration = summarize_surface_calibration_hint(value.get("calibration_hint"));
    format!(
        "I perceive persistent geometry: floor confidence {}, {} stable surfaces, {} leftover clusters ({} moving; hints: {}), navigation clearance front {}, left {}, right {}, and calibration {}.",
        format_optional_magnitude(floor_confidence, ""),
        surfaces,
        clusters,
        moving_clusters,
        if hints.is_empty() {
            "none".to_string()
        } else {
            hints.join(", ")
        },
        format_optional_magnitude(front_clear_m, "m"),
        format_optional_magnitude(left_clear_m, "m"),
        format_optional_magnitude(right_clear_m, "m"),
        calibration
    )
}

fn summarize_surface_calibration_hint(value: Option<&serde_json::Value>) -> String {
    let Some(value) = value else {
        return "unknown".to_string();
    };
    let height_error = value
        .get("floor_height_error_m")
        .and_then(|value| value.as_f64());
    let tilt_deg = value
        .get("floor_tilt_rad")
        .and_then(|value| value.as_f64())
        .map(|value| value.to_degrees());
    match (height_error, tilt_deg) {
        (Some(height), Some(tilt)) => {
            format!("floor offset {height:.2}m and tilt {tilt:.1} degrees")
        }
        (Some(height), None) => format!("floor offset {height:.2}m"),
        (None, Some(tilt)) => format!("floor tilt {tilt:.1} degrees"),
        (None, None) => "unknown".to_string(),
    }
}

fn format_optional_magnitude(value: Option<f64>, unit: &str) -> String {
    value
        .map(|value| format!("{value:.2}{unit}"))
        .unwrap_or_else(|| "unknown".to_string())
}

fn push_now_input_impression(
    sensations: &mut Vec<Sensation>,
    impressions: &mut Vec<Impression>,
    t_ms: u64,
    kind: &str,
    source: &str,
    text: String,
    confidence: f32,
) {
    let text = ensure_natural_confidence_text(&text, confidence);
    let sensation = Sensation::new(
        kind,
        source,
        t_ms,
        t_ms,
        serde_json::json!({ "text": text }),
    )
    .with_summary(text.clone())
    .with_provenance(Provenance::direct().with_stage("now"));
    let impression = Impression::new(
        format!("{kind}.impression"),
        text,
        vec![sensation.id],
        t_ms,
        t_ms,
    )
    .with_confidence(confidence)
    .with_payload(serde_json::json!({
        "generator": "mechanical",
        "faculty": format!("{source}.mechanical_impression"),
        "source_experience_kind": kind,
        "source": source,
    }));
    sensations.push(sensation);
    impressions.push(impression);
}

fn asr_hearing_impression_text(transcript: &str, is_final: bool, confidence: f32) -> String {
    let transcript = transcript.trim();
    let confidence = confidence.clamp(0.0, 1.0);
    if is_final {
        if confidence >= 0.85 {
            format!("I'm confident I finally heard \"{transcript}\".")
        } else if confidence >= 0.60 {
            format!("I'm pretty sure I finally heard \"{transcript}\".")
        } else {
            format!("I think I finally heard \"{transcript}\".")
        }
    } else if confidence >= 0.85 {
        format!("I'm pretty sure I'm hearing \"{transcript}\".")
    } else if confidence >= 0.45 {
        format!("I think I heard \"{transcript}\".")
    } else {
        format!("I may have heard \"{transcript}\".")
    }
}

fn asr_possible_speech_impression_text(transcript: &str, confidence: f32) -> String {
    let transcript = transcript.trim();
    if confidence >= 0.75 {
        format!("I'm probably hearing the possible speech \"{transcript}\".")
    } else if confidence >= 0.45 {
        format!("I think the possible speech is \"{transcript}\".")
    } else {
        format!("I may be hearing possible speech like \"{transcript}\".")
    }
}

fn asr_committed_speech_impression_text(transcript: &str, confidence: f32) -> String {
    let transcript = transcript.trim();
    if confidence >= 0.85 {
        format!("I'm confident I can commit the heard speech as \"{transcript}\".")
    } else if confidence >= 0.60 {
        format!("I'm pretty sure I can commit the heard speech as \"{transcript}\".")
    } else {
        format!("I think I can commit the heard speech as \"{transcript}\".")
    }
}

fn ensure_natural_confidence_text(text: &str, confidence: f32) -> String {
    if starts_with_natural_confidence(text) {
        return text.to_string();
    }

    let claim = lower_first_char(text.trim());
    match confidence.clamp(0.0, 1.0) {
        value if value >= 0.85 => format!("I'm confident that {claim}"),
        value if value >= 0.65 => format!("I'm pretty sure that {claim}"),
        value if value >= 0.40 => format!("I think {claim}"),
        _ => format!("I'm not sure, but I think {claim}"),
    }
}

fn starts_with_natural_confidence(text: &str) -> bool {
    let text = text.trim();
    text.starts_with("I'm confident")
        || text.starts_with("I'm pretty sure")
        || text.starts_with("I think")
        || text.starts_with("I may have")
        || text.starts_with("I'm not sure")
}

fn lower_first_char(text: &str) -> String {
    let mut chars = text.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    first.to_lowercase().chain(chars).collect()
}

fn summarize_reign_command_for_runtime(input: &pete_actions::ReignInput) -> String {
    match &input.command {
        pete_actions::ReignCommand::Stop => "Stop".to_string(),
        pete_actions::ReignCommand::Go {
            intensity,
            duration_ms,
        } => format!("Go intensity {:.2} for {}ms", intensity, duration_ms),
        pete_actions::ReignCommand::Reverse {
            intensity,
            duration_ms,
        } => format!("Reverse intensity {:.2} for {}ms", intensity, duration_ms),
        pete_actions::ReignCommand::Drive {
            forward,
            turn,
            duration_ms,
        } => format!(
            "Drive forward {:.2}, turn {:.2} for {}ms",
            forward, turn, duration_ms
        ),
        pete_actions::ReignCommand::Turn {
            direction,
            intensity,
            duration_ms,
        } => format!(
            "Turn {:?} intensity {:.2} for {}ms",
            direction, intensity, duration_ms
        ),
        pete_actions::ReignCommand::Inspect { target } => format!("Inspect {:?}", target),
        pete_actions::ReignCommand::Approach { target } => format!("Approach {:?}", target),
        pete_actions::ReignCommand::Dock => "Dock".to_string(),
        pete_actions::ReignCommand::Explore { duration_ms } => {
            format!("Explore for {}ms", duration_ms)
        }
        pete_actions::ReignCommand::Speak { text } => format!("Speak {text}"),
        pete_actions::ReignCommand::Chirp { pattern } => format!("Chirp {:?}", pattern),
        pete_actions::ReignCommand::SetMode { mode } => format!("Set mode {:?}", mode),
    }
}

fn derive_direct_experiences(
    impressions: &[Impression],
    sensations: &[Sensation],
    t_ms: u64,
) -> Vec<Experience> {
    if impressions.is_empty() || sensations.is_empty() {
        return Vec::new();
    }
    vec![Experience::new(
        "realtime.situation",
        impressions
            .iter()
            .map(|value| value.text.as_str())
            .collect::<Vec<_>>()
            .join(" "),
        impressions.iter().map(|value| value.id).collect(),
        sensations.iter().map(|value| value.id).collect(),
        t_ms,
        t_ms,
    )]
}

fn add_drive_impulse(drives: &mut DriveSense, name: &DriveName, value: f32) {
    match name {
        DriveName::BatteryHunger => {
            drives.battery_hunger = (drives.battery_hunger + value).clamp(0.0, 1.0)
        }
        DriveName::DangerAvoidance => {
            drives.danger_avoidance = (drives.danger_avoidance + value).clamp(0.0, 1.0)
        }
        DriveName::Curiosity => drives.curiosity = (drives.curiosity + value).clamp(0.0, 1.0),
        DriveName::SocialInterest => {
            drives.social_interest = (drives.social_interest + value).clamp(0.0, 1.0)
        }
        DriveName::Fatigue => drives.fatigue = (drives.fatigue + value).clamp(0.0, 1.0),
        DriveName::UncertaintyPressure => {
            drives.uncertainty_pressure = (drives.uncertainty_pressure + value).clamp(0.0, 1.0)
        }
    }
}

fn describe_safety_reason(reason: Option<SafetyReason>) -> &'static str {
    match reason {
        Some(SafetyReason::Charging) => "charging",
        Some(SafetyReason::WheelDrop) => "wheel drop",
        Some(SafetyReason::Cliff) => "cliff",
        Some(SafetyReason::BatteryCritical) => "critical battery",
        Some(SafetyReason::StaleSensors) => "stale sensors",
        Some(SafetyReason::LostBodyComms) => "lost body comms",
        Some(SafetyReason::MotorOutOfRange) => "motor out of range",
        Some(SafetyReason::HighDanger) => "high danger",
        Some(SafetyReason::RawLlmMotorRejected) => "raw llm motor rejected",
        Some(SafetyReason::ReadOnlyMode) => "read-only mode",
        Some(SafetyReason::Contact) => "contact",
        None => "unknown reason",
    }
}

