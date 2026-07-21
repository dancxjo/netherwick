fn build_combobulator_prompt(
    now: &Now,
    impressions: &[Impression],
    embodied: Option<&EmbodiedContext>,
    z: &ExperienceLatent,
    futures: &[FuturePrediction],
    recall_summary: &str,
) -> String {
    let timeline = render_combobulator_timeline(impressions);
    let embodied = render_embodied_context(embodied);
    let futures = summarize_futures(futures);
    let self_context = render_self_model_context(now);
    let uuid_options = render_prompt_uuid_options();
    format!(
        "You are the combobulator for an embodied robot.\n\
Given recent impressions and predicted futures in timeline order, distill what appears to be happening right now.\n\
You run continuously over the recent timeline; each pass tries to understand what is going on right now. Write from first-person lived experience from the robot's point of view, using I/my/me naturally.\n\
This summary will be used as a basic understanding of the current situation for a system that may need to act immediately. Think of it as telling someone with amnesia as quickly as possible, but as thoroughly as needed for them to act reasonably.\n\
Use only the evidence below. The impressions are first-person present-tense embodied claims such as \"I see...\", \"I hear...\", or \"My body...\"; preserve that lived point of view. Prefer concrete body facts, nearby people or speech, visible scene details, memory, safety, and immediate context. Explain what appears to be happening right now, not a redundant list of events.\n\
{SENSOR_GROUNDING_RULES}\n\
{COMBOBULATOR_DISTILLATION_RULES}\n\
{LIVE_EVENT_RULES}\n\
Return JSON only with this schema:\n\
{{\"summary\":\"...\",\"confidence\":0.0}}\n\n\
If any output field calls for a new UUID or id, choose one of these exact UUID options and do not invent your own:\n\
{}\n\n\
CONTEXT FRAME\n\
WHO\n\
- embodied robot\n\
WHAT\n\
- current awareness synthesis from impressions\n\
WHERE\n\
- current body location if sensors or memory reveal it; otherwise unknown\n\
WHEN\n\
- now at {} ms\n\
WHY\n\
- produce a compact awareness statement useful to the next action decision\n\
HOW\n\
- distill text impressions produced from body, hearing, vision, range, memory, predictions, surprise, and remote controls\n\n\
Latent confidence: {:.2}\n\
Latent prediction error: {:.2}\n\
Recall summary: {}\n\
Current embodied experience:\n{}\n\
Canonical self-model (capabilities and authority):\n{}\n\
Timeline evidence:\n{}\n\
Predicted futures:\n{}\n",
        uuid_options,
        now.t_ms,
        z.confidence,
        z.prediction_error,
        recall_summary,
        embodied,
        self_context,
        timeline,
        futures
    )
}

fn build_agent_prompt(
    now: &Now,
    embodied: Option<&EmbodiedContext>,
    z: &ExperienceLatent,
    futures: &[FuturePrediction],
    recall_summary: &str,
    awareness_summary: Option<&str>,
    config: &LlmConfig,
) -> String {
    let senses = summarized_senses(now)
        .into_iter()
        .map(|line| format!("- {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    let embodied = render_embodied_context(embodied);
    let futures = summarize_futures(futures);
    let self_context = render_self_model_context(now);
    let uuid_options = render_prompt_uuid_options();
    format!(
        "You are the conscious LLM layer for an embodied robot.\n\
When commands are enabled, choose a high-level action primitive whenever movement, speech, inspection, docking, or stopping is appropriate.\n\
You are in autonomous discovery mode: safely explore, inspect uncertain or interesting stimuli, and prefer active information-gathering when there is no higher-priority goal or danger.\n\
The action field is an executable command candidate for the robot body, not only a suggestion or note.\n\
Never output raw motor control such as wheel speeds, PWM values, serial bytes, or velocity arrays.\n\
Never claim a self-capability absent from the canonical self-model; preserve explicit uncertainty and unavailable reasons.\n\
Treat Reign controls as present-tense command input. If a Reign command is active and safe, set action to the matching allowed action; if you choose something else, explain why in critique.\n\
{LIVE_EVENT_RULES}\n\
Allowed action kinds: stop, go, turn, inspect, approach, dock, explore, speak, chirp.\n\
Allowed chirp patterns and meanings:\n{}\n\
If commands are disabled, leave action null. Commands enabled: {}. Teaching enabled: {}.\n\
Return JSON only with this schema:\n\
{{\n\
  \"summary\":\"short first-person command or reflection\",\n\
  \"critique\":\"optional critique\",\n\
  \"confidence\":0.0,\n\
  \"action\":{{\"kind\":\"dock\"}} or null,\n\
  \"counterfactuals\":[{{\"instead_of\":null,\"proposed\":{{\"kind\":\"turn\",\"direction\":\"left\",\"intensity\":0.4,\"duration_ms\":1000}},\"reason\":\"...\",\"weight\":0.5}}],\n\
  \"memory_notes\":[\"...\"]\n\
}}\n\n\
If any output field calls for a new UUID or id, choose one of these exact UUID options and do not invent your own:\n\
{}\n\n\
Current time: {} ms\n\
Awareness summary: {}\n\
Current embodied experience:\n{}\n\
Canonical self-model (the sole source for what you can currently do and who controls action):\n{}\n\
Recall summary: {}\n\
Battery: {:.2}\n\
Surprise: {:.2}\n\
Latent confidence: {:.2}\n\
Predicted futures:\n{}\n\
Summarized senses:\n{}\n",
        CHIRP_PATTERN_PROMPT,
        config.allow_commands,
        config.allow_teaching,
        uuid_options,
        now.t_ms,
        awareness_summary.unwrap_or("none"),
        embodied,
        self_context,
        recall_summary,
        now.body.battery_level,
        now.surprise.total,
        z.confidence,
        futures,
        senses
    )
}

fn render_self_model_context(now: &Now) -> String {
    let self_model = &now.world.self_model;
    let available = self_model
        .capabilities
        .capabilities
        .values()
        .filter(|capability| {
            matches!(
                capability.availability,
                pete_now::CapabilityAvailability::Available
                    | pete_now::CapabilityAvailability::Degraded
            ) && capability.authorized
        })
        .map(|capability| capability.id.0.as_str())
        .collect::<Vec<_>>();
    let unavailable = self_model
        .capabilities
        .capabilities
        .values()
        .filter(|capability| {
            matches!(
                capability.availability,
                pete_now::CapabilityAvailability::Unavailable
                    | pete_now::CapabilityAvailability::Unknown
            ) || !capability.authorized
        })
        .map(|capability| {
            format!(
                "{} ({})",
                capability.id.0,
                capability
                    .unavailable_reason
                    .as_deref()
                    .or(capability.authority_reason.as_deref())
                    .unwrap_or("availability is unknown")
            )
        })
        .collect::<Vec<_>>();
    format!(
        "organism_id={} body_id={} controller={:?} possessed={} armed={} active_goal={} active_behavior={} available=[{}] unavailable=[{}]",
        self_model.organism_id.value.0,
        self_model.body.body_id.value.0,
        self_model.agency.controller,
        self_model.agency.possessed.value,
        self_model.agency.armed.value,
        self_model
            .motivation
            .selected_goal
            .as_deref()
            .unwrap_or("none"),
        self_model
            .active_control
            .behavior_id
            .as_deref()
            .unwrap_or("none"),
        available.join(", "),
        unavailable.join(", ")
    )
}

pub fn build_scientific_review_prompt(request: &LlmReviewRequest) -> String {
    let available_actions = request
        .available_actions
        .iter()
        .filter_map(|action| serde_json::to_string(&action_spec_json(action)).ok())
        .collect::<Vec<_>>();
    let training_examples = request
        .training_examples
        .iter()
        .map(|example| {
            serde_json::json!({
                "example_id": example.example_id,
                "behavior": example.behavior,
                "input_summary": example.input_summary,
                "expected_summary": example.expected_summary,
                "actual_summary": example.actual_summary,
                "reward": example.reward,
                "source": example.source,
                "contradictions": example.contradictions,
                "missing_evidence": example.missing_evidence,
            })
        })
        .collect::<Vec<_>>();
    format!(
        "You are Pete's scientific critic, not its source of truth.\n\
Inspect the target below and produce skeptical, evidence-grounded review JSON only.\n\
You may identify weak evidence, possible fused clusters, suspicious training rows, plausible labels, counterfactual actions, and tests that would reduce uncertainty.\n\
You must not declare identity as certain, override safety, merge entities, accept bindings, invent sensor evidence, pretend movement happened, or mark a training row as true.\n\
Motion actions are suggestions only; downstream safety and admission systems decide what can happen.\n\
Be compact, explicit about uncertainty, and prefer \"plausible but unproven\" language when evidence is incomplete.\n\
Return JSON only with this schema:\n\
{{\n\
  \"id\":\"optional review id\",\n\
  \"critique\":\"optional critique grounded in evidence\",\n\
  \"counterfactuals\":[{{\"instead_of\":null,\"proposed\":{{\"kind\":\"inspect\",\"target\":\"novelty\"}},\"reason\":\"...\",\"weight\":0.5}}],\n\
  \"suggested_tests\":[{{\"action\":{{\"kind\":\"inspect\",\"target\":\"novelty\"}},\"question\":\"...\",\"expected_observation\":\"...\",\"disconfirming_observation\":\"...\",\"risk_note\":null,\"confidence\":0.5}}],\n\
  \"suspicious_training_examples\":[{{\"example_id\":\"...\",\"reason\":\"...\",\"severity\":0.5,\"suspected_issue\":\"unsupported_label\",\"supporting_evidence\":[\"...\"],\"missing_evidence\":[\"...\"],\"suggested_fix\":\"human review\"}}],\n\
  \"label_proposals\":[{{\"example_id\":\"...\",\"proposed_label\":\"...\",\"rationale\":\"...\",\"confidence\":0.5,\"requires_human_review\":true}}],\n\
  \"human_review_prompts\":[\"...\"],\n\
  \"confidence\":0.0\n\
}}\n\n\
REVIEW TARGET\n\
- target_id: {}\n\
- target_kind: {:?}\n\
- review_time_ms: {}\n\
- candidate_explanation: {}\n\
- current_confidence: {:.2}\n\
- safety_state: {}\n\n\
OBSERVED EVIDENCE\n{}\n\n\
KNOWN CONTRADICTIONS\n{}\n\n\
MISSING EVIDENCE\n{}\n\n\
AVAILABLE ACTIONS JSON\n{}\n\n\
TRAINING EXAMPLES JSON\n{}\n",
        prompt_json_string(&request.target_id),
        request.target_kind,
        request.t_ms,
        prompt_json_string(request.candidate_explanation.as_deref().unwrap_or("none")),
        request.current_confidence.clamp(0.0, 1.0),
        prompt_json_string(request.safety_state.as_deref().unwrap_or("unknown")),
        prompt_lines(&request.observed_evidence),
        prompt_lines(&request.known_contradictions),
        prompt_lines(&request.missing_evidence),
        if available_actions.is_empty() {
            "- none".to_string()
        } else {
            available_actions
                .into_iter()
                .map(|line| format!("- {line}"))
                .collect::<Vec<_>>()
                .join("\n")
        },
        serde_json::to_string_pretty(&training_examples)
            .unwrap_or_else(|_| "[]".to_string())
    )
}

fn prompt_lines(lines: &[String]) -> String {
    if lines.is_empty() {
        return "- none".to_string();
    }
    lines
        .iter()
        .map(|line| format!("- {}", prompt_json_string(line)))
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_prompt_uuid_options() -> String {
    (0..PROMPT_UUID_OPTION_COUNT)
        .map(|_| format!("- {}", Uuid::new_v4()))
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_embodied_context(context: Option<&EmbodiedContext>) -> String {
    let Some(context) = context else {
        return "- unavailable".to_string();
    };

    let mut lines = Vec::new();
    if let Some(id) = context.experience_id {
        lines.push(format!("- experience_id: {id}"));
    }
    if !context.summary.trim().is_empty() {
        lines.push(format!(
            "- summary: {}",
            compact_line(&context.summary, 240)
        ));
    }
    lines.push(format!(
        "- counts: sensations={}, derived_sensations={}, impressions={}, lineage_edges={}",
        context.sensations.len(),
        context.derived_sensation_count(),
        context.impressions.len(),
        context.lineage.len()
    ));
    for sensation in context.sensations.iter().take(8) {
        let parent = sensation
            .parent_id
            .map(|id| format!(" parent={id}"))
            .unwrap_or_default();
        let summary = sensation
            .summary
            .as_deref()
            .map(|text| format!(" summary=\"{}\"", compact_line(text, 120)))
            .unwrap_or_default();
        lines.push(format!(
            "- sensation {}: modality={} payload={} kind={}{}{}",
            sensation.id,
            sensation.modality.as_str(),
            sensation.payload_kind.as_str(),
            sensation.kind,
            parent,
            summary
        ));
    }
    for impression in context.impressions.iter().rev().take(6).rev() {
        let target = impression
            .sensation_id
            .map(|id| format!("sensation={id}"))
            .or_else(|| {
                impression
                    .experience_id
                    .map(|id| format!("experience={id}"))
            })
            .unwrap_or_else(|| "target=unknown".to_string());
        lines.push(format!(
            "- impression {}: {} \"{}\"",
            impression.id,
            target,
            compact_line(&impression.text, 160)
        ));
    }
    for edge in context.lineage.iter().take(8) {
        lines.push(format!(
            "- lineage: {} -> {}",
            edge.parent_id, edge.child_id
        ));
    }
    for vector in context.sensation_vectors.iter().take(6) {
        lines.push(format!(
            "- sensation_vector: sensation={} model={} dim={} modality={} payload={}",
            vector.source_sensation_id,
            vector.model_id,
            vector.dim,
            vector.modality.as_str(),
            vector.payload_kind.as_str()
        ));
    }
    for prediction in context.predictions.iter().take(4) {
        let vector = prediction
            .vector
            .as_ref()
            .map(|vector| {
                format!(
                    " vector_model={} vector_dim={}",
                    vector.model_id, vector.dim
                )
            })
            .unwrap_or_default();
        lines.push(format!(
            "- prediction +{}ms confidence={:.2}{}: {}",
            prediction.offset_ms,
            prediction.confidence,
            vector,
            compact_line(&prediction.text, 140)
        ));
    }
    for link in context.memory_links.iter().take(4) {
        let text = link
            .text
            .as_deref()
            .map(|text| format!(" \"{}\"", compact_line(text, 120)))
            .unwrap_or_default();
        lines.push(format!(
            "- memory_link: target={} relation={} score={:.2}{}",
            link.target_id, link.relation, link.score, text
        ));
    }
    lines.join("\n")
}

fn compact_line(text: &str, max_chars: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.chars().count() <= max_chars {
        return compact;
    }
    let mut out = compact
        .chars()
        .take(max_chars.saturating_sub(3))
        .collect::<String>();
    out.push_str("...");
    out
}

fn summarize_reign_command(input: &pete_actions::ReignInput) -> String {
    match &input.command {
        pete_actions::ReignCommand::Stop => "Stop".to_string(),
        pete_actions::ReignCommand::Go {
            intensity,
            duration_ms,
        } => format!("Go, intensity {:.2}, {}ms", intensity, duration_ms),
        pete_actions::ReignCommand::Reverse {
            intensity,
            duration_ms,
        } => format!("Reverse, intensity {:.2}, {}ms", intensity, duration_ms),
        pete_actions::ReignCommand::Drive {
            forward,
            turn,
            duration_ms,
        } => format!(
            "Drive, forward {:.2}, turn {:.2}, {}ms",
            forward, turn, duration_ms
        ),
        pete_actions::ReignCommand::Turn {
            direction,
            intensity,
            duration_ms,
        } => format!(
            "Turn {:?}, intensity {:.2}, {}ms",
            direction, intensity, duration_ms
        ),
        pete_actions::ReignCommand::Inspect { target } => {
            format!("Inspect {:?}", target)
        }
        pete_actions::ReignCommand::Approach { target } => {
            format!("Approach {:?}", target)
        }
        pete_actions::ReignCommand::Dock => "Dock".to_string(),
        pete_actions::ReignCommand::Explore { duration_ms } => {
            format!("Explore for {}ms", duration_ms)
        }
        pete_actions::ReignCommand::Speak { text } => {
            format!("Speak {text}")
        }
        pete_actions::ReignCommand::Chirp { pattern } => {
            format!("Chirp {:?}", pattern)
        }
        pete_actions::ReignCommand::SetMode { mode } => {
            format!("Set mode {:?}", mode)
        }
    }
}

fn reign_command_summary(now: &Now) -> Option<String> {
    let input = now.reign.latest.as_ref()?;
    if !reign_command_can_drive_agent(now, input) {
        return None;
    }
    Some(format!(
        "Following Reign command: {}",
        summarize_reign_command(input)
    ))
}

fn active_reign_action(now: &Now) -> Option<ActionPrimitive> {
    let input = now.reign.latest.as_ref()?;
    if !reign_command_can_drive_agent(now, input) {
        return None;
    }
    input.command.to_action()
}

fn action_spec_json(action: &ActionPrimitive) -> serde_json::Value {
    match action {
        ActionPrimitive::Stop => serde_json::json!({ "kind": "stop" }),
        ActionPrimitive::Go {
            intensity,
            duration_ms,
        } => serde_json::json!({
            "kind": "go",
            "intensity": intensity,
            "duration_ms": duration_ms,
        }),
        ActionPrimitive::Drive {
            forward,
            turn,
            duration_ms,
        } => serde_json::json!({
            "kind": "drive",
            "forward": forward,
            "turn": turn,
            "duration_ms": duration_ms,
        }),
        ActionPrimitive::Turn {
            direction,
            intensity,
            duration_ms,
        } => serde_json::json!({
            "kind": "turn",
            "direction": match direction {
                TurnDir::Left => "left",
                TurnDir::Right => "right",
            },
            "intensity": intensity,
            "duration_ms": duration_ms,
        }),
        ActionPrimitive::Inspect { target } => serde_json::json!({
            "kind": "inspect",
            "target": match target {
                InspectTarget::Novelty => "novelty",
                InspectTarget::Charger => "charger",
                InspectTarget::Person => "person",
                InspectTarget::Sound => "sound",
            },
        }),
        ActionPrimitive::Approach { target } => serde_json::json!({
            "kind": "approach",
            "target": match target {
                ApproachTarget::Charger => "charger",
                ApproachTarget::Person => "person",
                ApproachTarget::Sound => "sound",
            },
        }),
        ActionPrimitive::Dock => serde_json::json!({ "kind": "dock" }),
        ActionPrimitive::Explore { style, duration_ms } => serde_json::json!({
            "kind": "explore",
            "style": match style {
                ExploreStyle::Wander => "wander",
                ExploreStyle::RandomWalk => "random_walk",
                ExploreStyle::WallFollow => "wall_follow",
            },
            "duration_ms": duration_ms,
        }),
        ActionPrimitive::Speak { text } => serde_json::json!({
            "kind": "speak",
            "text": text,
        }),
        ActionPrimitive::Chirp { pattern } => serde_json::json!({
            "kind": "chirp",
            "pattern": chirp_pattern_name(pattern),
        }),
    }
}

fn chirp_pattern_name(pattern: &ChirpPattern) -> &'static str {
    match pattern {
        ChirpPattern::Confirm => "confirm",
        ChirpPattern::Warning => "warning",
        ChirpPattern::Hello => "hello",
        ChirpPattern::Goodbye => "goodbye",
        ChirpPattern::Curious => "curious",
        ChirpPattern::Idea => "idea",
        ChirpPattern::GoalAcquired => "goal_acquired",
        ChirpPattern::Searching => "searching",
        ChirpPattern::SawSomething => "saw_something",
        ChirpPattern::Surprise => "surprise",
        ChirpPattern::Learned => "learned",
        ChirpPattern::PersonRecognized => "person_recognized",
        ChirpPattern::ObjectRecognized => "object_recognized",
        ChirpPattern::PlaceRecognized => "place_recognized",
        ChirpPattern::DidntUnderstand => "didnt_understand",
        ChirpPattern::Docking => "docking",
        ChirpPattern::ChargingStarted => "charging_started",
        ChirpPattern::Sleep => "sleep",
        ChirpPattern::Wake => "wake",
    }
}

fn reign_command_can_drive_agent(now: &Now, input: &pete_actions::ReignInput) -> bool {
    if !now.reign.active {
        return false;
    }
    !matches!(input.mode, pete_actions::ReignMode::ObserveOnly)
}

fn summarize_futures(futures: &[FuturePrediction]) -> String {
    if futures.is_empty() {
        return "- none".to_string();
    }
    futures
        .iter()
        .map(|future| {
            format!(
                "- +{}ms confidence {:.2}{}",
                future.offset_ms,
                future.confidence,
                future
                    .summary
                    .as_ref()
                    .map(|summary| format!(": {summary}"))
                    .unwrap_or_default()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_combobulator_timeline(impressions: &[Impression]) -> String {
    if impressions.is_empty() {
        return "- no impressions".to_string();
    }

    let mut ordered = impressions.to_vec();
    ordered.sort_by_key(|impression| (impression.occurred_at_ms, impression.observed_at_ms));
    let start_ms = ordered
        .first()
        .map(|impression| impression.occurred_at_ms)
        .unwrap_or_default();
    let clusters = impression_clusters(&ordered, COMBOBULATOR_CLUSTER_GAP_MS);
    let mut out = String::new();
    for (index, cluster) in clusters.iter().enumerate() {
        if index > 0 {
            out.push('\n');
        }
        out.push_str(&format_impression_cluster(cluster, start_ms));
        for impression in *cluster {
            out.push_str(&format_impression_timeline_entry(impression, start_ms));
        }
    }
    out
}

fn local_iso_ms(t_ms: u64) -> String {
    match Local.timestamp_millis_opt(t_ms as i64).single() {
        Some(value) => value.to_rfc3339_opts(SecondsFormat::Millis, false),
        None => format!("{t_ms}ms"),
    }
}

fn impression_clusters(impressions: &[Impression], max_gap_ms: u64) -> Vec<&[Impression]> {
    if impressions.is_empty() {
        return Vec::new();
    }

    let mut clusters = Vec::new();
    let mut start = 0usize;
    let mut previous_ms = impressions[0].occurred_at_ms;
    for (index, impression) in impressions.iter().enumerate().skip(1) {
        if impression.occurred_at_ms.saturating_sub(previous_ms) > max_gap_ms {
            clusters.push(&impressions[start..index]);
            start = index;
        }
        previous_ms = impression.occurred_at_ms;
    }
    clusters.push(&impressions[start..]);
    clusters
}

fn format_impression_cluster(cluster: &[Impression], start_ms: u64) -> String {
    let first_ms = cluster
        .first()
        .map(|impression| impression.occurred_at_ms)
        .unwrap_or(start_ms);
    let last_ms = cluster
        .last()
        .map(|impression| impression.occurred_at_ms)
        .unwrap_or(first_ms);
    format!(
        "[T+{:06.3} - T+{:06.3} | {} to {}]\n",
        elapsed_seconds(start_ms, first_ms),
        elapsed_seconds(start_ms, last_ms),
        local_iso_ms(first_ms),
        local_iso_ms(last_ms)
    )
}

fn format_impression_timeline_entry(impression: &Impression, start_ms: u64) -> String {
    let generator = impression
        .payload
        .get("generator")
        .and_then(|value| value.as_str())
        .unwrap_or("unknown");
    let faculty = impression
        .payload
        .get("faculty")
        .and_then(|value| value.as_str())
        .unwrap_or("unknown");
    format!(
        "T+{:06.3} occurred_at={}\n  IMPRESSION id={} kind={} generator={} faculty={} observed_at={} confidence={:.3} about=[{}] payload={} text={}\n",
        elapsed_seconds(start_ms, impression.occurred_at_ms),
        local_iso_ms(impression.occurred_at_ms),
        impression.id,
        impression.kind,
        prompt_json_string(generator),
        prompt_json_string(faculty),
        local_iso_ms(impression.observed_at_ms),
        impression.confidence,
        impression
            .about
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(","),
        prompt_json_string(&impression.payload.to_string()),
        prompt_json_string(&impression.text)
    )
}

fn elapsed_seconds(start_ms: u64, t_ms: u64) -> f64 {
    t_ms.saturating_sub(start_ms) as f64 / MILLIS_PER_SECOND
}

fn prompt_json_string(text: &str) -> String {
    serde_json::to_string(text)
        .expect("prompt string fragment is serializable")
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
}
