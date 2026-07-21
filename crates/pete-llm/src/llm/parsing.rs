fn heuristic_combobulation(now: &Now, recall_summary: &str) -> Combobulation {
    let summary = if let Some(transcript) = now
        .ear
        .transcript
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        format!("I hear: {transcript}")
    } else if now.body.flags.bump_left || now.body.flags.bump_right {
        "My body feels blocked by contact.".to_string()
    } else if now.body.flags.cliff_left
        || now.body.flags.cliff_front_left
        || now.body.flags.cliff_front_right
        || now.body.flags.cliff_right
        || now.body.cliff_sensors.max() >= 0.5
    {
        "I feel the floor fall away near me.".to_string()
    } else if let Some(nearest_m) = now.range.nearest_m {
        format!("Nearest obstacle is {:.2} meters away.", nearest_m)
    } else if !recall_summary.trim().is_empty() {
        recall_summary.trim().to_string()
    } else {
        format!("I am active at t={}ms.", now.t_ms)
    };
    Combobulation {
        summary,
        confidence: 0.35,
    }
}

fn parse_counterfactual_spec(spec: CounterfactualSpec) -> Option<CounterfactualAction> {
    Some(CounterfactualAction {
        instead_of: spec.instead_of.and_then(parse_action_spec),
        proposed: parse_action_spec(spec.proposed)?,
        reason: spec.reason,
        weight: spec.weight.clamp(0.0, 1.0),
    })
}

pub fn parse_llm_decision_json(text: &str, commands_enabled: bool) -> Result<LlmDecision> {
    let json = extract_json_object(text).unwrap_or_else(|| text.trim().to_string());
    let reply: AgentReply = serde_json::from_str(&json).context("failed to parse llm decision")?;
    let summary = reply.summary.trim().to_string();
    Ok(LlmDecision {
        summary,
        critique: reply
            .critique
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty()),
        confidence: reply.confidence.clamp(0.0, 1.0),
        action: if commands_enabled {
            reply.action.and_then(parse_action_spec)
        } else {
            None
        },
        counterfactuals: reply
            .counterfactuals
            .into_iter()
            .filter_map(parse_counterfactual_spec)
            .collect(),
        memory_notes: reply.memory_notes,
    })
}

pub fn parse_scientific_review_json(
    request: &LlmReviewRequest,
    text: &str,
) -> Result<LlmScientificReview> {
    let json = extract_json_object(text).unwrap_or_else(|| text.trim().to_string());
    let reply: ScientificReviewReply =
        serde_json::from_str(&json).context("failed to parse llm scientific review")?;
    Ok(scientific_review_from_reply(request, reply))
}

fn scientific_review_from_reply(
    request: &LlmReviewRequest,
    reply: ScientificReviewReply,
) -> LlmScientificReview {
    let default_id = stable_uuid_for_text(&format!(
        "llm-review:{}:{:?}:{}",
        request.target_id, request.target_kind, request.t_ms
    ))
    .to_string();
    LlmScientificReview {
        id: reply
            .id
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or(default_id),
        t_ms: request.t_ms,
        target_id: request.target_id.clone(),
        target_kind: request.target_kind.clone(),
        critique: reply
            .critique
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty()),
        counterfactuals: reply
            .counterfactuals
            .into_iter()
            .filter_map(parse_counterfactual_spec)
            .collect(),
        suggested_tests: reply
            .suggested_tests
            .into_iter()
            .map(|test| LlmSuggestedTest {
                action: test.action.and_then(parse_action_spec),
                question: test.question,
                expected_observation: test.expected_observation,
                disconfirming_observation: test.disconfirming_observation,
                risk_note: test
                    .risk_note
                    .map(|value| value.trim().to_string())
                    .filter(|value| !value.is_empty()),
                confidence: test.confidence.clamp(0.0, 1.0),
            })
            .collect(),
        suspicious_training_examples: reply
            .suspicious_training_examples
            .into_iter()
            .map(|warning| LlmTrainingWarning {
                example_id: if warning.example_id.trim().is_empty() {
                    request.target_id.clone()
                } else {
                    warning.example_id
                },
                reason: warning.reason,
                severity: warning.severity.clamp(0.0, 1.0),
                suspected_issue: warning.suspected_issue,
                supporting_evidence: warning.supporting_evidence,
                missing_evidence: warning.missing_evidence,
                suggested_fix: warning
                    .suggested_fix
                    .map(|value| value.trim().to_string())
                    .filter(|value| !value.is_empty()),
            })
            .collect(),
        label_proposals: reply
            .label_proposals
            .into_iter()
            .map(|proposal| LlmLabelProposal {
                example_id: if proposal.example_id.trim().is_empty() {
                    request.target_id.clone()
                } else {
                    proposal.example_id
                },
                proposed_label: proposal.proposed_label,
                rationale: proposal.rationale,
                confidence: proposal.confidence.clamp(0.0, 1.0),
                requires_human_review: proposal.requires_human_review,
            })
            .collect(),
        human_review_prompts: reply
            .human_review_prompts
            .into_iter()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .collect(),
        confidence: reply.confidence.clamp(0.0, 1.0),
    }
}

fn parse_action_spec(spec: ActionSpec) -> Option<ActionPrimitive> {
    let kind = spec.kind.to_ascii_lowercase();
    match kind.as_str() {
        "stop" => Some(ActionPrimitive::Stop),
        "go" => Some(ActionPrimitive::Go {
            intensity: spec.intensity.unwrap_or(0.2).clamp(0.0, 1.0),
            duration_ms: spec.duration_ms.unwrap_or(1_000),
        }),
        "turn" => Some(ActionPrimitive::Turn {
            direction: match spec.direction.as_deref()?.to_ascii_lowercase().as_str() {
                "left" => TurnDir::Left,
                "right" => TurnDir::Right,
                _ => return None,
            },
            intensity: spec.intensity.unwrap_or(0.4).clamp(0.0, 1.0),
            duration_ms: spec.duration_ms.unwrap_or(1_000),
        }),
        "inspect" => Some(ActionPrimitive::Inspect {
            target: match spec.target.as_deref()?.to_ascii_lowercase().as_str() {
                "novelty" => InspectTarget::Novelty,
                "charger" => InspectTarget::Charger,
                "person" => InspectTarget::Person,
                "sound" => InspectTarget::Sound,
                _ => return None,
            },
        }),
        "approach" => Some(ActionPrimitive::Approach {
            target: match spec.target.as_deref()?.to_ascii_lowercase().as_str() {
                "charger" => ApproachTarget::Charger,
                "person" => ApproachTarget::Person,
                "sound" => ApproachTarget::Sound,
                _ => return None,
            },
        }),
        "dock" => Some(ActionPrimitive::Dock),
        "explore" => Some(ActionPrimitive::Explore {
            style: match spec
                .style
                .as_deref()
                .unwrap_or("random_walk")
                .to_ascii_lowercase()
                .as_str()
            {
                "wander" => ExploreStyle::Wander,
                "random_walk" => ExploreStyle::RandomWalk,
                "wall_follow" => ExploreStyle::WallFollow,
                _ => return None,
            },
            duration_ms: spec.duration_ms.unwrap_or(1_000),
        }),
        "speak" => Some(ActionPrimitive::Speak {
            text: spec.text.unwrap_or_default(),
        }),
        "chirp" => Some(ActionPrimitive::Chirp {
            pattern: parse_chirp_pattern(spec.pattern.as_deref().unwrap_or("confirm"))?,
        }),
        _ => None,
    }
}

fn parse_chirp_pattern(pattern: &str) -> Option<ChirpPattern> {
    match pattern
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase()
        .as_str()
    {
        "confirm" => Some(ChirpPattern::Confirm),
        "warning" => Some(ChirpPattern::Warning),
        "hello" => Some(ChirpPattern::Hello),
        "goodbye" => Some(ChirpPattern::Goodbye),
        "curious" => Some(ChirpPattern::Curious),
        "idea" => Some(ChirpPattern::Idea),
        "goalacquired" => Some(ChirpPattern::GoalAcquired),
        "searching" => Some(ChirpPattern::Searching),
        "sawsomething" => Some(ChirpPattern::SawSomething),
        "surprise" => Some(ChirpPattern::Surprise),
        "learned" => Some(ChirpPattern::Learned),
        "personrecognized" => Some(ChirpPattern::PersonRecognized),
        "objectrecognized" => Some(ChirpPattern::ObjectRecognized),
        "placerecognized" => Some(ChirpPattern::PlaceRecognized),
        "didntunderstand" => Some(ChirpPattern::DidntUnderstand),
        "docking" => Some(ChirpPattern::Docking),
        "chargingstarted" => Some(ChirpPattern::ChargingStarted),
        "sleep" => Some(ChirpPattern::Sleep),
        "wake" => Some(ChirpPattern::Wake),
        _ => None,
    }
}

fn extract_json_object(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if let Some(json) = normalize_json_candidate(trimmed) {
        return Some(json);
    }

    if let Some(unfenced) = strip_markdown_fence(trimmed) {
        if let Some(json) = normalize_json_candidate(&unfenced) {
            return Some(json);
        }
    }

    let mut start = None;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (index, ch) in text.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            continue;
        }

        if ch == '"' {
            in_string = true;
            continue;
        }

        if ch == '{' {
            if start.is_none() {
                start = Some(index);
            }
            depth += 1;
        } else if ch == '}' {
            if depth == 0 {
                continue;
            }
            depth -= 1;
            if depth == 0 {
                let candidate = &text[start?..=index];
                if let Some(json) = normalize_json_candidate(candidate) {
                    return Some(json);
                }
            }
        }
    }
    None
}

fn normalize_json_candidate(candidate: &str) -> Option<String> {
    let trimmed = candidate.trim();
    if serde_json::from_str::<Value>(trimmed).is_ok() {
        return Some(trimmed.to_string());
    }

    if let Ok(json_text) = serde_json::from_str::<String>(trimmed) {
        let inner = json_text.trim();
        if serde_json::from_str::<Value>(inner).is_ok() {
            return Some(inner.to_string());
        }
    }

    None
}

fn strip_markdown_fence(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if !(trimmed.starts_with("```") && trimmed.ends_with("```")) {
        return None;
    }

    let mut lines = trimmed.lines();
    let first = lines.next()?;
    if !first.starts_with("```") {
        return None;
    }

    let mut content = lines.collect::<Vec<_>>();
    if content.last().copied() != Some("```") {
        return None;
    }
    content.pop();
    Some(content.join("\n").trim().to_string())
}
