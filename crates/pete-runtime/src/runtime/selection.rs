fn predict_baseline_futures(
    predictor: &mut ReplaceableBehavior<FutureInput, FuturePrediction>,
    latent: &ExperienceLatent,
    t_ms: TimeMs,
) -> Result<(Vec<FuturePrediction>, Vec<ErasedBehaviorRunRecord>)> {
    let mut out = Vec::new();
    let mut records = Vec::new();
    for action in default_candidate_actions() {
        for offset_ms in [100, 500, 1_000, 5_000] {
            let input = FutureInput {
                latent: latent.clone(),
                action: action.clone(),
                offset_ms,
            };
            let run = predictor.infer(&input, t_ms)?;
            out.push(run.chosen);
            records.push(run.record.erase());
        }
    }
    Ok((out, records))
}

fn attach_structured_predictions_to_experience(
    experience: &mut Experience,
    futures: &[FuturePrediction],
    now: &Now,
    action: Option<&ActionPrimitive>,
) {
    for future in futures.iter().take(2) {
        let text = future
            .summary
            .clone()
            .unwrap_or_else(|| "latent future estimated from ExperienceLatent".to_string());
        experience.predictions.push(Prediction {
            offset_ms: future.offset_ms,
            text: format!("next_state: {text}"),
            confidence: future.confidence.clamp(0.0, 1.0),
            vector: None,
        });
    }

    let danger = now
        .predictions
        .danger_model
        .or(now.predictions.danger_hardcoded);
    if let Some(danger) = danger {
        experience.predictions.push(Prediction {
            offset_ms: 100,
            text: format!(
                "hazard: bump={:.2} cliff={:.2} wheel_drop={:.2} stuck={:.2}",
                danger.bump_risk, danger.cliff_risk, danger.wheel_drop_risk, danger.stuck_risk
            ),
            confidence: danger.confidence.clamp(0.0, 1.0),
            vector: None,
        });
    }

    let charge = now
        .predictions
        .charge_model
        .or(now.predictions.charge_hardcoded);
    if let Some(charge) = charge {
        experience.predictions.push(Prediction {
            offset_ms: 500,
            text: format!(
                "charge: probability={:.2} battery_delta={:.3} dock={:.2}",
                charge.charge_probability, charge.expected_battery_delta, charge.dock_likelihood
            ),
            confidence: charge.confidence.clamp(0.0, 1.0),
            vector: None,
        });
    }

    let action_value = now
        .predictions
        .action_values_model
        .iter()
        .chain(now.predictions.action_values_hardcoded.iter())
        .find(|prediction| {
            action
                .map(|action| prediction.action == *action)
                .unwrap_or(true)
        });
    if let Some(action_value) = action_value {
        experience.predictions.push(Prediction {
            offset_ms: 250,
            text: format!(
                "action_value: action={:?} value={:.2}",
                action_value.action, action_value.value
            ),
            confidence: action_value.confidence.clamp(0.0, 1.0),
            vector: None,
        });
    }

    if !now.predictions.expected_events.is_empty() {
        experience.predictions.push(Prediction {
            offset_ms: 500,
            text: format!(
                "social_object_changes: expected_events={}",
                now.predictions.expected_events.join(", ")
            ),
            confidence: (1.0 - now.predictions.uncertainty).clamp(0.0, 1.0),
            vector: None,
        });
    }

    experience.predictions.push(Prediction {
        offset_ms: 500,
        text: format!(
            "uncertainty: {:.2}",
            now.predictions.uncertainty.clamp(0.0, 1.0)
        ),
        confidence: (1.0 - now.predictions.uncertainty).clamp(0.05, 1.0),
        vector: None,
    });
}

fn default_candidate_actions() -> Vec<ActionPrimitive> {
    vec![
        ActionPrimitive::Stop,
        ActionPrimitive::Go {
            intensity: 0.15,
            duration_ms: 1_000,
        },
        ActionPrimitive::Go {
            intensity: -0.12,
            duration_ms: 750,
        },
        ActionPrimitive::Turn {
            direction: TurnDir::Left,
            intensity: 0.25,
            duration_ms: 750,
        },
        ActionPrimitive::Turn {
            direction: TurnDir::Right,
            intensity: 0.25,
            duration_ms: 750,
        },
        ActionPrimitive::Inspect {
            target: InspectTarget::Novelty,
        },
        ActionPrimitive::Approach {
            target: ApproachTarget::Charger,
        },
        ActionPrimitive::Dock,
        ActionPrimitive::Explore {
            style: ExploreStyle::Wander,
            duration_ms: 2_000,
        },
    ]
}

#[derive(Clone, Copy, Debug, Default)]
struct CandidateModelSignals {
    danger: Option<DangerOutput>,
    charge: Option<ChargeOutput>,
    action_value: Option<ActionValueOutput>,
}

fn score_action_candidate(
    now: &Now,
    action: &ActionPrimitive,
    signals: CandidateModelSignals,
    previous_action: Option<&ActionPrimitive>,
) -> ActionSelectionCandidateScore {
    let danger = signals
        .danger
        .map(max_danger_risk)
        .unwrap_or_else(|| fallback_collision_risk(now, action));
    let charge = signals.charge.map(charge_score).unwrap_or_else(|| {
        if matches!(
            action,
            ActionPrimitive::Dock
                | ActionPrimitive::Approach {
                    target: ApproachTarget::Charger
                }
        ) {
            now.memory.place_charge_value.max(0.1)
        } else {
            0.0
        }
    });
    let action_value = signals.action_value.map(|value| value.value).unwrap_or(0.0);
    let (charger_near, charger_visible) = charger_signal_scores(now);
    let charger_contact_plausible = now.body.charging || charger_near >= 0.92;
    let charger_approach_bonus = if matches!(
        action,
        ActionPrimitive::Approach {
            target: ApproachTarget::Charger
        }
    ) {
        if charger_contact_plausible {
            0.08
        } else {
            let memory = now.memory.place_charge_value.clamp(0.0, 1.0);
            (charger_visible.max(charger_near) * 0.35 + memory * 0.18).min(0.45)
        }
    } else {
        0.0
    };
    let dock_distance_penalty =
        if matches!(action, ActionPrimitive::Dock) && !charger_contact_plausible {
            if charger_visible >= 0.20 || charger_near >= 0.25 {
                0.65
            } else {
                0.95
            }
        } else {
            0.0
        };
    let curiosity = curiosity_action_bonus(now, action);
    let collision_risk = fallback_collision_risk(now, action).max(danger);
    let low_battery_risk = if now.body.battery_level <= 0.2
        && matches!(
            action,
            ActionPrimitive::Go { .. } | ActionPrimitive::Explore { .. }
        ) {
        0.25
    } else {
        0.0
    };
    let repeat_penalty = if previous_action == Some(action) {
        0.03
    } else {
        0.0
    };
    let recovery_bonus = recovery_candidate_bonus(now, action, previous_action);
    let fallback_used =
        signals.danger.is_none() || signals.charge.is_none() || signals.action_value.is_none();
    let score = (-1.6 * danger)
        + (1.2 * charge)
        + action_value
        + curiosity
        + recovery_bonus
        + charger_approach_bonus
        - (0.8 * collision_risk)
        - low_battery_risk
        - dock_distance_penalty
        - repeat_penalty;

    ActionSelectionCandidateScore {
        action: action.clone(),
        score,
        danger,
        charge,
        action_value,
        curiosity,
        collision_risk,
        low_battery_risk,
        repeat_penalty,
        fallback_used,
    }
}

fn curiosity_action_bonus(now: &Now, action: &ActionPrimitive) -> f32 {
    let curiosity = now.drives.curiosity.clamp(0.0, 1.0);
    let novelty = now.memory.place_novelty.clamp(0.0, 1.0);
    let pressure = curiosity.max(novelty * 0.75);
    match action {
        ActionPrimitive::Explore { .. } => pressure * 0.24,
        ActionPrimitive::Inspect {
            target: InspectTarget::Novelty,
        } => pressure * 0.22,
        ActionPrimitive::Turn { .. } => pressure * 0.10,
        ActionPrimitive::Go { intensity, .. } if *intensity > 0.0 => pressure * 0.06,
        _ => 0.0,
    }
}

fn map_memory_decision_debug(
    now: &Now,
    chosen_action: &ActionPrimitive,
    baseline_action: Option<&ActionPrimitive>,
    forced_action: bool,
) -> MapMemoryDecisionDebug {
    let corrected_map_trust = corrected_map_trust_status(now);
    let mut debug = MapMemoryDecisionDebug {
        corrected_map_trusted: corrected_map_trust.trusted,
        corrected_map_untrusted_reason: corrected_map_trust.reason.clone(),
        place_danger: now.memory.place_danger,
        place_charge_value: now.memory.place_charge_value,
        place_novelty: now.memory.place_novelty,
        safe_direction_rad: now.memory.nearby_best_safe_direction_rad,
        charge_direction_rad: now.memory.nearby_best_charge_direction_rad,
        frontier_direction_rad: now.memory.nearby_frontier_direction_rad,
        recent_trap_direction_rad: now.memory.recent_trap_direction_rad,
        map_confidence: now.memory.map_confidence,
        recent_trap_confidence: now.memory.recent_trap_confidence,
        selected_action: Some(chosen_action.clone()),
        chosen_action: Some(chosen_action.clone()),
        ..MapMemoryDecisionDebug::default()
    };
    if forced_action || baseline_action != Some(chosen_action) {
        return debug;
    }

    debug.reason = map_memory_decision_reason(now, chosen_action);
    debug.influenced = debug.reason.is_some();
    if let Some(reason) = debug.reason.as_deref() {
        debug.navigation_intent = Some(map_memory_navigation_intent(reason));
        debug.reason_string = Some(map_memory_reason_string(reason, now));
        debug.signal = Some(map_memory_signal(reason));
        debug.signal_value = map_memory_signal_value(reason, now);
        debug.signal_confidence = map_memory_confidence(reason, now);
        debug.confidence = debug.signal_confidence;
    }
    debug
}

#[derive(Clone, Debug, Default, PartialEq)]
struct CorrectedMapTrustStatus {
    trusted: bool,
    reason: Option<String>,
}

fn corrected_map_trust_status(now: &Now) -> CorrectedMapTrustStatus {
    if let Some(sensor_truth) = now
        .extensions
        .get("sensor_truth")
        .or_else(|| now.extensions.get("geometry.sensor_truth"))
    {
        if sensor_truth
            .get("ready_for_real_slam")
            .and_then(serde_json::Value::as_bool)
            == Some(false)
        {
            return CorrectedMapTrustStatus {
                trusted: false,
                reason: Some("sensor_truth.ready_for_real_slam is false".to_string()),
            };
        }
    }

    let Some(map) = now.extensions.get(MAP_EXTENSION_NAME) else {
        return CorrectedMapTrustStatus {
            trusted: false,
            reason: Some(format!("{MAP_EXTENSION_NAME} summary is missing")),
        };
    };
    if let Some(slam_status) = map.get("slam_status") {
        let mode = slam_status
            .get("mode")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("odometry_only");
        if mode != "loop_closed_pose_graph" {
            let detail = slam_status
                .get("reasons")
                .and_then(serde_json::Value::as_array)
                .and_then(|reasons| reasons.iter().find_map(serde_json::Value::as_str))
                .unwrap_or("map is not in loop-closed pose-graph SLAM mode");
            return CorrectedMapTrustStatus {
                trusted: false,
                reason: Some(format!("slam_status.mode is {mode}: {detail}")),
            };
        }
        return CorrectedMapTrustStatus {
            trusted: true,
            reason: None,
        };
    }

    let accepted = map
        .get("loop_closures_accepted")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if accepted == 0 {
        return CorrectedMapTrustStatus {
            trusted: false,
            reason: Some("no accepted loop-closure edges in the live pose graph".to_string()),
        };
    }
    let optimized_nodes = map
        .get("pose_graph_optimization")
        .and_then(|value| value.get("optimized_nodes"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let active_edges = map
        .get("pose_graph_optimization")
        .and_then(|value| value.get("active_edges"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if optimized_nodes == 0 || active_edges == 0 {
        return CorrectedMapTrustStatus {
            trusted: false,
            reason: Some("pose graph has not optimized corrected live nodes".to_string()),
        };
    }
    let remap_submaps = map
        .get("remap")
        .and_then(|value| value.get("submaps"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let remap_generation = map
        .get("remap")
        .and_then(|value| value.get("generation"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if remap_submaps == 0 || remap_generation == 0 {
        return CorrectedMapTrustStatus {
            trusted: false,
            reason: Some("occupancy has not been rebuilt from corrected submaps".to_string()),
        };
    }

    CorrectedMapTrustStatus {
        trusted: true,
        reason: None,
    }
}

fn memory_for_navigation_with_map_trust(
    mut memory: MemorySense,
    trust: CorrectedMapTrustStatus,
) -> MemorySense {
    if trust.trusted {
        return memory;
    }
    memory.place_danger = 0.0;
    memory.place_charge_value = 0.0;
    memory.place_social_value = 0.0;
    memory.place_novelty = 0.0;
    memory.nearby_best_charge_direction_rad = None;
    memory.nearby_best_safe_direction_rad = None;
    memory.nearby_frontier_direction_rad = None;
    memory.recent_trap_direction_rad = None;
    memory.recent_trap_confidence = 0.0;
    memory.map_confidence = 0.0;
    memory
}

fn memory_navigation_candidate_context(now: &Now, action: &ActionPrimitive) -> bool {
    if !corrected_map_trust_status(now).trusted {
        return false;
    }
    map_memory_decision_reason(now, action).is_some()
}

fn map_memory_decision_reason(now: &Now, action: &ActionPrimitive) -> Option<String> {
    if !corrected_map_trust_status(now).trusted {
        return None;
    }
    const CRITICAL_BATTERY: f32 = 0.10;
    const LOW_BATTERY: f32 = 0.20;
    const DANGER_THRESHOLD: f32 = 0.70;
    const NOVELTY_THRESHOLD: f32 = 0.50;

    if now.memory.place_danger >= DANGER_THRESHOLD {
        if let ActionPrimitive::Turn { direction, .. } = action {
            if let Some(bearing) = now.memory.nearby_best_safe_direction_rad {
                let expected = if bearing < 0.0 {
                    TurnDir::Right
                } else {
                    TurnDir::Left
                };
                if direction == &expected {
                    return Some("danger_safe_direction".to_string());
                }
            }
            return Some("danger_current_cell".to_string());
        }
    }

    if now.memory.recent_trap_confidence >= 0.6 {
        if let ActionPrimitive::Turn { direction, .. } = action {
            if let Some(bearing) = now.memory.nearby_best_safe_direction_rad {
                let expected = if bearing < 0.0 {
                    TurnDir::Right
                } else {
                    TurnDir::Left
                };
                if direction == &expected {
                    return Some("recent_trap_safe_direction".to_string());
                }
            }
            return Some("recent_trap_turn".to_string());
        }
    }

    if now.body.battery_level <= LOW_BATTERY && now.memory.place_charge_value > 0.5 {
        match action {
            ActionPrimitive::Turn { direction, .. } => {
                if let Some(bearing) = now.memory.nearby_best_charge_direction_rad {
                    let expected = if bearing < 0.0 {
                        TurnDir::Right
                    } else {
                        TurnDir::Left
                    };
                    if bearing.abs() > 0.20 && direction == &expected {
                        return Some("charge_direction_turn".to_string());
                    }
                }
            }
            ActionPrimitive::Approach {
                target: ApproachTarget::Charger,
            } => return Some("charge_direction_aligned".to_string()),
            _ => {}
        }
    }

    if now.body.battery_level <= CRITICAL_BATTERY
        && matches!(action, ActionPrimitive::Stop)
        && now.memory.place_charge_value < 0.25
        && now.memory.nearby_best_charge_direction_rad.is_none()
    {
        return Some("charge_low_confidence_fallback".to_string());
    }

    if now.memory.place_novelty >= NOVELTY_THRESHOLD && now.memory.place_danger < DANGER_THRESHOLD {
        if let ActionPrimitive::Turn { direction, .. } = action {
            if let Some(bearing) = now.memory.nearby_frontier_direction_rad {
                let expected = if bearing < 0.0 {
                    TurnDir::Right
                } else {
                    TurnDir::Left
                };
                if direction == &expected {
                    return Some("frontier_direction_turn".to_string());
                }
            }
        }
        if matches!(
            action,
            ActionPrimitive::Inspect {
                target: InspectTarget::Novelty
            }
        ) {
            return Some("safe_novelty_inspect".to_string());
        }
    }

    None
}

fn map_memory_navigation_intent(reason: &str) -> NavigationIntent {
    if reason.starts_with("danger_") {
        NavigationIntent::AvoidKnownDangerCell
    } else if reason.starts_with("recent_trap_") {
        NavigationIntent::ReturnToFamiliarSafeCell
    } else if reason == "charge_low_confidence_fallback" {
        NavigationIntent::StopAskForHelpWhenUncertain
    } else if reason.starts_with("charge_") {
        NavigationIntent::GoTowardKnownCharger
    } else if reason.starts_with("frontier_") || reason.starts_with("safe_novelty_") {
        NavigationIntent::InspectSafeNovelFrontier
    } else {
        NavigationIntent::Explore
    }
}

fn map_memory_signal(reason: &str) -> String {
    match reason {
        "danger_safe_direction" => "memory.nearby_best_safe_direction_rad",
        "danger_current_cell" => "memory.place_danger",
        "recent_trap_safe_direction" => {
            "memory.recent_trap_confidence+nearby_best_safe_direction_rad"
        }
        "recent_trap_turn" => "memory.recent_trap_confidence",
        "charge_direction_turn" => "memory.nearby_best_charge_direction_rad",
        "charge_direction_aligned" => "memory.place_charge_value",
        "charge_low_confidence_fallback" => "memory.nearby_best_charge_direction_rad",
        "frontier_direction_turn" => "memory.nearby_frontier_direction_rad",
        "safe_novelty_inspect" => "memory.place_novelty",
        _ => "memory.map",
    }
    .to_string()
}

fn map_memory_signal_value(reason: &str, now: &Now) -> Option<f32> {
    match reason {
        "danger_safe_direction" | "recent_trap_safe_direction" => {
            now.memory.nearby_best_safe_direction_rad
        }
        "danger_current_cell" => Some(now.memory.place_danger),
        "recent_trap_turn" => Some(now.memory.recent_trap_confidence),
        "charge_direction_turn" => now.memory.nearby_best_charge_direction_rad,
        "charge_direction_aligned" => Some(now.memory.place_charge_value),
        "charge_low_confidence_fallback" => now.memory.nearby_best_charge_direction_rad,
        "frontier_direction_turn" => now.memory.nearby_frontier_direction_rad,
        "safe_novelty_inspect" => Some(now.memory.place_novelty),
        _ => None,
    }
}

fn map_memory_confidence(reason: &str, now: &Now) -> f32 {
    let (charger_near, charger_visible) = charger_signal_scores(now);
    let charge_confidence = charger_near
        .max(charger_visible)
        .max(now.memory.place_charge_value)
        .clamp(0.0, 1.0);
    match reason {
        reason if reason.starts_with("danger_") => now.memory.place_danger.clamp(0.0, 1.0),
        reason if reason.starts_with("recent_trap_") => {
            now.memory.recent_trap_confidence.clamp(0.0, 1.0)
        }
        reason if reason.starts_with("charge_") => charge_confidence,
        "frontier_direction_turn" | "safe_novelty_inspect" => {
            now.memory.place_novelty.clamp(0.0, 1.0)
        }
        _ => now.memory.map_confidence.clamp(0.0, 1.0),
    }
}

fn map_memory_reason_string(reason: &str, now: &Now) -> String {
    match reason {
        "danger_safe_direction" => format!(
            "avoiding remembered danger {:.2} using safe bearing {:?}",
            now.memory.place_danger, now.memory.nearby_best_safe_direction_rad
        ),
        "danger_current_cell" => format!(
            "avoiding remembered/current danger {:.2} with local range clearance",
            now.memory.place_danger
        ),
        "recent_trap_safe_direction" => format!(
            "returning toward familiar safe cell from trap confidence {:.2}",
            now.memory.recent_trap_confidence
        ),
        "recent_trap_turn" => format!(
            "turning away from recent trap confidence {:.2}",
            now.memory.recent_trap_confidence
        ),
        "charge_direction_turn" => format!(
            "turning toward remembered charger bearing {:?} with charge value {:.2}",
            now.memory.nearby_best_charge_direction_rad, now.memory.place_charge_value
        ),
        "charge_direction_aligned" => format!(
            "approaching charger from remembered charge value {:.2}",
            now.memory.place_charge_value
        ),
        "charge_low_confidence_fallback" => format!(
            "critical battery but charger memory is too weak: charge value {:.2}, bearing {:?}",
            now.memory.place_charge_value, now.memory.nearby_best_charge_direction_rad
        ),
        "frontier_direction_turn" => format!(
            "inspecting safe novel frontier bearing {:?} with novelty {:.2}",
            now.memory.nearby_frontier_direction_rad, now.memory.place_novelty
        ),
        "safe_novelty_inspect" => format!(
            "inspecting safe novel place with novelty {:.2}",
            now.memory.place_novelty
        ),
        _ => "memory/map signal influenced navigation".to_string(),
    }
}

fn select_action_from_scores(
    mode: ActionSelectorMode,
    now: &Now,
    baseline_action: ActionPrimitive,
    candidates: Vec<ActionSelectionCandidateScore>,
) -> ActionSelectionDecision {
    if mode != ActionSelectorMode::Baseline {
        if let Some(action) = hard_safety_action(now) {
            return ActionSelectionDecision {
                mode,
                candidates,
                selected_action: Some(action),
                baseline_action: Some(baseline_action),
                selected_score: None,
                safety_overrode: true,
                fallback_warnings: fallback_warnings_for_mode(mode),
                ..ActionSelectionDecision::default()
            };
        }
    }

    let selected = match mode {
        ActionSelectorMode::Baseline => Some(ActionSelectionCandidateScore {
            action: baseline_action.clone(),
            score: 0.0,
            ..ActionSelectionCandidateScore::default()
        }),
        ActionSelectorMode::Scripted => candidates
            .iter()
            .find(|candidate| {
                matches!(
                    candidate.action,
                    ActionPrimitive::Approach {
                        target: ApproachTarget::Charger
                    } | ActionPrimitive::Dock
                )
            })
            .cloned()
            .or_else(|| candidates.first().cloned()),
        ActionSelectorMode::Random => {
            if candidates.is_empty() {
                None
            } else {
                candidates
                    .get(now.t_ms as usize % candidates.len())
                    .cloned()
            }
        }
        ActionSelectorMode::ModelAssisted => candidates
            .iter()
            .max_by(|left, right| {
                left.score
                    .partial_cmp(&right.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned(),
        ActionSelectorMode::GoalShadow | ActionSelectorMode::Goal => {
            Some(ActionSelectionCandidateScore {
                action: baseline_action.clone(),
                score: 0.0,
                ..ActionSelectionCandidateScore::default()
            })
        }
    };
    let fallback_warnings = if candidates.iter().any(|candidate| candidate.fallback_used) {
        fallback_warnings_for_mode(mode)
    } else {
        Vec::new()
    };
    ActionSelectionDecision {
        mode,
        selected_action: selected
            .as_ref()
            .map(|candidate| candidate.action.clone())
            .or(Some(baseline_action.clone())),
        selected_score: selected.as_ref().map(|candidate| candidate.score),
        candidates,
        baseline_action: Some(baseline_action),
        safety_overrode: false,
        fallback_warnings,
        ..ActionSelectionDecision::default()
    }
}

fn fallback_warnings_for_mode(mode: ActionSelectorMode) -> Vec<String> {
    if mode == ActionSelectorMode::ModelAssisted {
        vec!["model-assisted selector used hardcoded fallback estimates".to_string()]
    } else {
        Vec::new()
    }
}

fn recovery_candidate_bonus(
    now: &Now,
    action: &ActionPrimitive,
    baseline_action: Option<&ActionPrimitive>,
) -> f32 {
    if !recovery_candidate_context(now) || !is_recovery_locomotion_action(action) {
        return 0.0;
    }
    if baseline_action == Some(action) {
        3.0
    } else {
        0.75
    }
}

fn recovery_candidate_context(now: &Now) -> bool {
    let contact = now.body.flags.bump_left || now.body.flags.bump_right || now.body.flags.wall;
    let close_range = now
        .range
        .nearest_m
        .map(|nearest| nearest < 0.35)
        .unwrap_or(false);
    contact || close_range || sim_stuck_active(now)
}

fn is_recovery_locomotion_action(action: &ActionPrimitive) -> bool {
    match action {
        ActionPrimitive::Go { intensity, .. } => intensity.abs() <= 0.25,
        ActionPrimitive::Turn { intensity, .. } => *intensity >= 0.5,
        _ => false,
    }
}

fn hard_safety_action(now: &Now) -> Option<ActionPrimitive> {
    if now.body.flags.wheel_drop {
        return Some(ActionPrimitive::Stop);
    }
    if now.body.flags.cliff_left
        || now.body.flags.cliff_front_left
        || now.body.flags.cliff_front_right
        || now.body.flags.cliff_right
    {
        return Some(ActionPrimitive::Stop);
    }
    if now.body.flags.bump_left || now.body.flags.bump_right || now.body.flags.wall {
        let direction = if now.body.flags.bump_left && !now.body.flags.bump_right {
            TurnDir::Right
        } else if now.body.flags.bump_right && !now.body.flags.bump_left {
            TurnDir::Left
        } else if range_clearer_on_right(&now.range.beams) {
            TurnDir::Right
        } else {
            TurnDir::Left
        };
        return Some(ActionPrimitive::Turn {
            direction,
            intensity: 0.7,
            duration_ms: 1_200,
        });
    }
    if now.body.battery_level <= 0.10 && !charger_reachable_signal(now) {
        return Some(ActionPrimitive::Stop);
    }
    let danger = now
        .predictions
        .danger_model
        .or(now.predictions.danger_hardcoded)
        .map(|prediction| {
            prediction
                .bump_risk
                .max(prediction.cliff_risk)
                .max(prediction.wheel_drop_risk)
                .max(prediction.stuck_risk)
        })
        .unwrap_or(0.0);
    if danger >= 0.70 {
        return Some(ActionPrimitive::Turn {
            direction: TurnDir::Left,
            intensity: 0.5,
            duration_ms: 1_000,
        });
    }
    None
}

fn charger_reachable_signal(now: &Now) -> bool {
    let (charger_near, charger_visible) = charger_signal_scores(now);
    now.body.charging
        || charger_near >= 0.25
        || charger_visible >= 0.20
        || now.memory.place_charge_value >= 0.5
        || now.memory.nearby_best_charge_direction_rad.is_some()
        || now
            .predictions
            .charge_model
            .or(now.predictions.charge_hardcoded)
            .map(|prediction| prediction.charge_probability >= 0.7)
            .unwrap_or(false)
}

fn range_clearer_on_right(beams: &[f32]) -> bool {
    if beams.len() < 2 {
        return false;
    }
    let (left, _, right) = beam_clearance_buckets(beams);
    right > left
}

fn max_danger_risk(output: DangerOutput) -> f32 {
    output
        .bump_risk
        .max(output.cliff_risk)
        .max(output.wheel_drop_risk)
        .max(output.stuck_risk)
        .clamp(0.0, 1.0)
}

fn charge_score(output: ChargeOutput) -> f32 {
    (output.charge_probability + output.dock_likelihood + output.expected_battery_delta.max(0.0))
        .clamp(0.0, 1.0)
}

fn fallback_collision_risk(now: &Now, action: &ActionPrimitive) -> f32 {
    let forward = action_to_motor_command(Some(action)).forward;
    let anticipated_risk = anticipated_surface_collision_risk(now);
    let nearest_risk = now
        .range
        .nearest_m
        .map(|nearest| ((0.35 - nearest) / 0.35).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let contact_risk =
        if now.body.flags.bump_left || now.body.flags.bump_right || now.body.flags.wall {
            1.0
        } else {
            0.0
        };
    if forward <= 0.0 {
        return anticipated_risk.max(contact_risk);
    }
    nearest_risk.max(contact_risk).max(anticipated_risk)
}

fn now_with_surface_anticipation(
    now: &Now,
    surface_output: Option<&SurfaceExtractorOutput>,
    action: &ActionPrimitive,
) -> Now {
    let Some(surface_output) = surface_output else {
        return now.clone();
    };
    let mut next = now.clone();
    let frames = anticipate_surfaces(surface_output, now.body.odometry, action);
    let max_risk = frames
        .iter()
        .map(|frame| frame.navigation.collision_risk)
        .fold(0.0f32, f32::max);
    let nearest_front = frames
        .iter()
        .filter_map(|frame| frame.navigation.front_clear_m)
        .min_by(|left, right| left.total_cmp(right));
    let anticipation_value = serde_json::json!({
        "action": action,
        "frames": frames,
        "max_collision_risk": max_risk,
        "nearest_front_clear_m": nearest_front,
    });
    let entry = next
        .extensions
        .entry("surface.scene_graph".to_string())
        .or_insert_with(|| serde_json::json!({}));
    if let Some(object) = entry.as_object_mut() {
        object.insert("anticipation".to_string(), anticipation_value);
    }
    next
}

fn anticipated_surface_collision_risk(now: &Now) -> f32 {
    now.extensions
        .get("surface.scene_graph")
        .and_then(|value| value.get("anticipation"))
        .and_then(|value| value.get("max_collision_risk"))
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0) as f32
}

fn action_value_candidate_actions(
    proposals: &[ActionPrimitive],
    reign_action: Option<&ActionPrimitive>,
) -> Vec<ActionPrimitive> {
    let mut candidates = default_candidate_actions();
    if let Some(action) = reign_action {
        push_unique_action(&mut candidates, action.clone());
    }
    for action in proposals {
        push_unique_action(&mut candidates, action.clone());
    }
    candidates
}

fn llm_explicit_safety_reason(llm_tick: &LlmTickResult) -> bool {
    let Some(decision) = llm_tick.decision.as_ref() else {
        return false;
    };
    [
        Some(decision.summary.as_str()),
        decision.critique.as_deref(),
    ]
    .into_iter()
    .flatten()
    .any(|text| {
        let text = text.to_ascii_lowercase();
        [
            "safety",
            "unsafe",
            "danger",
            "hazard",
            "collision",
            "cliff",
            "wheel drop",
            "blocked",
            "veto",
        ]
        .iter()
        .any(|needle| text.contains(needle))
    })
}

fn push_unique_action(actions: &mut Vec<ActionPrimitive>, action: ActionPrimitive) {
    if !actions.iter().any(|existing| existing == &action) {
        actions.push(action);
    }
}
