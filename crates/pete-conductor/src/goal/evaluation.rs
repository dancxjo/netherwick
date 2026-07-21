type EvaluationParts = (f32, f32, f32, Vec<Affordance>, Vec<EvaluationContribution>);

fn evaluate_seek_charger(
    interpretation: &GoalInterpretation,
    context: &GoalEvaluationContext<'_>,
) -> EvaluationParts {
    let energy = context.drives.energy.activation;
    let urgency = ((0.25 - context.world.self_model.battery_level) / 0.20).clamp(0.0, 1.0);
    let confidence = interpretation.target_confidence;
    let mut affordances = Vec::new();
    match interpretation.target_distance_m {
        Some(distance) if distance <= 0.35 && confidence >= 0.65 => {
            affordances.push(
                affordance(
                    "dock",
                    ActionPrimitive::Dock,
                    confidence,
                    1.0,
                    1.0,
                    0.05,
                    0.02,
                    2_000,
                    interpretation.target.clone(),
                    &interpretation.provenance,
                )
                .with_bearing(interpretation.target_bearing_rad)
                .with_skill(SkillId::AlignWithDock, Some(0.20))
                .with_skill_range(interpretation.target_distance_m),
            );
        }
        Some(distance) if distance > 0.35 => affordances.push(rejected_affordance(
            "dock",
            "charger is outside docking range",
            interpretation.target.clone(),
            interpretation.target_bearing_rad,
            &interpretation.provenance,
        )),
        Some(_) => affordances.push(rejected_affordance(
            "dock",
            "charger confidence is too low for docking",
            interpretation.target.clone(),
            interpretation.target_bearing_rad,
            &interpretation.provenance,
        )),
        None => affordances.push(rejected_affordance(
            "dock",
            "no localized charger target",
            None,
            None,
            &interpretation.provenance,
        )),
    }
    if let Some(bearing) = interpretation.target_bearing_rad {
        if confidence < 0.35 {
            affordances.push(rejected_affordance(
                "approach_charger",
                "charger confidence is too low for locomotion",
                interpretation.target.clone(),
                Some(bearing),
                &interpretation.provenance,
            ));
        } else if !interpretation.target_reachable {
            affordances.push(rejected_affordance(
                "approach_charger",
                "the charger target is not currently reachable",
                interpretation.target.clone(),
                Some(bearing),
                &interpretation.provenance,
            ));
        } else if bearing.abs() > 0.20 {
            affordances.push(
                affordance(
                    "turn_toward_charger",
                    ActionPrimitive::Turn {
                        direction: if bearing >= 0.0 {
                            TurnDir::Left
                        } else {
                            TurnDir::Right
                        },
                        intensity: 0.4,
                        duration_ms: 700,
                    },
                    confidence,
                    0.65,
                    0.75,
                    interpretation.danger * 0.25,
                    0.05,
                    700,
                    interpretation.target.clone(),
                    &interpretation.provenance,
                )
                .with_bearing(Some(bearing))
                .with_skill(SkillId::TurnTowardTarget, None)
                .with_skill_range(interpretation.target_distance_m),
            );
        } else {
            affordances.push(
                affordance(
                    "approach_charger",
                    ActionPrimitive::Approach {
                        target: ApproachTarget::Charger,
                    },
                    confidence,
                    0.8,
                    0.9,
                    interpretation.danger,
                    0.15,
                    1_000,
                    interpretation.target.clone(),
                    &interpretation.provenance,
                )
                .with_bearing(Some(bearing))
                .with_skill(SkillId::ApproachTarget, Some(0.30))
                .with_skill_range(interpretation.target_distance_m),
            );
        }
    } else {
        affordances.push(rejected_affordance(
            "approach_charger",
            "charger bearing is unknown",
            interpretation.target.clone(),
            None,
            &interpretation.provenance,
        ));
    }
    affordances.push(affordance(
        "inspect_for_charger",
        ActionPrimitive::Inspect {
            target: InspectTarget::Charger,
        },
        (1.0 - confidence).max(0.35),
        0.35,
        0.35,
        interpretation.danger * 0.25,
        0.03,
        750,
        interpretation.target.clone(),
        &interpretation.provenance,
    ));
    affordances.push(
        affordance(
            "systematic_charger_search",
            ActionPrimitive::Explore {
                style: ExploreStyle::WallFollow,
                duration_ms: 1_000,
            },
            (1.0 - confidence).max(0.25),
            0.8,
            0.20,
            interpretation.danger,
            0.2,
            1_000,
            None,
            &interpretation.provenance,
        )
        .with_skill(SkillId::SystematicSearch, None),
    );
    if urgency > 0.8 && confidence < 0.2 && context.runtime.frustration > 0.6 {
        affordances.push(affordance(
            "request_charge_help",
            ActionPrimitive::Speak {
                // Solresol: "Help! I'm hungry!" (dosido = help; dsod = hungry).
                text: "Dosido! Dore dsod!".to_string(),
            },
            0.9,
            0.55,
            0.5,
            0.0,
            0.0,
            2_000,
            None,
            &[],
        ));
    }
    if let Some(question) = context
        .world
        .epistemic
        .active_questions
        .iter()
        .find(|question| question.family == EpistemicQuestionFamily::ChargerIdentityOrBearing)
    {
        for goal_affordance in &mut affordances {
            let epistemic_behavior = match goal_affordance.behavior_id.as_str() {
                "turn_toward_charger" => Some("orient_for_charger_evidence"),
                "inspect_for_charger" => Some("inspect_charger_hypothesis"),
                "systematic_charger_search" => Some("search_for_charger_evidence"),
                _ => None,
            };
            let Some(epistemic_behavior) = epistemic_behavior else {
                continue;
            };
            if let Some(epistemic) = context
                .world
                .epistemic
                .affordances
                .iter()
                .find(|candidate| {
                    candidate.question_id == question.question_id
                        && candidate.behavior_id == epistemic_behavior
                })
            {
                goal_affordance.epistemic_question_id = Some(question.question_id.clone());
                goal_affordance.expected_information_gain = epistemic.expected_information_gain;
                goal_affordance.expected_uncertainty_after =
                    Some(epistemic.expected_uncertainty_after);
            }
        }
    }
    let dock_available = affordances
        .iter()
        .any(|affordance| affordance.behavior_id == "dock" && affordance.available);
    if context.world.self_model.contact && !dock_available {
        for affordance in &mut affordances {
            affordance.available = false;
            affordance.rejection_reason = Some(
                "immediate contact must be cleared before charger seeking resumes".to_string(),
            );
        }
    }
    for goal_affordance in &mut affordances {
        goal_affordance.semantic_relation_ids = charger_affordance_semantics(
            context.world,
            interpretation.target.as_ref(),
            &goal_affordance.behavior_id,
        );
    }
    let semantic_confidence = context
        .world
        .semantic
        .relations
        .values()
        .filter(|relation| {
            relation.subject == SemanticNodeRef::Concept(SemanticConceptId("charger".to_string()))
                && matches!(
                    relation.predicate,
                    SemanticPredicate::Restores | SemanticPredicate::SatisfiesDrive
                )
        })
        .map(|relation| relation.confidence)
        .fold(0.0f32, f32::max);
    (
        (0.85 * energy + 0.15 * confidence).clamp(0.0, 1.0),
        urgency,
        context.drives.energy.satisfaction,
        affordances,
        vec![
            contribution("drive.energy", energy),
            contribution("world.charger_confidence", confidence),
            contribution("semantic.charger_energy_meaning", semantic_confidence),
        ],
    )
}

fn charger_affordance_semantics(
    world: &WorldModelSnapshot,
    target: Option<&EntityId>,
    behavior_id: &str,
) -> Vec<SemanticRelationId> {
    let charger = SemanticNodeRef::Concept(SemanticConceptId("charger".to_string()));
    let semantic_behavior = match behavior_id {
        "dock" => Some("dock"),
        "approach_charger" | "turn_toward_charger" => Some("approach_charger"),
        _ => None,
    };
    world
        .semantic
        .relations
        .values()
        .filter(|relation| {
            (relation.subject == charger
                && matches!(
                    relation.predicate,
                    SemanticPredicate::Restores
                        | SemanticPredicate::SatisfiesDrive
                        | SemanticPredicate::HelpsGoal
                ))
                || semantic_behavior.is_some_and(|behavior| {
                    relation.subject == charger
                        && relation.predicate == SemanticPredicate::Affords
                        && relation.object
                            == SemanticNodeRef::Behavior(SemanticBehaviorId(behavior.to_string()))
                })
                || target.is_some_and(|target| {
                    (relation.subject == SemanticNodeRef::Entity(target.clone())
                        && relation.predicate == SemanticPredicate::IsA
                        && relation.object == charger)
                        || (relation.predicate == SemanticPredicate::Blocks
                            && relation.object == SemanticNodeRef::Entity(target.clone()))
                })
        })
        .map(|relation| relation.id.clone())
        .collect()
}

fn evaluate_escape(
    interpretation: &GoalInterpretation,
    context: &GoalEvaluationContext<'_>,
) -> EvaluationParts {
    let danger = interpretation.danger;
    let contact = context.world.self_model.contact;
    let stuck = context.world.self_model.stuck;
    let corner_trap = context
        .world
        .self_model
        .stuck_trap_kind
        .as_ref()
        .is_some_and(|belief| belief.value == pete_now::StuckTrapKind::Corner);
    let confidence = context
        .world
        .context
        .map_confidence
        .as_ref()
        .map(|belief| belief.value)
        .unwrap_or(0.0)
        .max(0.5);
    let direction = if context.world.self_model.bump_left {
        TurnDir::Right
    } else if context.world.self_model.contact {
        TurnDir::Left
    } else if context
        .world
        .local_geometry
        .right_clearance_m
        .as_ref()
        .map(|belief| belief.value)
        .unwrap_or(0.0)
        > context
            .world
            .local_geometry
            .left_clearance_m
            .as_ref()
            .map(|belief| belief.value)
            .unwrap_or(0.0)
    {
        TurnDir::Right
    } else if let Some(bearing) = context
        .world
        .context
        .safe_bearing_rad
        .as_ref()
        .map(|belief| belief.value)
    {
        if bearing >= 0.0 {
            TurnDir::Left
        } else {
            TurnDir::Right
        }
    } else {
        TurnDir::Left
    };
    let mut affordances = Vec::new();
    if contact || (stuck && !corner_trap) {
        affordances.push(
            affordance(
                "reverse_from_danger",
                ActionPrimitive::Go {
                    intensity: -0.18,
                    duration_ms: 300,
                },
                0.95,
                0.7,
                0.8,
                0.1,
                0.08,
                300,
                None,
                &[],
            )
            .with_skill(SkillId::BackAway, None),
        );
    }
    let clearance_bearing = Some(match &direction {
        TurnDir::Left => 0.75,
        TurnDir::Right => -0.75,
    });
    affordances.push(
        affordance(
            "turn_toward_clearance",
            ActionPrimitive::Turn {
                direction: direction.clone(),
                intensity: 0.75,
                duration_ms: 500,
            },
            confidence,
            0.65,
            0.7,
            0.15,
            0.08,
            500,
            None,
            &[],
        )
        .with_bearing(clearance_bearing)
        .with_skill(SkillId::TurnTowardTarget, None),
    );
    let center_clearance = context
        .world
        .local_geometry
        .center_clearance_m
        .as_ref()
        .map(|belief| belief.value);
    if center_clearance.is_some_and(|clearance| clearance >= 0.30) || (corner_trap && !contact) {
        affordances.push(affordance(
            "probe_clearance",
            ActionPrimitive::Go {
                intensity: 0.14,
                duration_ms: 300,
            },
            confidence,
            0.55,
            0.65,
            0.15,
            0.05,
            300,
            None,
            &[],
        ));
    } else {
        affordances.push(rejected_affordance(
            "probe_clearance",
            "center clearance is below 0.30 m or unknown",
            None,
            None,
            &[],
        ));
    }
    affordances.push(affordance(
        "inspect_clearance",
        ActionPrimitive::Inspect {
            target: InspectTarget::Novelty,
        },
        confidence * (1.0 - interpretation.danger * 0.5),
        0.5,
        0.35,
        0.0,
        0.01,
        500,
        None,
        &[],
    ));
    (
        danger.max(if contact { 1.0 } else { 0.0 }),
        danger.max(if contact { 1.0 } else { 0.0 }),
        context.drives.safety.satisfaction,
        affordances,
        vec![contribution("drive.safety", danger)],
    )
}

fn evaluate_explore(
    interpretation: &GoalInterpretation,
    context: &GoalEvaluationContext<'_>,
) -> EvaluationParts {
    let drives = context.drives;
    let activation = (0.15 + 0.65 * drives.curiosity.activation
        - 0.55 * drives.energy.activation
        - 0.65 * drives.safety.activation
        - 0.50 * drives.rest.activation
        - 0.25 * drives.certainty.activation)
        .clamp(0.0, 1.0);
    let frontier_bearing = context
        .world
        .context
        .frontier_bearing_rad
        .as_ref()
        .map(|belief| belief.value);
    let mut affordances = vec![
        affordance(
            "random_walk_exploration",
            ActionPrimitive::Explore {
                style: ExploreStyle::RandomWalk,
                duration_ms: 1_000,
            },
            (1.0 - interpretation.danger).clamp(0.0, 1.0),
            0.45,
            0.6,
            interpretation.danger,
            0.2,
            1_000,
            None,
            &[],
        )
        .with_skill(SkillId::SystematicSearch, None),
        affordance(
            "wall_follow_exploration",
            ActionPrimitive::Explore {
                style: ExploreStyle::WallFollow,
                duration_ms: 1_000,
            },
            (0.9 - interpretation.danger).clamp(0.0, 1.0),
            0.4,
            0.55,
            interpretation.danger,
            0.18,
            1_000,
            None,
            &[],
        )
        .with_skill(SkillId::SystematicSearch, None),
    ];
    if let Some(bearing) = frontier_bearing {
        affordances.push(
            affordance(
                "follow_frontier",
                ActionPrimitive::Turn {
                    direction: if bearing >= 0.0 {
                        TurnDir::Left
                    } else {
                        TurnDir::Right
                    },
                    intensity: 0.35,
                    duration_ms: 500,
                },
                (1.0 - interpretation.danger).clamp(0.0, 1.0),
                0.55,
                0.7,
                interpretation.danger,
                0.12,
                700,
                None,
                &[],
            )
            .with_bearing(Some(bearing))
            .with_skill(SkillId::FollowBearing, None),
        );
    }
    if interpretation.novelty > 0.55 {
        affordances.push(affordance(
            "inspect_novelty",
            ActionPrimitive::Inspect {
                target: InspectTarget::Novelty,
            },
            (1.0 - interpretation.danger).clamp(0.0, 1.0),
            0.5,
            0.5,
            interpretation.danger * 0.5,
            0.05,
            750,
            None,
            &[],
        ));
    }
    (
        activation,
        0.1,
        drives.curiosity.satisfaction,
        affordances,
        vec![contribution("drive.curiosity", drives.curiosity.activation)],
    )
}

fn evaluate_socialize(
    interpretation: &GoalInterpretation,
    context: &GoalEvaluationContext<'_>,
) -> EvaluationParts {
    let social = context.drives.social.activation;
    let person = context.world.social.most_relevant_person();
    let identity = person.and_then(|person| person.best_identity());
    let identity_confidence = identity.map(|identity| identity.confidence).unwrap_or(0.0);
    let confidence = person
        .map(|person| person.presence.confidence)
        .unwrap_or(interpretation.target_confidence)
        .max(identity_confidence * 0.8);
    let identity_uncertain = person.is_some_and(|person| person.identity_is_uncertain());
    if person.is_some() && !identity_uncertain {
        // Recognition creates an encounter-scoped `greet_person` goal. The
        // general social goal must not reproduce the old direct greeting.
        return (0.0, 0.0, 1.0, Vec::new(), Vec::new());
    }
    let person_target = person.map(|person| EntityId(person.person_id.0.clone()));
    let person_distance = person
        .and_then(|person| person.location.as_ref())
        .and_then(|location| location.distance_m)
        .or(interpretation.target_distance_m);
    let person_bearing = person
        .and_then(|person| person.location.as_ref())
        .and_then(|location| location.bearing_rad)
        .or(interpretation.target_bearing_rad);
    let action = match person_distance {
        Some(distance) if distance <= 0.8 => ActionPrimitive::Speak {
            text: "Hello. What should I call you?".to_string(),
        },
        Some(_) => ActionPrimitive::Approach {
            target: ApproachTarget::Person,
        },
        None => ActionPrimitive::Inspect {
            target: InspectTarget::Person,
        },
    };
    let mut engagement = affordance(
        if identity_uncertain {
            "clarify_person_identity"
        } else {
            "social_engagement"
        },
        action.clone(),
        confidence.max(0.25),
        0.55,
        0.55,
        interpretation.danger,
        0.1,
        1_000,
        person_target.or_else(|| interpretation.target.clone()),
        person
            .map(|person| person.meta.provenance.as_slice())
            .unwrap_or(&interpretation.provenance),
    )
    .with_bearing(person_bearing);
    if matches!(action, ActionPrimitive::Approach { .. }) {
        engagement = engagement
            .with_skill(SkillId::ApproachTarget, Some(0.75))
            .with_skill_range(person_distance);
    }
    if identity_uncertain {
        if let Some(epistemic) = context
            .world
            .epistemic
            .affordances
            .iter()
            .filter(|affordance| affordance.action_kind == EpistemicActionKind::AskPerson)
            .max_by(|left, right| {
                left.epistemic_utility()
                    .total_cmp(&right.epistemic_utility())
            })
        {
            engagement = engagement.with_epistemic(epistemic);
        }
    }
    let pending_request = context
        .world
        .social
        .active_interaction
        .as_ref()
        .is_some_and(|interaction| !interaction.unresolved_requests.is_empty());
    (
        (0.70 * social + 0.30 * confidence + if pending_request { 0.20 } else { 0.0 }
            - 0.60 * interpretation.danger
            - 0.40 * context.drives.rest.activation)
            .clamp(0.0, 1.0),
        0.2,
        context.drives.social.satisfaction,
        vec![engagement],
        vec![
            contribution("drive.social", social),
            contribution("world.social.person_confidence", confidence),
            contribution("world.social.pending_request", pending_request as u8 as f32),
        ],
    )
}

fn evaluate_greet_person(
    interpretation: &GoalInterpretation,
    context: &GoalEvaluationContext<'_>,
) -> EvaluationParts {
    let Some(interaction) = context.world.social.active_interaction.as_ref() else {
        return (0.0, 0.0, 1.0, Vec::new(), Vec::new());
    };
    let candidate = interaction
        .participants
        .iter()
        .filter_map(|person_id| context.world.social.people.get(person_id))
        .filter(|person| person.presence.present && !person.identity_is_uncertain())
        .filter(|person| {
            !interaction.has_acknowledgment(
                &person.person_id,
                SocialAcknowledgmentKind::GreetingAttempted,
            )
        })
        .max_by(|left, right| {
            let left_score = left.presence.confidence + left.current_identity_confidence;
            let right_score = right.presence.confidence + right.current_identity_confidence;
            left_score.total_cmp(&right_score)
        });
    let Some(person) = candidate else {
        return (0.0, 0.0, 1.0, Vec::new(), Vec::new());
    };
    let confidence = person
        .presence
        .confidence
        .min(person.current_identity_confidence)
        .clamp(0.0, 1.0);
    let name = person
        .preferred_name
        .as_ref()
        .map(|name| name.value.as_str())
        .unwrap_or("recognized person");
    let behavior_id = format!(
        "greet:{}:{}",
        person.person_id.0, interaction.interaction_id.0
    );
    let mut greeting = affordance(
        &behavior_id,
        ActionPrimitive::Speak {
            text: format!("Greet {name}"),
        },
        confidence,
        0.65,
        1.0,
        interpretation.danger,
        0.02,
        5_000,
        Some(EntityId(person.person_id.0.clone())),
        &person.meta.provenance,
    )
    .with_bearing(
        person
            .location
            .as_ref()
            .and_then(|location| location.bearing_rad),
    )
    .with_runtime_skill("motherbrain.greet");
    if let Some(request) = &mut greeting.skill_request {
        request.range_m = person
            .location
            .as_ref()
            .and_then(|location| location.distance_m);
        request.progress_metric = "social_acknowledgment".to_string();
        request.progress_baseline = Some(0.0);
    }
    let danger = context.drives.safety.activation.max(interpretation.danger);
    let activation =
        (0.45 + 0.20 * confidence - 0.80 * danger - 0.50 * context.drives.rest.activation)
            .clamp(0.0, 1.0);
    (
        activation,
        0.35,
        0.0,
        vec![greeting],
        vec![
            contribution("world.social.new_recognized_encounter", 1.0),
            contribution("world.social.identity_confidence", confidence),
            contribution("drive.safety", -danger),
        ],
    )
}

fn evaluate_rest(
    _interpretation: &GoalInterpretation,
    context: &GoalEvaluationContext<'_>,
) -> EvaluationParts {
    let rest = context.drives.rest.activation;
    (
        rest,
        if context.world.self_model.charging {
            0.8
        } else {
            rest * 0.5
        },
        context.drives.rest.satisfaction,
        vec![affordance(
            "remain_stationary",
            ActionPrimitive::Stop,
            1.0,
            0.35,
            0.5,
            0.0,
            0.0,
            1_000,
            None,
            &[],
        )],
        vec![contribution("drive.rest", rest)],
    )
}

fn evaluate_investigate(
    interpretation: &GoalInterpretation,
    context: &GoalEvaluationContext<'_>,
) -> EvaluationParts {
    let uncertainty = context.drives.certainty.activation;
    let frustration = interpretation.stalled_goal_frustration;
    let question = context.world.epistemic.most_important_question();
    let epistemic_pressure = question
        .map(|question| question.importance * question.current_uncertainty)
        .unwrap_or(0.0);
    let mut affordances = question
        .map(|question| {
            context
                .world
                .epistemic
                .affordances_for(&question.question_id)
                .filter(|affordance| affordance.available)
                .map(conductor_epistemic_affordance)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if affordances.is_empty() {
        affordances.push(affordance(
            "gather_evidence",
            if interpretation.target.is_some() {
                ActionPrimitive::Inspect {
                    target: InspectTarget::Sound,
                }
            } else {
                ActionPrimitive::Inspect {
                    target: InspectTarget::Novelty,
                }
            },
            (1.0 - uncertainty).max(0.3),
            0.45,
            0.6,
            interpretation.danger * 0.25,
            0.05,
            750,
            interpretation.target.clone(),
            &interpretation.provenance,
        ));
    }
    (
        (0.50 * uncertainty
            + 0.55 * epistemic_pressure
            + 0.25
                * context
                    .world
                    .context
                    .surprise
                    .as_ref()
                    .map(|belief| belief.value)
                    .unwrap_or(0.0)
            + 0.35 * frustration
            - 0.50 * interpretation.danger)
            .clamp(0.0, 1.0),
        (0.25 + frustration * 0.5).clamp(0.0, 1.0),
        context.drives.certainty.satisfaction,
        affordances,
        vec![
            contribution("drive.certainty", uncertainty),
            contribution("world.epistemic.question_pressure", epistemic_pressure),
            contribution("self.stalled_goal", frustration),
        ],
    )
}

fn conductor_epistemic_affordance(source: &EpistemicAffordance) -> Affordance {
    let inspect_target = if source.affected_belief.0.contains("charger") {
        InspectTarget::Charger
    } else if source.affected_belief.0.contains("person") {
        InspectTarget::Person
    } else if source.affected_belief.0.contains("sound") {
        InspectTarget::Sound
    } else {
        InspectTarget::Novelty
    };
    let (action, skill) = match source.action_kind {
        EpistemicActionKind::OrientToBearing if source.bearing_rad.is_some() => (
            ActionPrimitive::Turn {
                direction: if source.bearing_rad.unwrap_or_default() >= 0.0 {
                    TurnDir::Left
                } else {
                    TurnDir::Right
                },
                intensity: 0.3,
                duration_ms: source.duration_ms,
            },
            Some(SkillId::TurnTowardTarget),
        ),
        EpistemicActionKind::SystematicSearch => (
            ActionPrimitive::Explore {
                style: ExploreStyle::WallFollow,
                duration_ms: source.duration_ms,
            },
            Some(SkillId::SystematicSearch),
        ),
        EpistemicActionKind::ScanClearance => (
            ActionPrimitive::Inspect {
                target: InspectTarget::Novelty,
            },
            Some(SkillId::InspectTarget),
        ),
        EpistemicActionKind::Listen => (
            ActionPrimitive::Inspect {
                target: InspectTarget::Sound,
            },
            Some(SkillId::InspectTarget),
        ),
        EpistemicActionKind::AskPerson => (
            ActionPrimitive::Speak {
                text: "Hello. What should I call you?".to_string(),
            },
            None,
        ),
        EpistemicActionKind::StopAndObserve | EpistemicActionKind::ComparePrediction => {
            (ActionPrimitive::Stop, Some(SkillId::StopAndStabilize))
        }
        EpistemicActionKind::InspectTarget
        | EpistemicActionKind::OrientToBearing
        | EpistemicActionKind::Unknown => (
            ActionPrimitive::Inspect {
                target: inspect_target,
            },
            Some(SkillId::InspectTarget),
        ),
    };
    let mut result = affordance(
        &source.behavior_id,
        action,
        source.confidence,
        source.expected_information_gain,
        source.expected_information_gain,
        source.risk,
        source.energy_cost,
        source.duration_ms,
        source.target.clone(),
        &[],
    )
    .with_bearing(source.bearing_rad)
    .with_epistemic(source);
    if let Some(skill) = skill {
        result = result.with_skill(skill, None);
    }
    result
}

fn evaluate_follow_task(
    interpretation: &GoalInterpretation,
    context: &GoalEvaluationContext<'_>,
) -> EvaluationParts {
    let affordances = interpretation
        .suggestions
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, action)| {
            let mut task_affordance = affordance(
                &format!("task_proposal_{index}"),
                action.clone(),
                context
                    .world
                    .context
                    .llm_confidence
                    .as_ref()
                    .map(|belief| belief.value)
                    .unwrap_or(0.5)
                    .max(0.5),
                0.5,
                0.5,
                interpretation.danger,
                0.1,
                1_000,
                None,
                &[],
            );
            if matches!(action, ActionPrimitive::Go { intensity, .. } if intensity < 0.0) {
                task_affordance = task_affordance.with_skill(SkillId::BackAway, None);
            }
            task_affordance
        })
        .collect::<Vec<_>>();
    let activation = if affordances.is_empty() { 0.0 } else { 0.45 };
    (
        activation,
        0.3,
        if affordances.is_empty() { 1.0 } else { 0.0 },
        affordances,
        vec![contribution(
            "proposal.count",
            interpretation.suggestions.len() as f32,
        )],
    )
}
