impl<L, M, R, C, S, A> MinimalRuntime<L, M, R, C, S, A>
where
    L: LedgerWriter + Sync,
    M: MemoryStore,
    R: Recall + Sync,
    C: Conductor,
    S: SafetyLayer,
    A: LlmAgent + 'static,
{
    pub async fn tick(
        &mut self,
        mut now: Now,
        _latent: ExperienceLatent,
        mut futures: Vec<FuturePrediction>,
    ) -> Result<RuntimeTick> {
        let frame_id = Uuid::new_v4();
        now.extensions.insert(
            "frame_id".to_string(),
            serde_json::Value::String(frame_id.to_string()),
        );
        apply_create_ir_charger_cue(&mut now);
        {
            let mut reign_queue = self
                .reign_queue
                .lock()
                .map_err(|_| anyhow::anyhow!("reign queue lock poisoned"))?;
            reign_queue.drain_expired(now.t_ms);
            now.reign = reign_queue.sense(now.t_ms);
        }
        let reign_input = now.reign.latest.clone();
        let reign_action = reign_input
            .as_ref()
            .and_then(|input| input.command.to_action());
        let mechanical_reign_action =
            mechanical_reign_action(&reign_input, self.action_selector_mode);

        let mut behavior_runs: Vec<ErasedBehaviorRunRecord> = Vec::new();
        let experience_input = ExperienceBehaviorInput::from_now(&now);
        let experience_run = self
            .models
            .behaviors
            .experience
            .infer(&experience_input, now.t_ms)?;
        let mut experience_record = experience_run.record;
        if let (Some(hard), Some(model)) = (
            experience_record.hardcoded_output.as_ref(),
            experience_record.model_output.as_ref(),
        ) {
            experience_record.disagreement = Some(experience_disagreement(hard, model));
        }
        if let Some(model_output) = experience_record.model_output.as_ref() {
            if let Some(loss) = model_output.reconstruction_loss {
                now.extensions.insert(
                    "experience.autoencoder".to_string(),
                    serde_json::json!({
                        "reconstruction_loss": loss,
                        "z_dim": model_output.latent.z.len(),
                    }),
                );
            }
        }
        behavior_runs.push(experience_record.erase());
        let latent = experience_run.chosen.latent.clone();

        let mut recall_query = RecallQuery::from_now(&now);
        let place_recognition_input =
            place_recognition_input_from_query_now(&now, Some(&latent), "runtime-pre-frame");
        recall_query.place_recognition_input = Some(place_recognition_input.clone());
        let loop_min_confidence = self.local_map.config.pose_graph_min_loop_confidence;
        let live_loop_candidates = self
            .memory_recall
            .loop_closure_candidates(&recall_query, loop_min_confidence, 10)
            .await?
            .iter()
            .map(|candidate| {
                place_candidate_to_loop_input(
                    candidate,
                    Some(frame_id.to_string()),
                    Some(&place_recognition_input),
                )
            })
            .collect::<Vec<_>>();
        let recall = self.memory_recall.recall(recall_query).await?;
        now.memory = recall.sense.clone();
        apply_recent_trap_memory_hints(&mut now);
        now.extensions.insert(
            "memory.place".to_string(),
            serde_json::json!({
                "danger": now.memory.place_danger,
                "charge": now.memory.place_charge_value,
                "social": now.memory.place_social_value,
                "novelty": now.memory.place_novelty,
                "confidence": now.memory.map_confidence,
                "places_visited": now.memory.places_visited,
                "nearby_best_charge_direction_rad": now.memory.nearby_best_charge_direction_rad,
                "nearby_best_safe_direction_rad": now.memory.nearby_best_safe_direction_rad,
                "nearby_frontier_direction_rad": now.memory.nearby_frontier_direction_rad,
                "recent_trap_direction_rad": now.memory.recent_trap_direction_rad,
                "recent_trap_confidence": now.memory.recent_trap_confidence,
            }),
        );
        if let Some(semantic_map) = &recall.semantic_map {
            now.extensions.insert(
                "memory.semantic_map".to_string(),
                serde_json::to_value(semantic_map)?,
            );
        }

        let mut surface_output_for_anticipation: Option<SurfaceExtractorOutput> = None;
        if !now.kinect.depth_m.is_empty()
            && now.kinect.depth_width > 0
            && now.kinect.depth_height > 0
        {
            let surface_output =
                self.surface_extractor
                    .process(&now.kinect, now.body.odometry, now.t_ms);
            surface_output_for_anticipation = Some(surface_output.clone());
            now.extensions.insert(
                "surface.scene_graph".to_string(),
                serde_json::json!({
                    "diagnostics": surface_output.diagnostics.clone(),
                    "floor": surface_output.floor.clone(),
                    "surfaces": surface_output.stable_surfaces.clone(),
                    "clusters": surface_output.clusters.clone(),
                    "navigation": surface_output.scene_graph.navigation.clone(),
                    "calibration_hint": surface_output.diagnostics.calibration_hint,
                    "obstacle_grid": {
                        "resolution_m": surface_output.obstacle_grid.resolution_m,
                        "half_extent_m": surface_output.obstacle_grid.half_extent_m,
                        "cells": surface_output.obstacle_grid.cells.clone(),
                    },
                }),
            );
        }

        let embodied_now = pete_experience::embody_now(&now).await?;
        let mut sensations = embodied_now.sensations;
        let mut impressions = embodied_now.impressions;
        if let Some(summary) = embodied_now.experience.summary_impression.clone() {
            impressions.push(summary);
        }
        let (direct_sensations, direct_impressions) = derive_direct_impressions_from_now(&now);
        sensations.extend(direct_sensations);
        impressions.extend(direct_impressions);
        let (recall_sensations, recall_impressions) =
            embodied_recall_sensations_and_impressions(&recall);
        let recall_sensation_ids = recall_sensations
            .iter()
            .map(|sensation| sensation.id)
            .collect::<Vec<_>>();
        let recall_impression_ids = recall_impressions
            .iter()
            .map(|impression| impression.id)
            .collect::<Vec<_>>();
        sensations.extend(recall_sensations);
        impressions.extend(recall_impressions);
        let mut experiences = derive_direct_experiences(&impressions, &sensations, now.t_ms);
        let mut embodied_experience = embodied_now.experience;
        embodied_experience
            .sensation_ids
            .extend(recall_sensation_ids);
        embodied_experience
            .impression_ids
            .extend(recall_impression_ids);
        if futures.is_empty() {
            let (predicted, records) =
                predict_baseline_futures(&mut self.models.behaviors.future, &latent, now.t_ms)?;
            futures = predicted;
            behavior_runs.extend(records);
        }
        let mut teachings = Vec::new();
        let mut notes = Vec::new();
        let mut drive_impulses = DriveSense::default();
        if let Some(stuck_values) = now
            .extensions
            .get("sim.stuck")
            .and_then(|value| value.get("values"))
            .and_then(|value| value.as_array())
        {
            let active = stuck_values
                .first()
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                > 0.0;
            let corner = stuck_values
                .get(1)
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                > 0.0;
            let duration_ms = stuck_values
                .get(3)
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0);
            let phase = stuck_values
                .get(4)
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0);
            let started = stuck_values
                .get(6)
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                > 0.0;
            let recovered = stuck_values
                .get(7)
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                > 0.0;
            let dead_battery = stuck_values
                .get(8)
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                > 0.0;
            if dead_battery {
                notes.push(
                    "VirtualDeadBattery: battery reached 0%; virtual motion stopped".to_string(),
                );
            }
            if started {
                notes.push("StuckDetected: classified as stuck/corner-trap".to_string());
            }
            if active {
                notes.push(format!(
                    "StuckRecovery: class={}, phase={}, duration_ms={duration_ms:.0}",
                    if corner { "corner-trap" } else { "stuck" },
                    stuck_phase_label(phase),
                ));
            }
            if recovered {
                notes.push("StuckRecovery: recovered and resumed exploration".to_string());
            }
        }
        if now
            .extensions
            .get("safety/read_only_veto")
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
        {
            notes.push("source = real_robot_read_only".to_string());
            notes.push("mode = read_only".to_string());
            notes.push("motor_applied = false".to_string());
            notes.push("ReadOnlyActionSuppressed: motion suppressed by read-only mode".to_string());
        }
        let map_observation = observation_from_now(&now, self.local_map.config);
        let map_summary = self
            .local_map
            .integrate_observation_with_loop_candidates(map_observation, &live_loop_candidates);
        now.extensions.insert(
            MAP_EXTENSION_NAME.to_string(),
            serde_json::to_value(&map_summary)?,
        );
        notes.push(format!(
            "ScanMatchedMap: {} cells ({} occupied, {} free); occupancy scan matching corrects odometry before integration",
            map_summary.cells, map_summary.occupied_cells, map_summary.free_cells
        ));
        let corrected_map_trust = corrected_map_trust_status(&now);
        if !corrected_map_trust.trusted {
            notes.push(format!(
                "MapTrustGate: navigation will not trust spatial memory until corrected SLAM is ready ({})",
                corrected_map_trust
                    .reason
                    .as_deref()
                    .unwrap_or("corrected map is not trusted")
            ));
        }
        let mut proposed_actions = Vec::new();

        let events = self.extractor.events_from_now(&now, Some(&recall));
        let ctx = EventContext {
            now: &now,
            latent: Some(&latent),
            recall: Some(&recall),
            predicted_futures: &futures,
            llm: Some(&now.llm),
            safety: None,
        };
        let event_output = self.bus.dispatch_all(&ctx, events)?;
        apply_responses(
            &mut now,
            event_output.responses,
            &mut sensations,
            &mut impressions,
            &mut experiences,
            &mut teachings,
            &mut notes,
            &mut drive_impulses,
        );
        let (event_script_forced_action, event_script_records) =
            self.run_event_scripts(&mut now, &mut notes, &mut proposed_actions)?;
        behavior_runs.extend(event_script_records);
        self.chirp_events
            .emit_pre_selection_chirps(&mut now, &mut notes)?;
        let runtime_instant = ExperienceInstant::from_parts(
            Some(&embodied_experience),
            &sensations,
            &impressions,
            &futures,
            &recall.recollections,
            &now,
            None,
            None,
            "runtime-live",
        );
        let embodied_context = runtime_instant.embodied_context();

        let accepted_llm = self
            .advance_cognition(
                &now,
                &impressions,
                &embodied_context,
                &latent,
                &futures,
                &recall.first_person_summary,
                &mut notes,
            )
            .await;
        now.llm = self.cognition.last_sense.clone();
        if let Some(accepted) = accepted_llm.as_ref() {
            if let Some(reflection) = accepted.reflection.as_ref() {
                append_combobulation(
                    &mut sensations,
                    &mut impressions,
                    &mut experiences,
                    accepted.requested_at_ms,
                    accepted.observed_at_ms,
                    &accepted.snapshot_ref,
                    reflection,
                );
            }
            apply_llm_tick(
                &accepted.tick,
                accepted.requested_at_ms,
                accepted.observed_at_ms,
                &accepted.snapshot_ref,
                &mut sensations,
                &mut impressions,
                &mut experiences,
                &mut teachings,
            );
        }
        // Higher cognition is advisory. Even a valid response cannot become a
        // Cockpit proposal; local goals, skills, Reign, and safety own motion.
        let combobulation = accepted_llm
            .as_ref()
            .and_then(|accepted| accepted.reflection.clone());
        let llm_advisory_action = accepted_llm.as_ref().and_then(|accepted| {
            accepted
                .tick
                .decision
                .as_ref()
                .and_then(|decision| decision.action.clone())
                .map(|action| LlmAdvisoryAction {
                    action,
                    source: LlmAdvisoryActionSource::ProviderDecision,
                    input_snapshot_ref: accepted.snapshot_ref.clone(),
                    disposition: LlmAdvisoryActionDisposition::DiscardedAtAdvisoryBoundary,
                })
        });
        let llm_tick = accepted_llm
            .map(|accepted| accepted.tick)
            .unwrap_or_default();
        let llm_command_action = None;
        let mut llm_action_proposal = LlmActionProposal {
            proposed_action: llm_command_action.clone(),
            advisory_action: llm_advisory_action.clone(),
            ignored_reason: llm_advisory_action.as_ref().map(|advisory| {
                format!(
                    "provider suggested {:?}; discarded at advisory boundary",
                    advisory.action
                )
            }),
            ..LlmActionProposal::default()
        };
        if let Some(advisory) = llm_advisory_action.as_ref() {
            notes.push(format!(
                "LlmAdvisoryAction: provider suggested {:?}; discarded at advisory boundary (input_snapshot_ref={})",
                advisory.action, advisory.input_snapshot_ref
            ));
        }
        let llm_has_safety_reason = crate::llm_explicit_safety_reason(&llm_tick);
        let direct_reign_active = reign_input
            .as_ref()
            .is_some_and(|input| input.mode == pete_actions::ReignMode::Direct);
        let mechanical_reign_action_for_selection =
            if mechanical_reign_action.is_some() && llm_has_safety_reason && !direct_reign_active {
                notes.push(
                    "LlmActionProposal: explicit safety reason allowed competition with Reign"
                        .to_string(),
                );
                None
            } else {
                mechanical_reign_action.clone()
            };

        let nudge_proposal = self.nudge.propose(&now, self.nudge_policy);
        if let Some(action) = nudge_proposal.clone() {
            notes.push(format!("ProdNudge: proposed {:?}", action));
            proposed_actions.push(action);
        } else if let Some(reason) = self.nudge.status.nudge_blocked_reason.clone() {
            notes.push(format!("ProdNudgeBlocked: {reason}"));
        }

        let mut proposals = proposed_actions.clone();
        if self.action_selector_mode == ActionSelectorMode::Goal {
            if let Some(reign_action) = now
                .reign
                .latest
                .as_ref()
                .filter(|input| matches!(input.mode, ReignMode::Assist | ReignMode::Suggest))
                .and_then(|input| input.command.to_action())
            {
                // In goal mode Assist/Suggest is consumed as an affordance-matched
                // bias by GoalSystem, not reintroduced as a generic task proposal.
                proposals.retain(|proposal| proposal != &reign_action);
            }
        }
        if let Some(action) = llm_command_action.clone() {
            notes.push(format!("LlmActionProposal: proposed {:?}", action));
            proposals.push(action);
        }
        if let Some(action) = event_script_forced_action.clone() {
            push_unique_action(&mut proposals, action);
        }
        self.goal_system
            .add_drive_impulses(std::mem::take(&mut drive_impulses));
        self.goal_system.seed_drives(now.drives.clone());
        let mut world_context = self.goal_system.world_model_update_context();
        if let Some(experience_id) = runtime_instant.experience_id.as_ref() {
            world_context
                .continuity
                .recent_experience_refs
                .push(format!("{experience_id:?}"));
        }
        if let Some(previous_control) = self.last_active_control.as_ref() {
            if let Some(action) = previous_control.action_kind.as_ref() {
                world_context
                    .continuity
                    .recent_self_action_refs
                    .push(action.clone());
            }
            world_context
                .continuity
                .recent_outcome_refs
                .extend(previous_control.veto_reasons.iter().cloned());
        }
        world_context.active_control = self.last_active_control.clone();
        world_context.semantic_observations = self.semantic_outcomes.take_pending();
        let cognition_busy = self.cognition.pending.is_some();
        let cognition_failure = self
            .cognition
            .last_outcome
            .as_ref()
            .and_then(|outcome| match outcome {
                CognitionOutcome::Failed(error) => Some(error.clone()),
                CognitionOutcome::Expired => Some("latest request expired".to_string()),
                CognitionOutcome::Cancelled => Some("latest request was cancelled".to_string()),
                _ => None,
            })
            .or_else(|| {
                (!self.cognition.provider_declared_available)
                    .then(|| self.cognition.provider_unavailable_reason.clone())
                    .flatten()
            });
        // Occupancy and health are separate: a pending task means the healthy
        // service is busy, not unavailable. The post-request cooldown is idle,
        // healthy time rather than either an outage or request occupancy.
        let enhanced_cognition_available =
            self.cognition.provider_declared_available && cognition_failure.is_none();
        world_context.cognitive_services.insert(
            "rich_language".to_string(),
            CognitiveServiceBelief {
                available: enhanced_cognition_available,
                busy: cognition_busy,
                confidence: 1.0,
                unavailable_reason: cognition_failure,
                meta: BeliefMeta {
                    confidence: 1.0,
                    observed_at_ms: now.t_ms,
                    valid_at_ms: now.t_ms,
                    freshness: Freshness::Current,
                    source_kind: BeliefSourceKind::Map,
                    ..BeliefMeta::default()
                },
                ..CognitiveServiceBelief::default()
            },
        );
        now = self.world_model.update(now, world_context);
        self.semantic_outcomes.observe_outcome(&now.world);
        now.extensions.insert(
            "self_model".to_string(),
            serde_json::to_value(&now.world.self_model)?,
        );
        now.extensions.insert(
            "temporal_context".to_string(),
            serde_json::to_value(&now.world.temporal)?,
        );
        now.extensions.insert(
            "social_world".to_string(),
            serde_json::to_value(&now.world.social)?,
        );
        now.extensions.insert(
            "epistemic_state".to_string(),
            serde_json::to_value(&now.world.epistemic)?,
        );
        now.extensions.insert(
            "semantic_graph".to_string(),
            serde_json::to_value(&now.world.semantic)?,
        );
        let sleep_input = runtime_sleep_input(
            &now,
            self.sleep_controller.expects_external_power(),
            enhanced_cognition_available,
        );
        let sleep_snapshot = self.sleep_controller.tick(sleep_input);
        let sleeping = self.sleep_controller.requires_quiescence();
        now.extensions
            .insert("sleep".to_string(), serde_json::to_value(&sleep_snapshot)?);
        let goal_cycle = if sleeping {
            self.goal_system.suspend_for_sleep(&now.world)
        } else {
            self.goal_system.tick(&now.world, &proposals)?
        };
        now.drives = goal_cycle.drives.legacy_sense();
        let goal_action = goal_cycle
            .behavior
            .as_ref()
            .map(|behavior| behavior.action.clone());
        let mut goal_skill_request = goal_cycle
            .behavior
            .as_ref()
            .and_then(|behavior| behavior.affordance.skill_request.clone());
        if goal_cycle
            .selection
            .selected_goal
            .as_ref()
            .is_some_and(|goal| goal.as_str() == "seek_charger")
        {
            if let (Some(request), Some(cue)) = (
                goal_skill_request.as_mut(),
                DockIrCue::from_character(now.body.infrared_character),
            ) {
                if matches!(
                    request.skill_id,
                    SkillId::TurnTowardTarget | SkillId::ApproachTarget | SkillId::AlignWithDock
                ) {
                    request.skill_id = SkillId::AlignWithDock;
                    request.bearing_rad = Some(cue.bearing_hint_rad());
                    request.progress_metric = "dock_ir_alignment".to_string();
                }
            }
        }
        now.extensions.insert(
            "goal_system".to_string(),
            serde_json::to_value(&goal_cycle)?,
        );
        let mut action_value_candidates =
            action_value_candidate_actions(&proposals, reign_action.as_ref(), &llm_tick);

        let conductor_memory =
            memory_for_navigation_with_map_trust(now.memory.clone(), corrected_map_trust);
        let mut baseline_action = self.conductor.choose(ConductorInput {
            latent: latent.clone(),
            drives: now.drives.clone(),
            memory: conductor_memory,
            predictions: now.predictions.clone(),
            surprise: now.surprise.clone(),
            llm: now.llm.clone(),
            safety: SafetySense::default(),
            reign: now.reign.clone(),
            range: now.range.clone(),
            body: now.body.clone(),
            charger_near_score: charger_signal_scores(&now).0,
            charger_visible_score: charger_signal_scores(&now).1,
            proposals: proposals.clone(),
        })?;
        let conductor_navigation_goal = Box::new(
            self.conductor
                .navigation_goal()
                .cloned()
                .unwrap_or_else(|| pete_conductor::NavigationGoalDecision {
                    intent: NavigationIntent::FollowProposal,
                    action: baseline_action.clone(),
                    confidence: 0.5,
                    reason:
                        "conductor selected an action without structured navigation diagnostics"
                            .to_string(),
                }),
        );
        now.extensions.insert(
            "conductor.navigation_goal".to_string(),
            serde_json::to_value(conductor_navigation_goal.as_ref())?,
        );
        if let Some(action) = mechanical_reign_action_for_selection.as_ref() {
            baseline_action = action.clone();
        }
        if recovery_candidate_context(&now) && is_recovery_locomotion_action(&baseline_action) {
            push_unique_action(&mut action_value_candidates, baseline_action.clone());
        }
        if memory_navigation_candidate_context(&now, &baseline_action) {
            push_unique_action(&mut action_value_candidates, baseline_action.clone());
        }

        let mut model_predictions = Vec::new();
        let mut hardcoded_predictions = Vec::new();
        let mut candidate_scores = Vec::new();
        for action in &action_value_candidates {
            let candidate_now = now_with_surface_anticipation(
                &now,
                surface_output_for_anticipation.as_ref(),
                action,
            );
            let candidate_danger_input =
                danger_behavior_input(&candidate_now, &latent, Some(action));
            let candidate_danger = self
                .models
                .behaviors
                .danger
                .infer(&candidate_danger_input, now.t_ms)?;
            let mut candidate_danger_record = candidate_danger.record;
            if let (Some(hard), Some(model)) = (
                candidate_danger_record.hardcoded_output.as_ref(),
                candidate_danger_record.model_output.as_ref(),
            ) {
                candidate_danger_record.disagreement = Some(danger_disagreement(hard, model));
            }
            let candidate_danger_output = candidate_danger_record
                .model_output
                .as_ref()
                .copied()
                .or(candidate_danger_record.selected_output.as_ref().copied());
            let candidate_danger_had_fallback = candidate_danger_record.model_output.is_none();
            behavior_runs.push(candidate_danger_record.erase());

            let candidate_charge_input = charge_behavior_input(&now, &latent, Some(action));
            let candidate_charge = self
                .models
                .behaviors
                .charge
                .infer(&candidate_charge_input, now.t_ms)?;
            let mut candidate_charge_record = candidate_charge.record;
            if let (Some(hard), Some(model)) = (
                candidate_charge_record.hardcoded_output.as_ref(),
                candidate_charge_record.model_output.as_ref(),
            ) {
                candidate_charge_record.disagreement = Some(charge_disagreement(hard, model));
            }
            let candidate_charge_output = candidate_charge_record
                .model_output
                .as_ref()
                .copied()
                .or(candidate_charge_record.selected_output.as_ref().copied());
            let candidate_charge_had_fallback = candidate_charge_record.model_output.is_none();
            behavior_runs.push(candidate_charge_record.erase());

            let candidate_action_value_input = action_value_behavior_input(
                &candidate_now,
                &latent,
                Some(action),
                candidate_danger_output,
                candidate_charge_output,
            );
            let action_value_run = self
                .models
                .behaviors
                .action_value
                .infer(&candidate_action_value_input, now.t_ms)?;
            let mut action_value_record = action_value_run.record;
            if let (Some(hard), Some(model)) = (
                action_value_record.hardcoded_output.as_ref(),
                action_value_record.model_output.as_ref(),
            ) {
                action_value_record.disagreement = Some(action_value_disagreement(hard, model));
            }
            if let Some(model) = action_value_record.model_output.as_ref() {
                model_predictions.push(action_value_prediction(action.clone(), *model));
            }
            if let Some(hardcoded) = action_value_record.hardcoded_output.as_ref() {
                hardcoded_predictions.push(action_value_prediction(action.clone(), *hardcoded));
            }
            let action_value_output = action_value_record
                .model_output
                .as_ref()
                .copied()
                .or(action_value_record.selected_output.as_ref().copied());
            let action_value_had_fallback = action_value_record.model_output.is_none();
            behavior_runs.push(action_value_record.erase());

            let mut candidate_score = score_action_candidate(
                &candidate_now,
                action,
                CandidateModelSignals {
                    danger: candidate_danger_output,
                    charge: candidate_charge_output,
                    action_value: action_value_output,
                },
                Some(&baseline_action),
            );
            candidate_score.fallback_used = candidate_score.fallback_used
                || candidate_danger_had_fallback
                || candidate_charge_had_fallback
                || action_value_had_fallback;
            candidate_scores.push(candidate_score);
        }
        now.predictions.action_values_model = model_predictions;
        now.predictions.action_values_hardcoded = hardcoded_predictions;

        let mut action_selection = select_action_from_scores(
            self.action_selector_mode,
            &now,
            baseline_action.clone(),
            candidate_scores,
        );
        action_selection.shadow_selected_goal = goal_cycle
            .selection
            .selected_goal
            .as_ref()
            .map(|goal| goal.as_str().to_string());
        action_selection.shadow_selected_behavior = goal_cycle
            .behavior
            .as_ref()
            .map(|behavior| behavior.behavior_id.clone());
        action_selection.shadow_goal_action = goal_action.clone();
        action_selection.shadow_diverged_from_baseline = goal_action
            .as_ref()
            .is_some_and(|action| action != &baseline_action);
        action_selection.goal_switched = goal_cycle.selection.switched;
        action_selection.goal_retained_by_commitment = goal_cycle.selection.retained_by_commitment;
        action_selection.goal_selection_reason = Some(goal_cycle.selection.reason.clone());
        if self.action_selector_mode == ActionSelectorMode::Goal {
            action_selection.selected_goal = action_selection.shadow_selected_goal.clone();
            action_selection.selected_behavior = goal_cycle
                .behavior
                .as_ref()
                .map(|behavior| behavior.behavior_id.clone());
            action_selection.selected_action = goal_action.clone();
            action_selection.selected_score = goal_cycle
                .selection
                .selected_goal
                .as_ref()
                .and_then(|selected| {
                    goal_cycle
                        .evaluations
                        .iter()
                        .find(|evaluation| &evaluation.goal_id == selected)
                })
                .map(|evaluation| evaluation.motivation.activation);
            action_selection.safety_overrode = false;
        }
        if let Some(action) = mechanical_reign_action_for_selection.as_ref() {
            action_selection.selected_action = Some(action.clone());
            action_selection.selected_score = None;
            action_selection.safety_overrode = false;
        } else if self.action_selector_mode != ActionSelectorMode::Goal {
            if let Some(action) = event_script_forced_action.as_ref() {
                action_selection.selected_action = Some(action.clone());
                action_selection.selected_score = None;
                action_selection.safety_overrode = false;
            }
        }
        for warning in &action_selection.fallback_warnings {
            notes.push(warning.clone());
        }
        now.extensions.insert(
            "action_selector".to_string(),
            serde_json::to_value(&action_selection)?,
        );
        let teacher_action = action_selection
            .selected_action
            .clone()
            .unwrap_or(baseline_action);
        let mut conductor_proposals = action_value_candidates.clone();
        conductor_proposals.push(teacher_action.clone());
        let conductor_behavior_input = ConductorInput {
            latent: latent.clone(),
            drives: now.drives.clone(),
            memory: now.memory.clone(),
            predictions: now.predictions.clone(),
            surprise: now.surprise.clone(),
            llm: now.llm.clone(),
            safety: SafetySense::default(),
            reign: now.reign.clone(),
            range: now.range.clone(),
            body: now.body.clone(),
            charger_near_score: charger_signal_scores(&now).0,
            charger_visible_score: charger_signal_scores(&now).1,
            proposals: conductor_proposals,
        };
        let teacher_source = if now.reign.active {
            TrainingSource::HumanReign
        } else {
            TrainingSource::HardcodedTeacher
        };
        let conductor_run = self.models.behaviors.conductor.infer_with_teacher_source(
            &conductor_behavior_input,
            now.t_ms,
            teacher_source,
        )?;
        let mut conductor_record = conductor_run.record;
        let conductor_controls = matches!(
            self.models.behaviors.conductor.regime,
            BehaviorRegime::ModelInfer | BehaviorRegime::ModelTrainAndInfer
        ) && self.action_selector_mode != ActionSelectorMode::Goal;
        let conductor_selected_action = if conductor_controls {
            conductor_run.chosen
        } else {
            teacher_action.clone()
        };
        conductor_record.selected_output = Some(conductor_selected_action.clone());
        if mechanical_reign_action_for_selection.is_some()
            && !matches!(conductor_record.selected_output, Some(ref action) if Some(action) == mechanical_reign_action_for_selection.as_ref())
        {
            conductor_record.selected_output = mechanical_reign_action_for_selection.clone();
        }
        let mut chosen_action = mechanical_reign_action_for_selection
            .clone()
            .or_else(|| {
                (self.action_selector_mode != ActionSelectorMode::Goal)
                    .then(|| event_script_forced_action.clone())
                    .flatten()
            })
            .unwrap_or(conductor_selected_action);
        if sleeping && mechanical_reign_action_for_selection.is_none() {
            chosen_action = ActionPrimitive::Stop;
        }
        let locomotion_input = self
            .locomotion_tracker
            .observe(now.t_ms, &now.body, &now.range);
        let locomotion_run = self
            .models
            .behaviors
            .locomotion
            .infer_with_disagreement(&locomotion_input, now.t_ms)?;
        let locomotion_shadow = if locomotion_run.record.regime == BehaviorRegime::ShadowInfer {
            locomotion_run
                .record
                .hardcoded_output
                .map(|baseline| {
                    LocomotionShadowFrame::new(
                        frame_id.to_string(),
                        now.t_ms,
                        locomotion_input.clone(),
                        baseline,
                        locomotion_run.record.model_output,
                        locomotion_run.chosen,
                        self.models.behaviors.locomotion.hardcoded_id(),
                        self.models
                            .behaviors
                            .locomotion
                            .model_id()
                            .unwrap_or("locomotion.model.missing"),
                        locomotion_run.record.hardcoded_inference_us,
                        locomotion_run.record.model_inference_us,
                        locomotion_run.record.confidence.or_else(|| {
                            locomotion_run
                                .record
                                .disagreement
                                .map(|distance| 1.0 / (1.0 + distance.max(0.0)))
                        }),
                        locomotion_run.record.error.clone(),
                    )
                })
        } else {
            None
        };
        let locomotion_output = locomotion_run.chosen.bounded(0.6, 1.0);
        let locomotion_applied = mechanical_reign_action_for_selection.is_none()
            && (event_script_forced_action.is_none()
                || self.action_selector_mode == ActionSelectorMode::Goal)
            && matches!(&chosen_action, ActionPrimitive::Explore { .. });
        if locomotion_applied {
            let duration_ms = match &chosen_action {
                ActionPrimitive::Explore { duration_ms, .. } => *duration_ms,
                _ => 1_000,
            };
            chosen_action = ActionPrimitive::Drive {
                forward: locomotion_output.forward_velocity_m_s,
                turn: locomotion_output.angular_velocity_rad_s,
                duration_ms,
            };
        }
        now.extensions.insert(
            "locomotion.nervous_system".to_string(),
            serde_json::json!({
                "schema_version": pete_neat::LOCOMOTION_SCHEMA_VERSION,
                "input": locomotion_input,
                "output": locomotion_output,
                "applied": locomotion_applied,
                "safety_authority": false,
                "shadow": locomotion_shadow,
            }),
        );
        behavior_runs.push(locomotion_run.record.erase());
        self.chirp_events
            .emit_post_selection_chirps(&mut now, &mut notes, &chosen_action)?;
        let mut map_memory_decision = map_memory_decision_debug(
            &now,
            &chosen_action,
            action_selection.baseline_action.as_ref(),
            mechanical_reign_action_for_selection.is_some() || event_script_forced_action.is_some(),
        );
        now = now_with_surface_anticipation(
            &now,
            surface_output_for_anticipation.as_ref(),
            &chosen_action,
        );
        let conductor_selected_output = conductor_record.selected_output.clone();
        behavior_runs.push(conductor_record.erase());

        if let Some(proposed) = llm_action_proposal.proposed_action.as_ref() {
            llm_action_proposal.accepted = proposed == &chosen_action;
            llm_action_proposal.final_action = Some(chosen_action.clone());
            if !llm_action_proposal.accepted && llm_action_proposal.ignored_reason.is_none() {
                llm_action_proposal.ignored_reason = if mechanical_reign_action_for_selection
                    .is_some()
                {
                    Some("safe active Reign command outranked LLM action".to_string())
                } else if mechanical_reign_action.is_some() && llm_has_safety_reason {
                    Some("LLM safety rationale competed with Reign but conductor selected another action".to_string())
                } else if event_script_forced_action.is_some() {
                    Some("event script action outranked LLM action".to_string())
                } else {
                    Some("conductor selected a different action".to_string())
                };
            }
        }

        let danger_input = danger_behavior_input(&now, &latent, Some(&chosen_action));
        let danger_run = self
            .models
            .behaviors
            .danger
            .infer(&danger_input, now.t_ms)?;
        let mut danger_record = danger_run.record;
        if let (Some(hard), Some(model)) = (
            danger_record.hardcoded_output.as_ref(),
            danger_record.model_output.as_ref(),
        ) {
            danger_record.disagreement = Some(danger_disagreement(hard, model));
        }
        if let Some(model) = danger_record.model_output.as_ref() {
            now.predictions.danger_model = Some(danger_prediction(*model));
        }
        if let Some(hardcoded) = danger_record.hardcoded_output.as_ref() {
            now.predictions.danger_hardcoded = Some(danger_prediction(*hardcoded));
        }
        behavior_runs.push(danger_record.erase());

        let charge_input = charge_behavior_input(&now, &latent, Some(&chosen_action));
        let charge_run = self
            .models
            .behaviors
            .charge
            .infer(&charge_input, now.t_ms)?;
        let mut charge_record = charge_run.record;
        if let (Some(hard), Some(model)) = (
            charge_record.hardcoded_output.as_ref(),
            charge_record.model_output.as_ref(),
        ) {
            charge_record.disagreement = Some(charge_disagreement(hard, model));
        }
        if let Some(model) = charge_record.model_output.as_ref() {
            now.predictions.charge_model = Some(charge_prediction(*model));
        }
        if let Some(hardcoded) = charge_record.hardcoded_output.as_ref() {
            now.predictions.charge_hardcoded = Some(charge_prediction(*hardcoded));
        }
        behavior_runs.push(charge_record.erase());

        let eye_next_input = eye_next_behavior_input(&now, &latent, Some(&chosen_action), 100);
        let eye_next_run = self
            .models
            .behaviors
            .eye_next
            .infer(&eye_next_input, now.t_ms)?;
        let mut eye_next_record = eye_next_run.record;
        if let (Some(hard), Some(model)) = (
            eye_next_record.hardcoded_output.as_ref(),
            eye_next_record.model_output.as_ref(),
        ) {
            eye_next_record.disagreement = Some(eye_next_disagreement(hard, model));
        }
        if let Some(model) = eye_next_record.model_output.as_ref() {
            now.predictions.eye_next_model = Some(eye_prediction(model));
        }
        if let Some(hardcoded) = eye_next_record.hardcoded_output.as_ref() {
            now.predictions.eye_next_hardcoded = Some(eye_prediction(hardcoded));
        }
        behavior_runs.push(eye_next_record.erase());

        let ear_next_input = ear_next_behavior_input(&now, &latent, Some(&chosen_action), 100);
        let ear_next_run = self
            .models
            .behaviors
            .ear_next
            .infer(&ear_next_input, now.t_ms)?;
        let mut ear_next_record = ear_next_run.record;
        if let (Some(hard), Some(model)) = (
            ear_next_record.hardcoded_output.as_ref(),
            ear_next_record.model_output.as_ref(),
        ) {
            ear_next_record.disagreement = Some(ear_next_disagreement(hard, model));
        }
        if let Some(model) = ear_next_record.model_output.as_ref() {
            now.predictions.ear_next_model = Some(ear_prediction(model));
        }
        if let Some(hardcoded) = ear_next_record.hardcoded_output.as_ref() {
            now.predictions.ear_next_hardcoded = Some(ear_prediction(hardcoded));
        }
        behavior_runs.push(ear_next_record.erase());

        let selected_goal_for_safety = (self.action_selector_mode == ActionSelectorMode::Goal
            && mechanical_reign_action_for_selection.is_none())
        .then(|| goal_cycle.selection.selected_goal.as_ref())
        .flatten()
        .map(|goal| goal.as_str());
        let desired_motor = action_to_motor_command(Some(&chosen_action));
        let safety = self.safety.filter_action(
            &now,
            selected_goal_for_safety,
            &chosen_action,
            desired_motor,
        );
        let control_provenance = if safety.vetoed {
            ControlProvenance::SafetyVeto
        } else if safety.reason == Some(SafetyReason::Contact) {
            ControlProvenance::AutonomicReflex
        } else if direct_reign_active {
            ControlProvenance::HumanDirect
        } else if now
            .reign
            .latest
            .as_ref()
            .is_some_and(|input| input.mode == ReignMode::Assist)
        {
            ControlProvenance::HumanAssist
        } else {
            ControlProvenance::Autonomous
        };
        let executed_goal_behavior = goal_cycle.behavior.as_ref().filter(|behavior| {
            self.action_selector_mode == ActionSelectorMode::Goal
                && !sleeping
                && !locomotion_applied
                && control_provenance == ControlProvenance::Autonomous
                && behavior.action == chosen_action
                && !safety.vetoed
                && safety.reason.is_none()
                && safety.command == desired_motor
        });
        self.last_active_control = Some(ActiveControlSummary {
            goal_id: executed_goal_behavior.map(|behavior| behavior.goal_id.as_str().to_string()),
            behavior_id: executed_goal_behavior.map(|behavior| behavior.behavior_id.clone()),
            action_kind: Some(format!("{chosen_action:?}")),
            provenance: control_provenance,
            safety_preempted: safety.reason.is_some(),
            veto_reasons: safety
                .reason
                .as_ref()
                .map(|reason| vec![describe_safety_reason(Some(reason.clone())).to_string()])
                .unwrap_or_default(),
            unable_to_act_reason: safety
                .vetoed
                .then(|| describe_safety_reason(safety.reason.clone()).to_string()),
            ..ActiveControlSummary::default()
        });
        action_selection.safety_overrode = safety.vetoed;
        now.extensions.insert(
            "action_selector".to_string(),
            serde_json::to_value(&action_selection)?,
        );
        now.extensions.insert(
            "goal_system.outcome".to_string(),
            serde_json::json!({
                "schema_version": 1,
                "world_revision": goal_cycle.world.revision,
                "selected_goal": goal_cycle.selection.selected_goal.clone(),
                "selected_behavior": goal_cycle.behavior.as_ref().map(|behavior| &behavior.behavior_id),
                "executed_goal_behavior": executed_goal_behavior.map(|behavior| &behavior.behavior_id),
                "selected_primitive": chosen_action.clone(),
                "safety": {
                    "vetoed": safety.vetoed,
                    "reason": safety
                        .reason
                        .clone()
                        .map(|reason| describe_safety_reason(Some(reason))),
                    "final_motor": safety.command,
                },
                "shadow_diverged_from_baseline": action_selection.shadow_diverged_from_baseline,
            }),
        );
        self.locomotion_tracker.observe_command(LocomotionOutput {
            forward_velocity_m_s: safety.command.forward,
            angular_velocity_rad_s: safety.command.turn,
            recovery_activation: locomotion_output.recovery_activation,
        });
        map_memory_decision.safety_overrode = safety.vetoed;
        self.nudge.observe_motor(safety.command);
        now.extensions.insert(
            "motor_gate".to_string(),
            serde_json::json!({
                "desired_motor": desired_motor,
                "final_motor": safety.command,
                "motor_applied": !is_near_zero_motor(safety.command),
                "vetoed": safety.vetoed,
                "safety_reason": safety.reason.clone().map(Some).map(describe_safety_reason),
            }),
        );
        if locomotion_shadow.is_some() {
            now.extensions.insert(
                "locomotion.shadow.safety_chain".to_string(),
                serde_json::json!({
                    "schema_version": 1,
                    "candidate_is_proposal_only": true,
                    "baseline_selected_in_shadow": true,
                    "conductor_gate_executed": true,
                    "autonomic_gate_executed": true,
                    "final_motor_gate_executed": true,
                    "locomotion_proposal_applied": locomotion_applied,
                    "desired_motor": desired_motor,
                    "final_motor": safety.command,
                    "safety_vetoed": safety.vetoed,
                    "required_external_authorities": ["possession_lease", "brainstem"],
                    "candidate_direct_motion_authority": false,
                }),
            );
        }
        now.extensions.insert(
            "action.motion_bridge".to_string(),
            serde_json::json!({
                "llm_action": llm_action_proposal.proposed_action.clone(),
                "llm_advisory_action": llm_action_proposal.advisory_action.clone(),
                "selected_action": action_selection.selected_action.clone(),
                "conductor_selected_action": conductor_selected_output.clone(),
                "conductor_navigation_goal": conductor_navigation_goal.as_ref(),
                "chosen_action": chosen_action.clone(),
                "map_memory_decision": map_memory_decision.clone(),
                "desired_motor": action_to_motor_command(Some(&chosen_action)),
                "final_motor": safety.command,
                "safety_override": safety.vetoed,
                "safety_reason": safety.reason.clone().map(Some).map(describe_safety_reason),
            }),
        );
        if map_memory_decision.influenced {
            notes.push(format!(
                "MapMemoryDecision reason={:?} action={:?} danger={:.2} charge={:.2} novelty={:.2}",
                map_memory_decision.reason,
                map_memory_decision.selected_action,
                map_memory_decision.place_danger,
                map_memory_decision.place_charge_value,
                map_memory_decision.place_novelty
            ));
        }
        notes.push(format!(
            "ActionMotorBridge llm_action={:?} llm_advisory_action={:?} selected_action={:?} conductor_selected_action={:?} chosen_action={:?} desired_motor={:?} final_motor={:?} safety_override={}",
            llm_action_proposal.proposed_action,
            llm_action_proposal.advisory_action,
            action_selection.selected_action,
            conductor_selected_output,
            chosen_action,
            action_to_motor_command(Some(&chosen_action)),
            safety.command,
            safety.vetoed
        ));
        now.extensions.insert(
            "prod.nudge".to_string(),
            serde_json::to_value(&self.nudge.status)?,
        );
        if safety.vetoed {
            if llm_action_proposal.accepted {
                let reason = describe_safety_reason(safety.reason.clone()).to_string();
                llm_action_proposal.safety_vetoed = true;
                llm_action_proposal.safety_reason = Some(reason.clone());
                notes.push(format!("LlmActionProposalSafetyVeto: {reason}"));
                teachings.push(pete_llm::LlmTeaching {
                    t_ms: now.t_ms,
                    summary: format!("Safety vetoed LLM action {:?}", chosen_action),
                    critique: Some(format!("LLM proposed an unsafe action: {reason}")),
                    counterfactuals: Vec::new(),
                    memory_notes: vec![format!(
                        "Avoid repeating LLM action {:?} when safety reports {reason}",
                        chosen_action
                    )],
                    confidence: now.llm.confidence,
                });
            }
            now.extensions
                .insert("safety.vetoed".to_string(), serde_json::Value::Bool(true));
            let veto_ctx = EventContext {
                now: &now,
                latent: Some(&latent),
                recall: Some(&recall),
                predicted_futures: &futures,
                llm: Some(&now.llm),
                safety: Some(&safety),
            };
            let veto_events = self
                .extractor
                .events_from_safety(&now, &chosen_action, &safety);
            let veto_output = self.bus.dispatch_all(&veto_ctx, veto_events)?;
            apply_responses(
                &mut now,
                veto_output.responses,
                &mut sensations,
                &mut impressions,
                &mut experiences,
                &mut teachings,
                &mut notes,
                &mut drive_impulses,
            );
            self.goal_system
                .add_drive_impulses(std::mem::take(&mut drive_impulses));
            notes.push(format!(
                "Safety vetoed {:?}: {}",
                chosen_action,
                describe_safety_reason(safety.reason.clone())
            ));
        }
        now.extensions.insert(
            "llm.action_proposal".to_string(),
            serde_json::to_value(&llm_action_proposal)?,
        );

        attach_structured_predictions_to_experience(
            &mut embodied_experience,
            &futures,
            &now,
            Some(&chosen_action),
        );
        experiences.push(embodied_experience);

        let reign_outcome = reign_input.as_ref().map(|input| {
            let accepted_by_conductor = reign_action
                .as_ref()
                .map(|action| action == &chosen_action)
                .unwrap_or(false);
            ReignOutcome {
                input_id: input.id,
                accepted_by_conductor,
                vetoed_by_safety: safety.vetoed,
                final_action: Some(chosen_action.clone()),
                reason: if safety.vetoed {
                    Some(describe_safety_reason(safety.reason.clone()).to_string())
                } else if accepted_by_conductor {
                    None
                } else {
                    Some("conductor chose another action".to_string())
                },
            }
        });

        if experiences.is_empty() {
            experiences.push(Experience::new(
                "realtime.state",
                format!(
                    "I am at t={}ms with battery {:.2}.",
                    now.t_ms, now.body.battery_level
                ),
                Vec::new(),
                Vec::new(),
                now.t_ms,
                now.t_ms,
            ));
        }

        self.last_behavior_runs = behavior_runs.clone();
        let mut frame = ExperienceFrame {
            id: frame_id,
            t_ms: now.t_ms,
            now: now.clone(),
            sensations,
            impressions,
            experiences: experiences.clone(),
            z: Some(latent.clone()),
            chosen_action: Some(chosen_action.clone()),
            conscious_command: llm_tick.conscious_command.clone(),
            reign_input,
            reign_outcome,
            predicted_futures: futures.clone(),
            behavior_runs,
            actual_next: None,
            reward: Reward::default(),
            surprise: now.surprise.clone(),
            memory_recall: recall.hits.clone(),
            recollections: recall.recollections.clone(),
            llm_teaching: teachings.clone(),
            counterfactuals: teachings
                .iter()
                .flat_map(|teaching| teaching.counterfactuals.clone())
                .collect(),
            notes,
        };
        attach_memory_links_to_frame(&mut frame);

        self.ledger.append(&frame).await?;
        self.memory_recall.observe_frame(&frame).await?;
        let surprise_computer = self.surprise_computer.clone();
        let reward_computer = self.reward_computer.clone();
        let mut inline_learning = InlineLearningTickStatus {
            enabled: self.inline_learning.is_enabled(),
            mode: self.inline_learning.mode,
            samples_observed: 0,
            train_steps_used: 0,
        };
        if let Some(transition) = self.transition_builder.observe(
            PendingFrame {
                frame_id: frame.id,
                now: frame.now.clone(),
                z: latent,
                action: frame.chosen_action.clone(),
                predicted_futures: frame.predicted_futures.clone(),
            },
            |previous, current| {
                let surprise = surprise_computer.compute(
                    &previous.predicted_futures,
                    &current.z,
                    &current.now,
                );
                reward_computer.compute(
                    &previous.now,
                    previous.action.as_ref(),
                    &current.now,
                    &surprise,
                )
            },
            |previous, current| {
                surprise_computer.compute(&previous.predicted_futures, &current.z, &current.now)
            },
        ) {
            self.ledger.append_transition(&transition).await?;
            self.memory_recall.observe_transition(&transition).await?;
            inline_learning = self.observe_inline_learning(&transition)?;
        }
        self.memory_store.store(&frame).await?;

        self.semantic_outcomes
            .remember(&now.world, executed_goal_behavior);
        Ok(RuntimeTick {
            frame,
            experience: experiences.last().cloned().unwrap_or_else(|| {
                Experience::new(
                    "realtime.state",
                    "I am active.",
                    Vec::new(),
                    Vec::new(),
                    now.t_ms,
                    now.t_ms,
                )
            }),
            chosen_action: Some(chosen_action),
            skill_request: (self.action_selector_mode == ActionSelectorMode::Goal
                && mechanical_reign_action_for_selection.is_none()
                && !sleeping)
                .then_some(goal_skill_request)
                .flatten(),
            skill_status: None,
            recall,
            llm: llm_tick,
            combobulation,
            inline_learning,
        })
    }

}
