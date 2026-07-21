impl WorldModelUpdater {
    pub fn update(&mut self, mut now: Now, context: WorldModelUpdateContext) -> Now {
        let previous = self.entities.clone();
        let mut trace = BeliefUpdateTrace {
            builder_implementation: "pete_now::WorldModelUpdater".to_string(),
            builder_version: "1".to_string(),
            ..BeliefUpdateTrace::default()
        };
        self.age_entities(now.t_ms, &mut trace);
        self.integrate_objects(&now, &mut trace);
        self.integrate_sound(&now, &mut trace);
        self.integrate_memory(&now, &mut trace);
        self.mark_contradictions(&mut trace);
        self.remove_expired(now.t_ms, &mut trace);

        for id in self.entities.keys() {
            if previous.contains_key(id) {
                if previous.get(id) != self.entities.get(id) {
                    trace.updated.push(id.0.clone());
                }
            } else {
                trace.added.push(id.0.clone());
            }
        }
        let local_geometry = local_geometry(&now);
        let hazards = hazard_beliefs(&now);
        let world_context = context_beliefs(&now);
        let authority = authority_belief(&now);
        let social = self.social.update(&now, &self.entities);
        let active_interaction = social.active_interaction.as_ref();
        let temporal = self.temporal.update(TemporalUpdateInput {
            monotonic_now_ms: now.t_ms,
            wall_clock_unix_ms: context.wall_clock_unix_ms,
            replay_now_ms: context.replay_now_ms,
            charging: now.body.charging,
            contact_or_recovery: now.body.flags.bump_left
                || now.body.flags.bump_right
                || context.active_goal.as_deref() == Some("escape_danger"),
            active_goal: context.active_goal.clone(),
            interaction_id: active_interaction
                .map(|interaction| interaction.interaction_id.0.clone()),
            interaction_participants: active_interaction
                .map(|interaction| {
                    interaction
                        .participants
                        .iter()
                        .map(|person| EntityId(person.0.clone()))
                        .collect()
                })
                .unwrap_or_default(),
            expectations: context.temporal_expectations.clone(),
            temporal_beliefs: temporal_beliefs(&now),
        });
        let epistemic = self.epistemic.update(
            &now,
            &self.entities,
            &local_geometry,
            &social,
            context.strategy_failure_pressure,
            context.epistemic_attempt.as_ref(),
        );
        let semantic = self
            .semantic
            .update(&now, &self.entities, &context.semantic_observations);
        let mut self_model = self.self_model(&now, &context);
        self_model.continuity.episode_id = temporal
            .current_episode
            .as_ref()
            .map(|episode_id| episode_id.0.clone())
            .or(self_model.continuity.episode_id);
        record_meta_evidence(&mut trace, &self_model.battery_meta);
        record_meta_evidence(&mut trace, &self_model.charging_meta);
        record_meta_evidence(&mut trace, &self_model.stuck_meta);
        record_meta_evidence(&mut trace, &self_model.pose_meta);
        if let Some(trap_kind) = &self_model.stuck_trap_kind {
            record_meta_evidence(&mut trace, &trap_kind.meta);
        }
        record_meta_evidence(&mut trace, &self_model.organism_id.meta);
        record_meta_evidence(&mut trace, &self_model.body.body_id.meta);
        record_meta_evidence(&mut trace, &self_model.body.pose.meta);
        record_meta_evidence(&mut trace, &self_model.body.energy.meta);
        record_meta_evidence(&mut trace, &self_model.body.charging.meta);
        record_meta_evidence(&mut trace, &self_model.body.health.meta);
        record_meta_evidence(&mut trace, &self_model.agency.meta);
        for capability in self_model.capabilities.capabilities.values() {
            record_meta_evidence(&mut trace, &capability.meta);
        }
        for service in self_model.service_state.services.values() {
            record_meta_evidence(&mut trace, &service.meta);
        }
        for status in self_model.goal_status.values() {
            record_meta_evidence(&mut trace, &status.meta);
        }
        for belief in [
            local_geometry.nearest_m.as_ref(),
            local_geometry.left_clearance_m.as_ref(),
            local_geometry.center_clearance_m.as_ref(),
            local_geometry.right_clearance_m.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            record_meta_evidence(&mut trace, &belief.meta);
        }
        for belief in [
            hazards.immediate_risk.as_ref(),
            hazards.remembered_risk.as_ref(),
            hazards.predicted_risk.as_ref(),
            world_context.novelty.as_ref(),
            world_context.surprise.as_ref(),
            world_context.prediction_uncertainty.as_ref(),
            world_context.map_confidence.as_ref(),
            world_context.safe_bearing_rad.as_ref(),
            world_context.frontier_bearing_rad.as_ref(),
            world_context.llm_confidence.as_ref(),
            world_context.expected_battery_delta.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            record_meta_evidence(&mut trace, &belief.meta);
        }
        if let Some(authority) = &authority {
            record_meta_evidence(&mut trace, &authority.meta);
        }
        for person in social.people.values() {
            record_meta_evidence(&mut trace, &person.meta);
            for identity in &person.identity_hypotheses {
                for evidence in &identity.evidence {
                    trace.input_evidence_ids.push(evidence.id.clone());
                }
            }
        }
        for question in &epistemic.active_questions {
            for evidence in &question.provenance {
                trace.input_evidence_ids.push(evidence.id.clone());
            }
        }
        for relation in semantic.relations.values() {
            for evidence in relation
                .supporting_evidence
                .iter()
                .chain(relation.contradicting_evidence.iter())
            {
                trace.input_evidence_ids.push(evidence.id.clone());
            }
        }
        trace.input_evidence_ids.sort();
        trace.input_evidence_ids.dedup();
        trace.added.sort();
        trace.updated.sort();
        trace.removed.sort();
        trace.freshness_changes.sort();
        trace.confidence_changes.sort();
        trace.contradiction_resolutions.sort();

        self.revision = self.revision.saturating_add(1);
        now.world = WorldModelSnapshot {
            schema_version: 3,
            revision: self.revision,
            t_ms: now.t_ms,
            entities: self.entities.clone(),
            self_model,
            local_geometry,
            hazards,
            context: world_context,
            temporal,
            social,
            epistemic,
            semantic,
            authority,
            update_trace: trace,
        };
        now
    }

    fn age_entities(&mut self, now_ms: u64, trace: &mut BeliefUpdateTrace) {
        for entity in self.entities.values_mut() {
            let age_ms = now_ms.saturating_sub(entity.last_observed_at_ms);
            let previous = entity.meta.freshness.clone();
            let previous_confidence = entity.confidence;
            entity.meta.freshness = freshness(age_ms, identity_policy(&entity.kind));
            entity.meta.valid_at_ms = now_ms;
            let base_confidence = entity
                .attributes
                .get("observed_confidence")
                .copied()
                .unwrap_or(entity.confidence);
            entity.confidence =
                decayed_confidence(base_confidence, age_ms, identity_policy(&entity.kind));
            entity.meta.confidence = entity.confidence;
            if (entity.confidence - previous_confidence).abs() > f32::EPSILON {
                trace.confidence_changes.push(format!(
                    "{}:{previous_confidence:.6}->{:.6}",
                    entity.id.0, entity.confidence
                ));
            }
            if previous != entity.meta.freshness {
                trace.freshness_changes.push(format!(
                    "{}:identity:{previous:?}->{:?}",
                    entity.id.0, entity.meta.freshness
                ));
            }
            if age_ms > bearing_policy(&entity.kind).aging_after_ms {
                if entity.bearing_rad.take().is_some() {
                    trace
                        .freshness_changes
                        .push(format!("{}:bearing:stale", entity.id.0));
                }
                entity.bearing_meta = None;
            }
            if age_ms > distance_policy(&entity.kind).aging_after_ms {
                entity.distance_m = None;
                entity.distance_meta = None;
                entity.reachability = ReachabilityEstimate::default();
                entity.reachability_meta = None;
            }
        }
    }

    fn integrate_objects(&mut self, now: &Now, trace: &mut BeliefUpdateTrace) {
        for (index, observation) in now.objects.observations.iter().enumerate() {
            let kind = WorldEntityKind::from(&observation.class);
            let id = EntityId(format!(
                "{}:{}",
                entity_kind_key(&kind),
                normalized_label(&observation.label)
            ));
            let source_kind = match observation.source {
                ObjectObservationSource::Sim | ObjectObservationSource::CreateIr => {
                    BeliefSourceKind::DirectObservation
                }
                ObjectObservationSource::HumanLabel => BeliefSourceKind::HumanClaim,
                ObjectObservationSource::Kinect | ObjectObservationSource::Captioner => {
                    BeliefSourceKind::DerivedPerception
                }
                ObjectObservationSource::Unknown => BeliefSourceKind::Unknown,
            };
            let evidence = evidence_ref(
                &format!("object.{:?}", observation.source).to_lowercase(),
                &format!("{}:{index}", observation.label),
                now.t_ms,
                "object-observation-v1",
            );
            trace.input_evidence_ids.push(evidence.id.clone());
            let meta = belief_meta(
                observation.confidence,
                now.t_ms,
                source_kind,
                evidence.clone(),
                Some("base_link".to_string()),
            );
            let pose = observation.distance_m.map(|distance| {
                let heading = now.body.odometry.heading_rad + observation.bearing_rad;
                WorldPose {
                    x_m: now.body.odometry.x_m + heading.cos() * distance,
                    y_m: now.body.odometry.y_m + heading.sin() * distance,
                }
            });
            let occluded = observation.distance_m.is_some_and(|target_distance| {
                now.objects.observations.iter().any(|other| {
                    other.class == ObjectClass::Obstacle
                        && other.distance_m.is_some_and(|obstacle_distance| {
                            obstacle_distance + 0.10 < target_distance
                                && normalize_angle(other.bearing_rad - observation.bearing_rad)
                                    .abs()
                                    < 0.28
                        })
                })
            });
            let reachable = observation.distance_m.is_some() && !occluded;
            self.entities.insert(
                id.clone(),
                WorldEntity {
                    id,
                    kind,
                    label: observation.label.clone(),
                    last_observed_at_ms: now.t_ms,
                    confidence: observation.confidence.clamp(0.0, 1.0),
                    meta: meta.clone(),
                    pose,
                    bearing_rad: Some(observation.bearing_rad),
                    bearing_meta: Some(meta.clone()),
                    distance_m: observation.distance_m,
                    distance_meta: observation.distance_m.map(|_| meta.clone()),
                    reachability: ReachabilityEstimate {
                        reachable,
                        confidence: observation.confidence.clamp(0.0, 1.0),
                    },
                    reachability_meta: Some(meta.clone()),
                    attributes: BTreeMap::from([(
                        "observed_confidence".to_string(),
                        observation.confidence.clamp(0.0, 1.0),
                    )]),
                    provenance: vec![evidence],
                },
            );
        }
    }

    fn integrate_sound(&mut self, now: &Now, trace: &mut BeliefUpdateTrace) {
        let Some(label) = now
            .ear
            .transcript
            .clone()
            .filter(|value| !value.trim().is_empty())
            .or_else(|| (!now.ear.features.is_empty()).then(|| "unidentified sound".to_string()))
        else {
            return;
        };
        let evidence = evidence_ref("ear", "sound_source", now.t_ms, "sound-hypothesis-v1");
        trace.input_evidence_ids.push(evidence.id.clone());
        let confidence = now.ear.asr.confidence.clamp(0.2, 1.0);
        let meta = belief_meta(
            confidence,
            now.t_ms,
            BeliefSourceKind::DerivedPerception,
            evidence.clone(),
            Some("base_link".to_string()),
        );
        let id = EntityId("sound_source:current".to_string());
        self.entities.insert(
            id.clone(),
            WorldEntity {
                id,
                kind: WorldEntityKind::SoundSource,
                label,
                last_observed_at_ms: now.t_ms,
                confidence,
                meta,
                provenance: vec![evidence],
                ..WorldEntity::default()
            },
        );
    }

    fn integrate_memory(&mut self, now: &Now, trace: &mut BeliefUpdateTrace) {
        let Some(bearing) = now.memory.nearby_best_charge_direction_rad else {
            return;
        };
        let confidence =
            (now.memory.place_charge_value * now.memory.map_confidence).clamp(0.0, 1.0);
        if confidence <= 0.01 {
            return;
        }
        let evidence = evidence_ref(
            "memory.recall",
            "charger_direction",
            now.t_ms,
            "memory-belief-v1",
        );
        trace.input_evidence_ids.push(evidence.id.clone());
        let meta = belief_meta(
            confidence,
            now.t_ms,
            BeliefSourceKind::MemoryRecall,
            evidence.clone(),
            Some("base_link".to_string()),
        );
        let id = EntityId("charger:remembered_home".to_string());
        self.entities.insert(
            id.clone(),
            WorldEntity {
                id,
                kind: WorldEntityKind::Charger,
                label: "remembered charger".to_string(),
                last_observed_at_ms: now.t_ms,
                confidence,
                meta: meta.clone(),
                bearing_rad: Some(bearing),
                bearing_meta: Some(meta.clone()),
                reachability: ReachabilityEstimate {
                    reachable: false,
                    confidence,
                },
                reachability_meta: Some(meta),
                attributes: BTreeMap::from([("observed_confidence".to_string(), confidence)]),
                provenance: vec![evidence],
                ..WorldEntity::default()
            },
        );
    }

    fn mark_contradictions(&mut self, trace: &mut BeliefUpdateTrace) {
        let mut by_label: BTreeMap<String, Vec<EntityId>> = BTreeMap::new();
        for entity in self.entities.values() {
            by_label
                .entry(normalized_label(&entity.label))
                .or_default()
                .push(entity.id.clone());
        }
        for ids in by_label.values().filter(|ids| ids.len() > 1) {
            let kinds = ids
                .iter()
                .filter_map(|id| self.entities.get(id))
                .map(|entity| entity.kind.clone())
                .collect::<BTreeSet<_>>();
            if kinds.len() <= 1 {
                continue;
            }
            for id in ids {
                let contradictions = ids
                    .iter()
                    .filter(|other| *other != id)
                    .filter_map(|other| self.entities.get(other))
                    .flat_map(|entity| entity.provenance.clone())
                    .collect::<Vec<_>>();
                if let Some(entity) = self.entities.get_mut(id) {
                    entity.meta.contradiction_refs = contradictions;
                }
            }
            trace.contradiction_resolutions.push(format!(
                "preserved:{}",
                ids.iter()
                    .map(|id| id.0.as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
    }

    fn remove_expired(&mut self, now_ms: u64, trace: &mut BeliefUpdateTrace) {
        self.entities.retain(|id, entity| {
            let keep = now_ms.saturating_sub(entity.last_observed_at_ms)
                <= identity_policy(&entity.kind).invalidate_after_ms
                && entity.confidence > 0.01;
            if !keep {
                trace.removed.push(id.0.clone());
            }
            keep
        });
    }

    fn self_model(&mut self, now: &Now, context: &WorldModelUpdateContext) -> SelfModelSnapshot {
        let body_evidence = evidence_ref("body", "state", now.t_ms, "body-belief-v1");
        let body_meta = belief_meta(
            1.0,
            now.t_ms,
            BeliefSourceKind::DirectObservation,
            body_evidence,
            Some("base_link".to_string()),
        );
        let stuck = now
            .extensions
            .get("sim.stuck")
            .and_then(|value| value.get("values"))
            .and_then(|value| value.as_array())
            .and_then(|values| values.first())
            .and_then(|value| value.as_f64())
            .is_some_and(|active| active > 0.0);
        let stuck_trap_kind = stuck.then(|| {
            let value = now
                .extensions
                .get("sim.stuck")
                .and_then(|value| value.get("values"))
                .and_then(|value| value.as_array())
                .and_then(|values| values.get(10))
                .and_then(|value| value.as_f64())
                .map(|code| match code.round() as i32 {
                    1 => StuckTrapKind::Wall,
                    2 => StuckTrapKind::Corner,
                    3 => StuckTrapKind::Column,
                    _ => StuckTrapKind::Unknown,
                })
                .unwrap_or_default();
            Belief {
                value,
                meta: simple_meta(
                    now.t_ms,
                    BeliefSourceKind::DerivedPerception,
                    "recovery.trap_kind",
                ),
            }
        });
        let mut goal_status = context.goal_status.clone();
        for (goal_id, status) in &mut goal_status {
            status.meta = simple_meta(
                now.t_ms,
                BeliefSourceKind::ActionOutcome,
                &format!("goal_outcome.{goal_id}"),
            );
        }
        let pose_meta = belief_meta(
            1.0,
            now.t_ms,
            BeliefSourceKind::DirectObservation,
            evidence_ref("body", "odometry", now.t_ms, "body-belief-v1"),
            Some("map".to_string()),
        );
        let identity_meta = simple_meta(
            now.t_ms,
            BeliefSourceKind::Map,
            "self.identity.configuration",
        );
        let possession = now.extensions.get("brainstem.possession");
        let device_id = possession
            .and_then(|value| value.get("brainstem_device_id"))
            .and_then(|value| value.as_str())
            .map(|value| BrainstemDeviceId(value.to_string()));
        let boot_id = possession
            .and_then(|value| value.get("brainstem_boot_id"))
            .and_then(|value| value.as_str())
            .map(|value| BootId(value.to_string()));
        let boot_changed = self
            .last_brainstem_boot_id
            .as_ref()
            .zip(boot_id.as_ref())
            .is_some_and(|(previous, current)| previous != current);
        if boot_id.is_some() {
            self.last_brainstem_boot_id = boot_id.clone();
        }
        let session_id = possession
            .and_then(|value| value.get("session_id"))
            .and_then(|value| value.as_str())
            .map(|value| SessionId(value.to_string()));
        let lease_id = possession
            .and_then(|value| value.get("lease_id"))
            .and_then(|value| value.as_str())
            .map(ToOwned::to_owned);
        if boot_changed {
            self.invalidated_authority_lease = Some(
                lease_id
                    .clone()
                    .unwrap_or_else(|| "<missing-lease>".to_string()),
            );
        } else if self
            .invalidated_authority_lease
            .as_ref()
            .zip(lease_id.as_ref())
            .is_some_and(|(invalidated, current)| invalidated != current)
        {
            self.invalidated_authority_lease = None;
        }
        let authority_invalidated = self.invalidated_authority_lease.as_ref().is_some_and(|id| {
            lease_id.as_deref() == Some(id.as_str())
                || (id == "<missing-lease>" && lease_id.is_none())
        });
        let reported_possessed = possession
            .and_then(|value| value.get("possessed"))
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let reported_armed = possession
            .and_then(|value| value.get("brainstem_armed"))
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let possessed = reported_possessed && !authority_invalidated;
        let armed = reported_armed && !authority_invalidated;
        let moving =
            now.body.velocity.forward_m_s.abs() > 0.01 || now.body.velocity.turn_rad_s.abs() > 0.01;
        let reign = now.reign.latest.as_ref();
        let controller = reign
            .map(|input| match input.mode {
                pete_actions::ReignMode::Direct => ControlProvenance::HumanDirect,
                pete_actions::ReignMode::Assist => ControlProvenance::HumanAssist,
                pete_actions::ReignMode::Suggest => ControlProvenance::HumanSuggestion,
                pete_actions::ReignMode::ObserveOnly => ControlProvenance::Autonomous,
            })
            .unwrap_or_else(|| {
                if context.active_goal.is_some() {
                    ControlProvenance::Autonomous
                } else {
                    ControlProvenance::None
                }
            });
        let agency_meta = if let Some(authority) = reign {
            simple_meta(
                now.t_ms,
                BeliefSourceKind::HumanClaim,
                &format!("reign.{}", authority.id),
            )
        } else {
            simple_meta(
                now.t_ms,
                BeliefSourceKind::ActionOutcome,
                "control.autonomous",
            )
        };

        let mut capabilities = CapabilitySelfModel::default();
        let body_current = now.t_ms.saturating_sub(now.body.last_update_ms) <= 1_000;
        insert_capability(
            &mut capabilities,
            "sensor:body",
            CapabilityKind::Sensor,
            body_current,
            (!body_current).then_some("body telemetry is stale"),
            now.t_ms,
        );
        let range_current = (!now.range.beams.is_empty() || now.range.nearest_m.is_some())
            && now.t_ms.saturating_sub(now.range.captured_at_ms) <= 1_000;
        insert_capability(
            &mut capabilities,
            "sensor:range",
            CapabilityKind::Sensor,
            range_current,
            (!range_current).then_some("range observations are missing or stale"),
            now.t_ms,
        );
        let visual_current = now.objects.observations.iter().any(|observation| {
            matches!(
                observation.source,
                ObjectObservationSource::Kinect | ObjectObservationSource::Captioner
            )
        });
        insert_capability(
            &mut capabilities,
            "sensor:vision",
            CapabilityKind::Sensor,
            visual_current,
            (!visual_current).then_some("camera evidence is unavailable this tick"),
            now.t_ms,
        );
        let drive_available = now.body.health.health > 0.2 && !now.body.flags.wheel_drop;
        insert_capability(
            &mut capabilities,
            "actuator:drive",
            CapabilityKind::Actuator,
            drive_available,
            (!drive_available).then_some("drive is unsafe or body health is degraded"),
            now.t_ms,
        );
        insert_capability(
            &mut capabilities,
            "actuator:speaker",
            CapabilityKind::Actuator,
            true,
            None,
            now.t_ms,
        );
        for goal in &context.registered_goals {
            insert_capability(
                &mut capabilities,
                &format!("goal:{goal}"),
                CapabilityKind::Goal,
                true,
                None,
                now.t_ms,
            );
        }
        for behavior in &context.registered_behaviors {
            insert_capability(
                &mut capabilities,
                &format!("behavior:{behavior}"),
                CapabilityKind::Behavior,
                true,
                None,
                now.t_ms,
            );
        }
        for skill in &context.registered_skills {
            insert_capability(
                &mut capabilities,
                &format!("skill:{skill}"),
                CapabilityKind::Skill,
                true,
                None,
                now.t_ms,
            );
        }
        for capability in &context.capability_evidence {
            capabilities
                .capabilities
                .insert(capability.id.clone(), capability.clone());
        }
        let hardware_authority_available = possession.is_none() || (possessed && armed);
        for capability in capabilities.capabilities.values_mut() {
            if matches!(
                capability.kind,
                CapabilityKind::Actuator | CapabilityKind::Behavior | CapabilityKind::Skill
            ) {
                capability.authorized = capability.authorized && hardware_authority_available;
                if !capability.authorized && capability.authority_reason.is_none() {
                    capability.authority_reason = Some(if authority_invalidated {
                        "brainstem reboot invalidated the control lease".to_string()
                    } else {
                        "no current actuation authority".to_string()
                    });
                }
            }
        }
        let mut service_state = CognitiveServiceSummary {
            services: context.cognitive_services.clone(),
        };
        integrate_cognitive_registry(now, &mut service_state);
        service_state
            .services
            .entry("local_language".to_string())
            .or_insert_with(|| CognitiveServiceBelief {
                available: true,
                confidence: 1.0,
                meta: simple_meta(now.t_ms, BeliefSourceKind::Map, "service.local_language"),
                ..CognitiveServiceBelief::default()
            });
        service_state
            .services
            .entry("rich_language".to_string())
            .or_insert_with(|| CognitiveServiceBelief {
                available: false,
                confidence: 1.0,
                unavailable_reason: Some("enhanced cognition was not reported available".into()),
                meta: simple_meta(
                    now.t_ms,
                    BeliefSourceKind::Map,
                    "service.rich_language.missing",
                ),
                ..CognitiveServiceBelief::default()
            });
        for (service, state) in &service_state.services {
            insert_capability(
                &mut capabilities,
                &format!("service:{service}"),
                CapabilityKind::CognitiveService,
                state.available,
                state.unavailable_reason.as_deref(),
                now.t_ms,
            );
        }
        let faults = [
            now.body.flags.wheel_drop.then_some("wheel_drop"),
            (now.body.health.health <= 0.2).then_some("body_health_critical"),
        ]
        .into_iter()
        .flatten()
        .map(|fault| Belief {
            value: fault.to_string(),
            meta: body_meta.clone(),
        })
        .collect::<Vec<_>>();
        let tilt_known = now.imu.orientation.len() >= 2;
        let tilted = tilt_known
            .then(|| now.imu.orientation[0].abs() > 0.35 || now.imu.orientation[1].abs() > 0.35);
        let body = SelfBodyBelief {
            body_id: Belief {
                value: BodyId("pete.primary_body".to_string()),
                meta: identity_meta.clone(),
            },
            implementation: Belief {
                value: "mobile_robot".to_string(),
                meta: identity_meta.clone(),
            },
            implementation_version: Belief {
                value: "1".to_string(),
                meta: identity_meta.clone(),
            },
            brainstem_device_id: device_id.map(|value| Belief {
                value,
                meta: agency_meta.clone(),
            }),
            brainstem_boot_id: boot_id.map(|value| Belief {
                value,
                meta: agency_meta.clone(),
            }),
            pose: Belief {
                value: now.body.odometry,
                meta: pose_meta.clone(),
            },
            envelope: Belief {
                value: BodyEnvelope {
                    radius_m: 0.18,
                    height_m: 0.10,
                },
                meta: identity_meta.clone(),
            },
            energy: Belief {
                value: now.body.battery_level,
                meta: body_meta.clone(),
            },
            charging: Belief {
                value: now.body.charging,
                meta: body_meta.clone(),
            },
            health: Belief {
                value: now.body.health.health,
                meta: body_meta.clone(),
            },
            faults,
            being_moved: Some(Belief {
                value: moving && context.active_behavior.is_none() && reign.is_none(),
                meta: body_meta.clone(),
            }),
            tilted: tilted.map(|value| Belief {
                value,
                meta: simple_meta(
                    now.t_ms,
                    BeliefSourceKind::DirectObservation,
                    "body.imu.tilt",
                ),
            }),
            blocked: Some(Belief {
                value: stuck || now.body.flags.bump_left || now.body.flags.bump_right,
                meta: body_meta.clone(),
            }),
            carried: now.body.flags.wheel_drop.then(|| Belief {
                value: true,
                meta: body_meta.clone(),
            }),
        };
        let motivation = MotivationSummary {
            drives: context.drive_summaries.clone(),
            selected_goal: context.active_goal.clone(),
            commitment_age_ms: context.commitment_age_ms,
            expected_progress: context.expected_progress,
            recent_progress: context.recent_progress,
            uncertainty: context.uncertainty,
            strategy_failure_pressure: context.strategy_failure_pressure,
        };
        let mut continuity = context.continuity.clone();
        continuity.session_id = continuity.session_id.or_else(|| session_id.clone());
        continuity.important_relationship_refs.extend(
            self.entities
                .values()
                .filter(|entity| entity.kind == WorldEntityKind::Person)
                .map(|entity| entity.id.clone()),
        );
        for entity in &now.memory.remembered_entities {
            if entity.has_label("Person") || entity.has_label("person") {
                continuity
                    .important_relationship_refs
                    .push(EntityId(entity.id.clone()));
            }
            if entity.has_label("Place") || entity.has_label("place") {
                continuity.important_place_refs.push(entity.id.clone());
            }
        }
        continuity.important_relationship_refs.sort();
        continuity.important_relationship_refs.dedup();
        continuity.important_place_refs.sort();
        continuity.important_place_refs.dedup();
        let mut active_control =
            context
                .active_control
                .clone()
                .unwrap_or_else(|| ActiveControlSummary {
                    goal_id: context.active_goal.clone(),
                    behavior_id: context.active_behavior.clone(),
                    skill_id: context.active_skill.clone(),
                    provenance: controller.clone(),
                    unable_to_act_reason: (!drive_available)
                        .then_some("drive capability is unavailable".to_string()),
                    ..ActiveControlSummary::default()
                });
        if reign.is_some_and(|input| input.mode == pete_actions::ReignMode::Direct) {
            active_control.provenance = ControlProvenance::HumanDirect;
        }
        SelfModelSnapshot {
            organism_id: Belief {
                value: OrganismId("pete".to_string()),
                meta: identity_meta,
            },
            body,
            capabilities,
            agency: AgencyState {
                controller,
                reign_mode: reign.map(|input| format!("{:?}", input.mode).to_ascii_lowercase()),
                reign_source: reign.map(|input| format!("{:?}", input.source).to_ascii_lowercase()),
                session_id: session_id.map(|value| Belief {
                    value,
                    meta: agency_meta.clone(),
                }),
                lease_id: lease_id.map(|value| Belief {
                    value,
                    meta: agency_meta.clone(),
                }),
                possessed: Belief {
                    value: possessed,
                    meta: agency_meta.clone(),
                },
                armed: Belief {
                    value: armed,
                    meta: agency_meta.clone(),
                },
                stopped: !moving,
                moving,
                pending_direct_override: reign
                    .is_some_and(|input| input.mode == pete_actions::ReignMode::Direct),
                authority_conflicts: if authority_invalidated {
                    agency_meta.provenance.clone()
                } else {
                    Vec::new()
                },
                meta: agency_meta,
            },
            motivation,
            active_control,
            continuity,
            service_state,
            meta: body_meta.clone(),
            battery_level: now.body.battery_level,
            battery_meta: body_meta.clone(),
            charging: now.body.charging,
            charging_meta: body_meta.clone(),
            stuck,
            stuck_meta: body_meta,
            stuck_trap_kind,
            pose: now.body.odometry,
            pose_meta,
            contact: now.body.flags.bump_left || now.body.flags.bump_right || now.body.flags.wall,
            bump_left: now.body.flags.bump_left,
            moving,
            range_nearest_m: now.range.nearest_m,
            active_goal: context.active_goal.clone(),
            goal_status,
        }
    }
}
