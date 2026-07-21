const ENTITY_CONFIDENCE_DECAY_PER_TICK: f32 = 0.998;
const ENTITY_OCCLUDE_THRESHOLD: f32 = 0.25;
const ENTITY_VANISH_THRESHOLD: f32 = 0.05;
const HYPOTHESIS_CONFIDENCE_DECAY_PER_TICK: f32 = 0.999;
const HYPOTHESIS_PROMOTION_THRESHOLD: f32 = 0.72;
const HYPOTHESIS_REVIEW_MARGIN: f32 = 0.08;
const HYPOTHESIS_STALE_MS: u64 = 30_000;
const HYPOTHESIS_REVIEW_STALE_MS: u64 = 120_000;

/// Stores and maintains all persistent entity hypotheses.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EntityMemory {
    /// All known entity records keyed by entity id.
    pub entities: BTreeMap<String, EntityHypothesis>,
    #[serde(default)]
    pub binding_candidates: Vec<BindingCandidate>,
    #[serde(default)]
    pub tracking_hypotheses: BTreeMap<String, TrackingHypothesis>,
    last_tick: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VectorBindingKind {
    Face,
    Voice,
    Scene,
}

impl EntityMemory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Process a single `Now` snapshot: merge object observations and update
    /// cross-modal evidence.
    pub fn observe_now(&mut self, now: &Now, cell_key: Option<PlaceCellKey>) {
        let elapsed_ticks = self
            .last_tick
            .map(|last| now.t_ms.saturating_sub(last) / 100)
            .unwrap_or(1)
            .max(1);
        self.decay(elapsed_ticks, now.t_ms);
        self.last_tick = Some(now.t_ms);

        for observation in &now.objects.observations {
            let kind = object_class_slug(&observation.class).to_string();
            let id = format!("entity:{}:{}", kind, stable_slug(&observation.label));
            if let Some(existing) = self.entities.get_mut(&id) {
                existing.merge_observation(observation, now.t_ms, cell_key);
            } else {
                let hypothesis =
                    EntityHypothesis::from_observation(observation, now.t_ms, cell_key);
                self.entities.insert(id, hypothesis);
            }
        }

        let current_entity_ids = now
            .objects
            .observations
            .iter()
            .map(|observation| {
                format!(
                    "entity:{}:{}",
                    object_class_slug(&observation.class),
                    stable_slug(&observation.label)
                )
            })
            .collect::<BTreeSet<_>>();

        // Face vectors propose person bindings; they do not fan out to every person.
        for artifact in &now.face.vectors {
            self.admit_vector_artifact(
                artifact,
                VectorBindingKind::Face,
                now.t_ms,
                cell_key,
                &current_entity_ids,
            );
        }

        // Attach object vectors to active non-person entities, or to an explicit source entity.
        for artifact in &now.objects.vectors {
            let object_ids: Vec<String> = if let Some(source_id) = artifact.source_id.as_ref() {
                vec![source_id.clone()]
            } else {
                self.entities
                    .values()
                    .filter(|entity| {
                        entity.lifecycle == EntityLifecycleState::Active
                            && !entity.id.starts_with("entity:person:")
                    })
                    .map(|entity| entity.id.clone())
                    .collect()
            };
            for id in object_ids {
                if let Some(entity) = self.entities.get_mut(&id) {
                    entity.add_object_vector(&artifact.point_id);
                }
            }
        }

        // Voice vectors propose speaker bindings; ambiguity is preserved for review.
        for artifact in &now.voice.vectors {
            self.admit_vector_artifact(
                artifact,
                VectorBindingKind::Voice,
                now.t_ms,
                cell_key,
                &current_entity_ids,
            );
        }

        // Scene vectors bind only when there is explicit spatial/object context.
        for artifact in &now.eye.scene_vectors {
            self.admit_vector_artifact(
                artifact,
                VectorBindingKind::Scene,
                now.t_ms,
                cell_key,
                &current_entity_ids,
            );
        }

        let text_labels = now
            .ear
            .transcript
            .as_ref()
            .into_iter()
            .chain(now.ear.asr.transcript.as_ref())
            .map(|text| text.trim())
            .filter(|text| !text.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>();
        if !text_labels.is_empty() {
            let active_ids = self
                .entities
                .values()
                .filter(|entity| entity.lifecycle == EntityLifecycleState::Active)
                .map(|entity| entity.id.clone())
                .collect::<Vec<_>>();
            for id in active_ids {
                if let Some(entity) = self.entities.get_mut(&id) {
                    for text in &text_labels {
                        entity.add_text_label(text.clone(), 0.6, now.t_ms);
                    }
                }
            }
        }
    }

    fn admit_vector_artifact(
        &mut self,
        artifact: &VectorArtifact,
        kind: VectorBindingKind,
        t_ms: u64,
        cell_key: Option<PlaceCellKey>,
        current_entity_ids: &BTreeSet<String>,
    ) {
        let plausible_ids = self.plausible_entity_ids(artifact, kind, t_ms, cell_key);
        if plausible_ids.is_empty() {
            let reason = match kind {
                VectorBindingKind::Face => "face vector observed but no plausible person entity",
                VectorBindingKind::Voice => "voice observed but no plausible person entity",
                VectorBindingKind::Scene => {
                    "scene vector active but no spatially compatible object cluster"
                }
            };
            self.record_binding_candidate(BindingCandidate {
                left_cluster_id: "unresolved".to_string(),
                right_cluster_id: vector_cluster_id(kind, &artifact.point_id),
                relation: BindingRelation::RequiresReview,
                evidence: vec![BindingEvidence {
                    kind: BindingEvidenceKind::VectorSimilarity,
                    score: 0.25,
                    reason: "single vector artifact without compatible entity context".to_string(),
                }],
                confidence: 0.0,
                decision: BindingDecision::CollectMoreEvidence,
                reason: reason.to_string(),
            });
            self.upsert_new_entity_hypothesis(artifact, kind, t_ms);
            return;
        }

        let family_id = tracking_family_id(kind, &artifact.point_id);
        let mut candidate_ids = Vec::new();
        for entity_id in plausible_ids.clone() {
            let Some(entity) = self.entities.get(&entity_id) else {
                continue;
            };
            let Some(object_cluster_id) = entity.primary_object_cluster_id() else {
                continue;
            };
            let right_cluster_id = vector_cluster_id(kind, &artifact.point_id);
            let candidate = qualify_binding_candidate(
                entity,
                artifact,
                kind,
                object_cluster_id,
                right_cluster_id,
                t_ms,
                cell_key,
                plausible_ids.len(),
                current_entity_ids.contains(&entity_id),
            );
            let candidate_id = binding_candidate_id(&candidate);
            candidate_ids.push(candidate_id.clone());
            if let Some(entity) = self.entities.get_mut(&entity_id) {
                entity.record_binding_candidate(candidate.clone());
            }
            self.upsert_tracking_hypothesis(
                tracking_kind_from_vector(kind),
                family_id.clone(),
                Some(entity_id),
                artifact.point_id.clone(),
                candidate_id,
                candidate.evidence,
                t_ms,
            );
        }
        self.upsert_unknown_competitor_hypothesis(artifact, kind, &family_id, &candidate_ids, t_ms);
        self.evaluate_hypothesis_family(&family_id, artifact, kind, t_ms);
    }

    fn plausible_entity_ids(
        &self,
        artifact: &VectorArtifact,
        kind: VectorBindingKind,
        t_ms: u64,
        cell_key: Option<PlaceCellKey>,
    ) -> Vec<String> {
        if let Some(source_id) = artifact.source_id.as_ref() {
            if self.entities.contains_key(source_id) {
                return vec![source_id.clone()];
            }
        }
        self.entities
            .values()
            .filter(|entity| match kind {
                VectorBindingKind::Face | VectorBindingKind::Voice => entity.kind == "person",
                VectorBindingKind::Scene => entity.lifecycle == EntityLifecycleState::Active,
            })
            .filter(|entity| {
                if entity.lifecycle != EntityLifecycleState::Active {
                    return false;
                }
                let recent = t_ms.saturating_sub(entity.last_seen_ms) <= 1_000;
                let same_cell = cell_key
                    .map(|key| entity.location_cells.contains(&key))
                    .unwrap_or(false);
                let prior_support = match kind {
                    VectorBindingKind::Face => !entity.modality_support.face_vector_ids.is_empty(),
                    VectorBindingKind::Voice => {
                        !entity.modality_support.voice_vector_ids.is_empty()
                    }
                    VectorBindingKind::Scene => {
                        !entity.modality_support.scene_vector_ids.is_empty()
                    }
                };
                let explicit_label = !entity.modality_support.text_labels.is_empty();
                match kind {
                    VectorBindingKind::Face | VectorBindingKind::Voice => {
                        recent || same_cell || prior_support || explicit_label
                    }
                    VectorBindingKind::Scene => same_cell || prior_support,
                }
            })
            .map(|entity| entity.id.clone())
            .collect()
    }

    fn record_binding_candidate(&mut self, candidate: BindingCandidate) {
        self.binding_candidates.push(candidate);
        const MAX_BINDING_CANDIDATES: usize = 128;
        if self.binding_candidates.len() > MAX_BINDING_CANDIDATES {
            let excess = self.binding_candidates.len() - MAX_BINDING_CANDIDATES;
            self.binding_candidates.drain(0..excess);
        }
    }

    fn upsert_new_entity_hypothesis(
        &mut self,
        artifact: &VectorArtifact,
        kind: VectorBindingKind,
        t_ms: u64,
    ) {
        let family_id = tracking_family_id(kind, &artifact.point_id);
        self.upsert_unknown_competitor_hypothesis(artifact, kind, &family_id, &[], t_ms);
    }

    fn upsert_unknown_competitor_hypothesis(
        &mut self,
        artifact: &VectorArtifact,
        kind: VectorBindingKind,
        family_id: &str,
        competing_candidate_ids: &[String],
        t_ms: u64,
    ) {
        let mut evidence = vec![BindingEvidence {
            kind: BindingEvidenceKind::VectorSimilarity,
            score: if competing_candidate_ids.is_empty() {
                0.45
            } else {
                0.22
            },
            reason: match kind {
                VectorBindingKind::Face => "face may belong to a new unknown person".to_string(),
                VectorBindingKind::Voice => "voice may belong to a new unknown speaker".to_string(),
                VectorBindingKind::Scene => {
                    "scene may describe a new place or object context".to_string()
                }
            },
        }];
        if competing_candidate_ids.len() > 1 {
            evidence.push(BindingEvidence {
                kind: BindingEvidenceKind::SimultaneousConflict,
                score: 0.5,
                reason: "known-entity competitors are still unresolved".to_string(),
            });
        }
        let candidate_id = format!(
            "candidate:{}:{}:new",
            tracking_kind_slug(&tracking_kind_from_vector(kind)),
            stable_slug(&artifact.point_id)
        );
        self.upsert_tracking_hypothesis(
            tracking_kind_from_vector(kind),
            family_id.to_string(),
            None,
            artifact.point_id.clone(),
            candidate_id,
            evidence,
            t_ms,
        );
    }

    fn upsert_tracking_hypothesis(
        &mut self,
        kind: TrackingHypothesisKind,
        family_id: String,
        target_id: Option<String>,
        observation_id: String,
        candidate_id: String,
        evidence: Vec<BindingEvidence>,
        t_ms: u64,
    ) {
        let target_slug = target_id
            .as_deref()
            .map(stable_slug)
            .unwrap_or_else(|| "new-entity".to_string());
        let id = format!(
            "hypothesis:{}:{}:{}",
            tracking_kind_slug(&kind),
            stable_slug(&family_id),
            target_slug
        );
        if let Some(existing) = self.tracking_hypotheses.get_mut(&id) {
            if !existing.observation_ids.contains(&observation_id) {
                existing.observation_ids.push(observation_id);
            }
            if !existing.binding_candidate_ids.contains(&candidate_id) {
                existing.binding_candidate_ids.push(candidate_id);
            }
            existing.add_evidence(evidence, t_ms);
        } else {
            self.tracking_hypotheses.insert(
                id,
                TrackingHypothesis::new(
                    kind,
                    family_id,
                    target_id,
                    observation_id,
                    candidate_id,
                    evidence,
                    t_ms,
                ),
            );
        }
    }

    fn evaluate_hypothesis_family(
        &mut self,
        family_id: &str,
        artifact: &VectorArtifact,
        kind: VectorBindingKind,
        t_ms: u64,
    ) {
        let mut family = self
            .tracking_hypotheses
            .values()
            .filter(|hypothesis| {
                hypothesis.family_id == family_id
                    && !matches!(
                        hypothesis.state,
                        HypothesisState::Rejected | HypothesisState::Expired
                    )
            })
            .cloned()
            .collect::<Vec<_>>();
        if family.is_empty() {
            return;
        }
        family.sort_by(|left, right| {
            right
                .confidence
                .partial_cmp(&left.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let winner = family[0].clone();
        let runner_up_confidence = family.get(1).map(|hypothesis| hypothesis.confidence);
        let near_equal_competitor = runner_up_confidence
            .map(|confidence| winner.confidence - confidence < HYPOTHESIS_REVIEW_MARGIN)
            .unwrap_or(false);
        let promotable = self.hypothesis_passes_promotion_gates(&winner, near_equal_competitor);

        if promotable {
            self.promote_tracking_hypothesis(&winner, artifact, kind, t_ms);
            for hypothesis in family {
                if let Some(stored) = self.tracking_hypotheses.get_mut(&hypothesis.id) {
                    stored.state = if hypothesis.id == winner.id {
                        HypothesisState::Promoted
                    } else {
                        HypothesisState::Rejected
                    };
                    stored.last_updated_ms = t_ms;
                }
            }
            return;
        }

        for (index, hypothesis) in family.iter().enumerate() {
            if let Some(stored) = self.tracking_hypotheses.get_mut(&hypothesis.id) {
                if has_hard_contradiction(&stored.evidence) {
                    stored.state = HypothesisState::Rejected;
                } else if near_equal_competitor || !stored.contradictions.is_empty() {
                    stored.state = HypothesisState::NeedsReview;
                } else if index == 0 {
                    stored.state = HypothesisState::Winning;
                } else {
                    stored.state = HypothesisState::Losing;
                }
            }
        }
    }

    fn hypothesis_passes_promotion_gates(
        &self,
        hypothesis: &TrackingHypothesis,
        near_equal_competitor: bool,
    ) -> bool {
        if hypothesis.target_id.is_none() || near_equal_competitor {
            return false;
        }
        if hypothesis.confidence < HYPOTHESIS_PROMOTION_THRESHOLD {
            return false;
        }
        if has_hard_contradiction(&hypothesis.evidence) {
            return false;
        }
        let independent_evidence_types = hypothesis
            .evidence
            .iter()
            .filter(|evidence| {
                !matches!(
                    evidence.kind,
                    BindingEvidenceKind::Contradiction
                        | BindingEvidenceKind::SimultaneousConflict
                        | BindingEvidenceKind::VectorSimilarity
                        | BindingEvidenceKind::LlmSuggested
                )
            })
            .map(|evidence| binding_evidence_kind_rank(&evidence.kind))
            .collect::<BTreeSet<_>>()
            .len();
        let human_confirmed = hypothesis
            .evidence
            .iter()
            .any(|evidence| evidence.kind == BindingEvidenceKind::HumanConfirmed);
        human_confirmed || (hypothesis.evidence.len() >= 3 && independent_evidence_types >= 2)
    }

    fn promote_tracking_hypothesis(
        &mut self,
        hypothesis: &TrackingHypothesis,
        artifact: &VectorArtifact,
        kind: VectorBindingKind,
        t_ms: u64,
    ) {
        let Some(entity_id) = hypothesis.target_id.as_ref() else {
            return;
        };
        let Some(entity) = self.entities.get_mut(entity_id) else {
            return;
        };
        let Some(object_cluster_id) = entity.primary_object_cluster_id() else {
            return;
        };
        let actual_cluster_id = match kind {
            VectorBindingKind::Face => entity.add_face_vector(&artifact.point_id),
            VectorBindingKind::Voice => entity.add_voice_vector(&artifact.point_id),
            VectorBindingKind::Scene => entity.add_scene_vector(&artifact.point_id),
        };
        entity.upsert_binding_edge(
            object_cluster_id,
            actual_cluster_id,
            match kind {
                VectorBindingKind::Face | VectorBindingKind::Voice => {
                    BindingRelation::LikelySameEntity
                }
                VectorBindingKind::Scene => BindingRelation::ProjectsTo,
            },
            hypothesis.confidence,
            t_ms,
        );
    }

    pub fn confirm_tracking_hypothesis(&mut self, hypothesis_id: &str, t_ms: u64) -> bool {
        let Some(hypothesis) = self.tracking_hypotheses.get_mut(hypothesis_id) else {
            return false;
        };
        hypothesis.add_evidence(
            vec![BindingEvidence {
                kind: BindingEvidenceKind::HumanConfirmed,
                score: 1.0,
                reason: "human confirmed this hypothesis".to_string(),
            }],
            t_ms,
        );
        true
    }

    pub fn observe_frame(&mut self, frame: &ExperienceFrame, cell_key: Option<PlaceCellKey>) {
        self.observe_now(&frame.now, cell_key);
        if self.entities.is_empty() {
            return;
        }
        let active_ids = self
            .entities
            .values()
            .filter(|entity| entity.lifecycle == EntityLifecycleState::Active)
            .map(|entity| entity.id.clone())
            .collect::<Vec<_>>();
        for entity_id in active_ids {
            if let Some(entity) = self.entities.get_mut(&entity_id) {
                for experience in &frame.experiences {
                    let point = entity.push_observation_point(
                        Modality::Memory,
                        format!("experience:{}", experience.id),
                        experience.salience,
                        frame.t_ms,
                    );
                    let cluster = entity.upsert_cluster(
                        Modality::Memory,
                        format!("experience:{}", experience.id),
                        point,
                        experience.salience,
                    );
                    entity.bind_with_object_cluster(
                        cluster,
                        BindingRelation::PredictsSameFutureEvents,
                        experience.salience,
                        frame.t_ms,
                    );
                }
                for impression in &frame.impressions {
                    entity.add_text_label(
                        impression.text.clone(),
                        impression.confidence,
                        frame.t_ms,
                    );
                }
            }
        }
    }

    /// Decay confidence of all entities.  Entities whose confidence falls
    /// below threshold transition to `Occluded` or `Vanished`.
    fn decay(&mut self, ticks: u64, now_ms: u64) {
        let factor = ENTITY_CONFIDENCE_DECAY_PER_TICK.powi(ticks as i32);
        let hypothesis_factor = HYPOTHESIS_CONFIDENCE_DECAY_PER_TICK.powi(ticks as i32);
        for entity in self.entities.values_mut() {
            if entity.lifecycle == EntityLifecycleState::Vanished {
                continue;
            }
            entity.confidence = (entity.confidence * factor).clamp(0.0, 1.0);
            entity.lifecycle = if entity.confidence < ENTITY_VANISH_THRESHOLD {
                EntityLifecycleState::Vanished
            } else if entity.confidence < ENTITY_OCCLUDE_THRESHOLD {
                EntityLifecycleState::Occluded
            } else {
                EntityLifecycleState::Active
            };
            if entity.lifecycle == EntityLifecycleState::Vanished {
                entity.constellation.state = EntityConstellationState::Vanished;
            }
            entity.decay_bindings((1.0 - factor).clamp(0.0, 1.0));
        }
        for hypothesis in self.tracking_hypotheses.values_mut() {
            if matches!(
                hypothesis.state,
                HypothesisState::Promoted | HypothesisState::Rejected | HypothesisState::Expired
            ) {
                continue;
            }
            hypothesis.confidence = (hypothesis.confidence * hypothesis_factor).clamp(0.0, 1.0);
            let stale_ms = now_ms.saturating_sub(hypothesis.last_updated_ms);
            if hypothesis.confidence < 0.25 && stale_ms >= HYPOTHESIS_STALE_MS {
                hypothesis.state = HypothesisState::Expired;
            } else if hypothesis.state == HypothesisState::NeedsReview
                && stale_ms >= HYPOTHESIS_REVIEW_STALE_MS
            {
                hypothesis.state = HypothesisState::Expired;
            }
        }
    }

    pub fn merge_entities(&mut self, primary_id: &str, secondary_id: &str) -> bool {
        if primary_id == secondary_id {
            return false;
        }
        let Some(mut secondary) = self.entities.remove(secondary_id) else {
            return false;
        };
        let Some(primary) = self.entities.get_mut(primary_id) else {
            self.entities.insert(secondary_id.to_string(), secondary);
            return false;
        };
        primary.observation_count = primary
            .observation_count
            .saturating_add(secondary.observation_count);
        primary.confidence = primary.confidence.max(secondary.confidence);
        for label in secondary.labels.drain(..) {
            if !primary.labels.contains(&label) {
                primary.labels.push(label);
            }
        }
        primary
            .constellation
            .merged_entity_ids
            .push(secondary_id.to_string());
        primary.constellation.state = EntityConstellationState::Merged;
        true
    }

    pub fn split_entity(&mut self, entity_id: &str, suffix: &str) -> Option<String> {
        let mut child = self.entities.get(entity_id)?.clone();
        let child_id = format!("{entity_id}:split:{}", stable_slug(suffix));
        child.id = child_id.clone();
        child.confidence = (child.confidence * 0.6).clamp(0.0, 1.0);
        child.constellation.state = EntityConstellationState::Split;
        if let Some(parent) = self.entities.get_mut(entity_id) {
            parent.constellation.split_entity_ids.push(child_id.clone());
            parent.constellation.state = EntityConstellationState::Split;
        }
        self.entities.insert(child_id.clone(), child);
        Some(child_id)
    }

    /// Build a summary report for dashboard/API consumption.
    pub fn report(&self) -> EntityMemoryReport {
        let total_entities = self.entities.len();
        let active_entities = self
            .entities
            .values()
            .filter(|e| e.lifecycle == EntityLifecycleState::Active)
            .count();
        let occluded_entities = self
            .entities
            .values()
            .filter(|e| e.lifecycle == EntityLifecycleState::Occluded)
            .count();
        let vanished_entities = self
            .entities
            .values()
            .filter(|e| e.lifecycle == EntityLifecycleState::Vanished)
            .count();

        let mut sorted: Vec<&EntityHypothesis> = self.entities.values().collect();
        sorted.sort_by(|a, b| {
            // Active before occluded before vanished, then by confidence descending.
            let state_order = |e: &EntityHypothesis| match e.lifecycle {
                EntityLifecycleState::Active => 0u8,
                EntityLifecycleState::Occluded => 1,
                EntityLifecycleState::Vanished => 2,
            };
            state_order(a).cmp(&state_order(b)).then_with(|| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
        sorted.truncate(20);
        let top_entities = sorted
            .iter()
            .map(|e| EntityHypothesisSummary::from(*e))
            .collect();

        let all_candidates = self
            .binding_candidates
            .iter()
            .cloned()
            .chain(
                self.entities
                    .values()
                    .flat_map(|entity| entity.constellation.binding_candidates.iter().cloned()),
            )
            .collect::<Vec<_>>();
        let accepted_binding_candidates = all_candidates
            .iter()
            .filter(|candidate| candidate.decision == BindingDecision::Accept)
            .cloned()
            .collect();
        let ambiguous_binding_candidates = all_candidates
            .iter()
            .filter(|candidate| {
                matches!(
                    candidate.decision,
                    BindingDecision::HoldAmbiguous
                        | BindingDecision::AskHuman
                        | BindingDecision::CollectMoreEvidence
                )
            })
            .cloned()
            .collect();
        let rejected_binding_candidates = all_candidates
            .iter()
            .filter(|candidate| candidate.decision == BindingDecision::Reject)
            .cloned()
            .collect();
        let active_tracking_hypotheses = self
            .tracking_hypotheses
            .values()
            .filter(|hypothesis| {
                matches!(
                    hypothesis.state,
                    HypothesisState::Active | HypothesisState::Winning | HypothesisState::Losing
                )
            })
            .cloned()
            .collect();
        let review_tracking_hypotheses = self
            .tracking_hypotheses
            .values()
            .filter(|hypothesis| hypothesis.state == HypothesisState::NeedsReview)
            .cloned()
            .collect();
        let promoted_tracking_hypotheses = self
            .tracking_hypotheses
            .values()
            .filter(|hypothesis| hypothesis.state == HypothesisState::Promoted)
            .cloned()
            .collect();
        let expired_tracking_hypotheses = self
            .tracking_hypotheses
            .values()
            .filter(|hypothesis| hypothesis.state == HypothesisState::Expired)
            .cloned()
            .collect();

        EntityMemoryReport {
            total_entities,
            active_entities,
            occluded_entities,
            vanished_entities,
            active_tracking_hypotheses,
            review_tracking_hypotheses,
            promoted_tracking_hypotheses,
            expired_tracking_hypotheses,
            accepted_binding_candidates,
            ambiguous_binding_candidates,
            rejected_binding_candidates,
            top_entities,
        }
    }
}

fn base_cross_modal_evidence(
    context: &BindingContext,
    left: &DiscoveredCluster,
    right: &DiscoveredCluster,
) -> Vec<BindingEvidence> {
    let mut evidence = Vec::new();
    if temporally_compatible(context, left, right) {
        let delta_ms = cluster_time_delta_ms(left, right);
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::TemporalOverlap,
            score: (1.0 - delta_ms as f32 / context.time_window_ms.max(1) as f32).clamp(0.0, 1.0),
            reason: format!("{} and {} occurred within {delta_ms} ms", left.id, right.id),
        });
    }
    if source_frame_matches(context, left, right) {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::ProjectionAgreement,
            score: 0.55,
            reason: "clusters share a source frame context".to_string(),
        });
    }
    if let Some(distance) = pose_distance_m(left, right) {
        if distance <= 0.75 {
            evidence.push(BindingEvidence {
                kind: BindingEvidenceKind::SpatialOverlap,
                score: (1.0 - distance / 0.75).clamp(0.0, 1.0),
                reason: format!("cluster poses are within {distance:.2} m"),
            });
        }
    }
    evidence
}

fn candidate_from_evidence(
    left: &DiscoveredCluster,
    right: &DiscoveredCluster,
    relation: BindingRelation,
    evidence: Vec<BindingEvidence>,
    fallback_reason: &str,
) -> BindingCandidate {
    let has_human_confirmation = evidence
        .iter()
        .any(|item| item.kind == BindingEvidenceKind::HumanConfirmed);
    let has_hard_contradiction = evidence
        .iter()
        .any(|item| item.kind == BindingEvidenceKind::Contradiction);
    let has_conflict = evidence.iter().any(|item| {
        matches!(
            item.kind,
            BindingEvidenceKind::Contradiction | BindingEvidenceKind::SimultaneousConflict
        )
    });
    let independent_positive_kinds = evidence
        .iter()
        .filter(|item| {
            !matches!(
                item.kind,
                BindingEvidenceKind::Contradiction
                    | BindingEvidenceKind::SimultaneousConflict
                    | BindingEvidenceKind::VectorSimilarity
                    | BindingEvidenceKind::LlmSuggested
            )
        })
        .map(|item| binding_evidence_kind_rank(&item.kind))
        .collect::<BTreeSet<_>>()
        .len();
    let mean_score = if evidence.is_empty() {
        0.0
    } else {
        evidence
            .iter()
            .map(|item| item.score.clamp(0.0, 1.0))
            .sum::<f32>()
            / evidence.len() as f32
    };
    let mut confidence = if has_human_confirmation {
        mean_score.max(0.9)
    } else {
        (mean_score * (independent_positive_kinds as f32 / 3.0).clamp(0.25, 1.0)).clamp(0.0, 1.0)
    };
    if has_conflict {
        confidence *= 0.35;
    }

    let (decision, reason) = if has_hard_contradiction {
        (
            BindingDecision::Reject,
            "candidate contains contradictory cross-modal evidence".to_string(),
        )
    } else if has_conflict {
        (
            BindingDecision::HoldAmbiguous,
            "candidate is plausible but has competing cross-modal evidence".to_string(),
        )
    } else if has_human_confirmation {
        (
            BindingDecision::Accept,
            "candidate has trusted human/source confirmation".to_string(),
        )
    } else if independent_positive_kinds >= 2 {
        (
            BindingDecision::Accept,
            "candidate has at least two independent cross-modal evidence types".to_string(),
        )
    } else if evidence
        .iter()
        .any(|item| item.kind == BindingEvidenceKind::LlmSuggested)
        && independent_positive_kinds == 0
    {
        (
            BindingDecision::CollectMoreEvidence,
            "LLM suggestion alone is not enough to bind clusters".to_string(),
        )
    } else if evidence.is_empty() {
        (
            BindingDecision::CollectMoreEvidence,
            fallback_reason.to_string(),
        )
    } else {
        (
            BindingDecision::CollectMoreEvidence,
            "candidate needs more independent evidence before admission".to_string(),
        )
    };

    BindingCandidate {
        left_cluster_id: left.id.clone(),
        right_cluster_id: right.id.clone(),
        relation,
        evidence,
        confidence: confidence.clamp(0.0, 1.0),
        decision,
        reason,
    }
}

fn proposal_candidate_from_evidence(
    left: &DiscoveredCluster,
    right: &DiscoveredCluster,
    relation: BindingRelation,
    evidence: Vec<BindingEvidence>,
    fallback_reason: &str,
) -> BindingCandidate {
    let mut candidate = candidate_from_evidence(left, right, relation, evidence, fallback_reason);
    if candidate.decision == BindingDecision::Accept {
        candidate.decision = BindingDecision::CollectMoreEvidence;
        candidate.reason =
            "candidate is proposal-only; conservative binding admission must accept it".to_string();
    }
    candidate
}

fn temporally_compatible(
    context: &BindingContext,
    left: &DiscoveredCluster,
    right: &DiscoveredCluster,
) -> bool {
    cluster_time_delta_ms(left, right) <= context.time_window_ms
}

fn cluster_time_delta_ms(left: &DiscoveredCluster, right: &DiscoveredCluster) -> u64 {
    if left.last_seen_ms < right.first_seen_ms {
        right.first_seen_ms.saturating_sub(left.last_seen_ms)
    } else if right.last_seen_ms < left.first_seen_ms {
        left.first_seen_ms.saturating_sub(right.last_seen_ms)
    } else {
        0
    }
}

fn source_frame_matches(
    context: &BindingContext,
    left: &DiscoveredCluster,
    right: &DiscoveredCluster,
) -> bool {
    if left.source_frame_id.is_some()
        && right.source_frame_id.is_some()
        && left.source_frame_id == right.source_frame_id
    {
        return true;
    }
    context.source_frame_id.as_ref().is_some_and(|frame| {
        left.source_frame_id.as_ref() == Some(frame)
            || right.source_frame_id.as_ref() == Some(frame)
    })
}

fn add_recent_cooccurrence(
    context: &BindingContext,
    left: &DiscoveredCluster,
    right: &DiscoveredCluster,
    evidence: &mut Vec<BindingEvidence>,
) {
    if context.recent_clusters.contains(&left.id) && context.recent_clusters.contains(&right.id) {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::RepeatedCooccurrence,
            score: 0.7,
            reason: "both clusters appeared in recent binding context".to_string(),
        });
    }
    if !left.feature_ids.is_empty()
        && !right.feature_ids.is_empty()
        && left
            .feature_ids
            .iter()
            .any(|id| context.recent_features.contains(id))
        && right
            .feature_ids
            .iter()
            .any(|id| context.recent_features.contains(id))
    {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::RepeatedCooccurrence,
            score: 0.7,
            reason: "both clusters reference recently observed features".to_string(),
        });
    }
}

fn add_repetition_evidence(
    left: &DiscoveredCluster,
    right: &DiscoveredCluster,
    evidence: &mut Vec<BindingEvidence>,
) {
    let repeats =
        metadata_u64(left, "cooccurrence_count").max(metadata_u64(right, "cooccurrence_count"));
    if repeats >= 2 {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::RepeatedCooccurrence,
            score: (repeats as f32 / 5.0).clamp(0.0, 1.0),
            reason: format!("clusters have repeated together in {repeats} observations"),
        });
    }
}

fn add_label_support(
    left: &DiscoveredCluster,
    right: &DiscoveredCluster,
    evidence: &mut Vec<BindingEvidence>,
) {
    let left_label = metadata_string(left, "label");
    let right_label = metadata_string(right, "label");
    if left_label.is_some() && left_label == right_label {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::HumanConfirmed,
            score: 0.85,
            reason: "clusters share a supporting label".to_string(),
        });
    }
}

fn projection_error_px(left: &DiscoveredCluster, right: &DiscoveredCluster) -> Option<f32> {
    let left_x = metadata_f32(left, "image_x");
    let left_y = metadata_f32(left, "image_y");
    let right_x =
        metadata_f32(right, "projected_image_x").or_else(|| metadata_f32(right, "image_x"));
    let right_y =
        metadata_f32(right, "projected_image_y").or_else(|| metadata_f32(right, "image_y"));
    left_x
        .zip(left_y)
        .zip(right_x.zip(right_y))
        .map(|((lx, ly), (rx, ry))| ((lx - rx).powi(2) + (ly - ry).powi(2)).sqrt())
        .or_else(|| {
            let right_x = metadata_f32(right, "image_x");
            let right_y = metadata_f32(right, "image_y");
            let left_x =
                metadata_f32(left, "projected_image_x").or_else(|| metadata_f32(left, "image_x"));
            let left_y =
                metadata_f32(left, "projected_image_y").or_else(|| metadata_f32(left, "image_y"));
            left_x
                .zip(left_y)
                .zip(right_x.zip(right_y))
                .map(|((lx, ly), (rx, ry))| ((lx - rx).powi(2) + (ly - ry).powi(2)).sqrt())
        })
}

fn pose_distance_m(left: &DiscoveredCluster, right: &DiscoveredCluster) -> Option<f32> {
    left.estimated_pose
        .zip(right.estimated_pose)
        .map(|(left, right)| {
            ((left.x_m - right.x_m).powi(2) + (left.y_m - right.y_m).powi(2)).sqrt()
        })
}

fn lag_score(lag_ms: u64, min_ms: u64, max_ms: u64) -> f32 {
    if lag_ms < min_ms || lag_ms > max_ms {
        return 0.0;
    }
    let midpoint = (min_ms + max_ms) as f32 / 2.0;
    let half_span = (max_ms.saturating_sub(min_ms)).max(1) as f32 / 2.0;
    (1.0 - ((lag_ms as f32 - midpoint).abs() / half_span)).clamp(0.1, 1.0)
}

fn metadata_f32(cluster: &DiscoveredCluster, key: &str) -> Option<f32> {
    cluster
        .metadata
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .map(|value| value as f32)
}

fn metadata_u64(cluster: &DiscoveredCluster, key: &str) -> u64 {
    cluster
        .metadata
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_default()
}

fn metadata_bool(cluster: &DiscoveredCluster, key: &str) -> bool {
    cluster
        .metadata
        .get(key)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

fn metadata_string(cluster: &DiscoveredCluster, key: &str) -> Option<String> {
    cluster
        .metadata
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

fn qualify_binding_candidate(
    entity: &EntityHypothesis,
    artifact: &VectorArtifact,
    kind: VectorBindingKind,
    left_cluster_id: String,
    right_cluster_id: String,
    t_ms: u64,
    cell_key: Option<PlaceCellKey>,
    plausible_count: usize,
    current_object_observed: bool,
) -> BindingCandidate {
    let mut evidence = Vec::new();
    evidence.push(BindingEvidence {
        kind: BindingEvidenceKind::VectorSimilarity,
        score: 0.45,
        reason: "vector artifact proposes a possible cross-modal correspondence".to_string(),
    });

    if artifact.source_id.as_deref() == Some(entity.id.as_str()) {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::HumanConfirmed,
            score: 1.0,
            reason: "vector source explicitly names this entity".to_string(),
        });
    } else if artifact
        .source_id
        .as_deref()
        .is_some_and(|source_id| source_id.starts_with("entity:"))
    {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::Contradiction,
            score: 1.0,
            reason: format!(
                "vector source names {}, not {}",
                artifact.source_id.as_deref().unwrap_or("unknown"),
                entity.id
            ),
        });
    }
    if t_ms.saturating_sub(entity.last_seen_ms) <= 1_000 {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::TemporalOverlap,
            score: 0.75,
            reason: "entity was observed in the current temporal window".to_string(),
        });
    }
    if cell_key
        .map(|key| entity.location_cells.contains(&key))
        .unwrap_or(false)
    {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::SpatialOverlap,
            score: 0.75,
            reason: "entity has a compatible current map cell".to_string(),
        });
    }
    if current_object_observed {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::ProjectionAgreement,
            score: 0.7,
            reason: "a current object observation anchors this entity".to_string(),
        });
    }
    if plausible_count == 1 {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::SingleCandidateContext,
            score: 0.65,
            reason: "only one plausible entity matched this vector context".to_string(),
        });
    } else if plausible_count > 1 {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::SimultaneousConflict,
            score: 0.8,
            reason: match kind {
                VectorBindingKind::Face => {
                    "face vector close to multiple active person entities".to_string()
                }
                VectorBindingKind::Voice => {
                    "voice observed while multiple person hypotheses are active".to_string()
                }
                VectorBindingKind::Scene => {
                    "scene vector has multiple spatially plausible entities".to_string()
                }
            },
        });
    }
    if entity.constellation.binding_edges.iter().any(|edge| {
        edge.left_cluster_id == right_cluster_id || edge.right_cluster_id == right_cluster_id
    }) {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::RepeatedCooccurrence,
            score: 0.8,
            reason: "prior binding history supports this correspondence".to_string(),
        });
    }

    let has_human_confirmation = evidence
        .iter()
        .any(|item| item.kind == BindingEvidenceKind::HumanConfirmed);
    let has_conflict = evidence.iter().any(|item| {
        matches!(
            item.kind,
            BindingEvidenceKind::Contradiction | BindingEvidenceKind::SimultaneousConflict
        )
    });
    let has_hard_contradiction = evidence
        .iter()
        .any(|item| item.kind == BindingEvidenceKind::Contradiction);
    let independent_positive_kinds = evidence
        .iter()
        .filter(|item| {
            !matches!(
                item.kind,
                BindingEvidenceKind::Contradiction
                    | BindingEvidenceKind::SimultaneousConflict
                    | BindingEvidenceKind::VectorSimilarity
            )
        })
        .map(|item| binding_evidence_kind_rank(&item.kind))
        .collect::<BTreeSet<_>>()
        .len();
    let mean_score = if evidence.is_empty() {
        0.0
    } else {
        evidence
            .iter()
            .map(|item| item.score.clamp(0.0, 1.0))
            .sum::<f32>()
            / evidence.len() as f32
    };
    let mut confidence = if has_human_confirmation {
        mean_score.max(0.9)
    } else {
        (mean_score * (independent_positive_kinds as f32 / 3.0).clamp(0.25, 1.0)).clamp(0.0, 1.0)
    };
    if has_conflict {
        confidence *= 0.35;
    }

    let (decision, reason) = if has_hard_contradiction {
        (
            BindingDecision::Reject,
            "candidate contradicts explicit entity source evidence".to_string(),
        )
    } else if has_human_confirmation {
        (
            BindingDecision::Accept,
            "human-confirmed or explicit source binding".to_string(),
        )
    } else if has_conflict {
        (
            BindingDecision::HoldAmbiguous,
            match kind {
                VectorBindingKind::Face => "face vector close to multiple active person entities",
                VectorBindingKind::Voice => {
                    "voice observed while multiple person hypotheses active"
                }
                VectorBindingKind::Scene => {
                    "scene vector active but multiple spatially compatible entities exist"
                }
            }
            .to_string(),
        )
    } else if independent_positive_kinds >= 2 {
        (
            BindingDecision::Accept,
            "candidate has at least two independent supporting evidence types".to_string(),
        )
    } else if evidence.len() == 1 {
        (
            BindingDecision::CollectMoreEvidence,
            "single vector similarity without supporting temporal/spatial evidence".to_string(),
        )
    } else {
        (
            BindingDecision::CollectMoreEvidence,
            "projection agreement missing or evidence is not yet independent".to_string(),
        )
    };

    BindingCandidate {
        left_cluster_id,
        right_cluster_id,
        relation: match kind {
            VectorBindingKind::Face | VectorBindingKind::Voice => BindingRelation::LikelySameEntity,
            VectorBindingKind::Scene => BindingRelation::ProjectsTo,
        },
        evidence,
        confidence: confidence.clamp(0.0, 1.0),
        decision,
        reason,
    }
}

fn vector_cluster_id(kind: VectorBindingKind, point_id: &str) -> String {
    let key = match kind {
        VectorBindingKind::Face => format!("face:{point_id}"),
        VectorBindingKind::Voice => format!("voice:{point_id}"),
        VectorBindingKind::Scene => format!("scene:{point_id}"),
    };
    let modality = match kind {
        VectorBindingKind::Face => Modality::Vision,
        VectorBindingKind::Voice => Modality::Audio,
        VectorBindingKind::Scene => Modality::Depth,
    };
    format!("cluster:{}:{}", modality.as_str(), stable_slug(&key))
}

fn binding_evidence_kind_rank(kind: &BindingEvidenceKind) -> u8 {
    match kind {
        BindingEvidenceKind::TemporalOverlap => 1,
        BindingEvidenceKind::SpatialOverlap => 2,
        BindingEvidenceKind::VectorSimilarity => 3,
        BindingEvidenceKind::ProjectionAgreement => 4,
        BindingEvidenceKind::PoseAgreement => 5,
        BindingEvidenceKind::RepeatedCooccurrence => 6,
        BindingEvidenceKind::SingleCandidateContext => 7,
        BindingEvidenceKind::HumanConfirmed => 8,
        BindingEvidenceKind::LlmSuggested => 9,
        BindingEvidenceKind::Contradiction => 10,
        BindingEvidenceKind::SimultaneousConflict => 11,
    }
}

fn tracking_kind_from_vector(kind: VectorBindingKind) -> TrackingHypothesisKind {
    match kind {
        VectorBindingKind::Face => TrackingHypothesisKind::FaceIdentity,
        VectorBindingKind::Voice => TrackingHypothesisKind::VoiceIdentity,
        VectorBindingKind::Scene => TrackingHypothesisKind::PlaceMatch,
    }
}

fn tracking_kind_slug(kind: &TrackingHypothesisKind) -> &'static str {
    match kind {
        TrackingHypothesisKind::FaceIdentity => "face-identity",
        TrackingHypothesisKind::VoiceIdentity => "voice-identity",
        TrackingHypothesisKind::CrossModalBinding => "cross-modal",
        TrackingHypothesisKind::PlaceMatch => "place",
        TrackingHypothesisKind::ObjectContinuity => "object-continuity",
        TrackingHypothesisKind::Other => "other",
    }
}

fn tracking_family_id(kind: VectorBindingKind, observation_id: &str) -> String {
    format!(
        "{}:{}",
        tracking_kind_slug(&tracking_kind_from_vector(kind)),
        observation_id
    )
}

fn binding_candidate_id(candidate: &BindingCandidate) -> String {
    format!(
        "candidate:{}:{}:{}",
        stable_slug(&candidate.left_cluster_id),
        stable_slug(&candidate.right_cluster_id),
        binding_relation_label_for_id(&candidate.relation)
    )
}

fn binding_relation_label_for_id(relation: &BindingRelation) -> &'static str {
    match relation {
        BindingRelation::CooccursInTime => "time",
        BindingRelation::CooccursInEstimatedSpace => "space",
        BindingRelation::MovesTogether => "moves",
        BindingRelation::PredictsSameFutureEvents => "future",
        BindingRelation::NamedBy => "named-by",
        BindingRelation::ProjectsTo => "projects-to",
        BindingRelation::HasColorAtPose => "color-pose",
        BindingRelation::LikelySameEntity => "same-entity",
        BindingRelation::ExplainsOutcome => "outcome",
        BindingRelation::Contradicts => "contradicts",
        BindingRelation::RequiresReview => "review",
    }
}

fn has_hard_contradiction(evidence: &[BindingEvidence]) -> bool {
    evidence
        .iter()
        .any(|item| item.kind == BindingEvidenceKind::Contradiction)
}

fn score_hypothesis_evidence(evidence: &[BindingEvidence], repeated_observations: f32) -> f32 {
    if evidence.is_empty() {
        return 0.0;
    }
    let human_confirmed = evidence
        .iter()
        .any(|item| item.kind == BindingEvidenceKind::HumanConfirmed);
    let positive = evidence
        .iter()
        .filter(|item| {
            !matches!(
                item.kind,
                BindingEvidenceKind::Contradiction | BindingEvidenceKind::SimultaneousConflict
            )
        })
        .map(|item| item.score.clamp(0.0, 1.0))
        .sum::<f32>();
    let positive_count = evidence
        .iter()
        .filter(|item| {
            !matches!(
                item.kind,
                BindingEvidenceKind::Contradiction | BindingEvidenceKind::SimultaneousConflict
            )
        })
        .count()
        .max(1) as f32;
    let independent_positive_kinds = evidence
        .iter()
        .filter(|item| {
            !matches!(
                item.kind,
                BindingEvidenceKind::Contradiction
                    | BindingEvidenceKind::SimultaneousConflict
                    | BindingEvidenceKind::VectorSimilarity
                    | BindingEvidenceKind::LlmSuggested
            )
        })
        .map(|item| binding_evidence_kind_rank(&item.kind))
        .collect::<BTreeSet<_>>()
        .len() as f32;
    let contradiction_count = evidence
        .iter()
        .filter(|item| {
            matches!(
                item.kind,
                BindingEvidenceKind::Contradiction | BindingEvidenceKind::SimultaneousConflict
            )
        })
        .count() as f32;
    let repetition_bonus = ((repeated_observations - 1.0).max(0.0) * 0.08).min(0.18);
    let independence_bonus = (independent_positive_kinds * 0.08).min(0.24);
    let mut score = positive / positive_count + repetition_bonus + independence_bonus;
    if human_confirmed {
        score = score.max(0.92);
    }
    score -= contradiction_count * 0.18;
    score.clamp(0.0, 1.0)
}

fn cluster_ids_from_observation(
    observation: &ConstellationObservation,
    accepted_bindings: &[&BindingCandidate],
) -> Vec<String> {
    let mut ids = accepted_bindings
        .iter()
        .flat_map(|candidate| {
            [
                candidate.left_cluster_id.clone(),
                candidate.right_cluster_id.clone(),
            ]
        })
        .chain(
            observation
                .clusters
                .iter()
                .map(|cluster| cluster.id.clone()),
        )
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    ids.sort();
    ids
}

fn merge_constellation_observation(
    constellation: &mut Constellation,
    observation: &ConstellationObservation,
    member_cluster_ids: &[String],
    member_binding_ids: &[String],
) {
    merge_unique(&mut constellation.member_cluster_ids, member_cluster_ids);
    merge_unique(&mut constellation.member_binding_ids, member_binding_ids);
    let feature_ids = observation
        .clusters
        .iter()
        .flat_map(|cluster| cluster.feature_ids.iter().copied())
        .collect::<Vec<_>>();
    merge_unique(&mut constellation.supporting_feature_ids, &feature_ids);
    merge_unique(
        &mut constellation.supporting_entity_ids,
        &observation.active_entity_ids,
    );
    merge_unique(
        &mut constellation.supporting_place_cells,
        &observation.place_cells,
    );
    merge_unique(&mut constellation.notes, &observation.llm_notes);
    constellation.last_seen_ms = observation.t_ms;
    constellation.evidence_count = constellation.evidence_count.saturating_add(1);
    constellation.prediction_value = (constellation.prediction_value * 0.75
        + observation.prediction_value * 0.25)
        .clamp(0.0, 1.0);
    if constellation.kind_hint.is_none() {
        constellation.kind_hint =
            infer_constellation_kind(&observation.clusters).map(|kind| kind.as_str().to_string());
    }
}

fn refresh_constellation_scores(
    constellation: &mut Constellation,
    observation: &ConstellationObservation,
    config: &ConstellationEngineConfig,
) {
    let accepted_bindings = observation
        .accepted_bindings
        .iter()
        .filter(|candidate| candidate.decision == BindingDecision::Accept)
        .collect::<Vec<_>>();
    let positive_binding_score = if accepted_bindings.is_empty() {
        0.0
    } else {
        accepted_bindings
            .iter()
            .map(|candidate| candidate.confidence.clamp(0.0, 1.0))
            .sum::<f32>()
            / accepted_bindings.len() as f32
    };
    let recurrence_score = (constellation.evidence_count as f32
        / config.min_evidence_for_stable.max(1) as f32)
        .clamp(0.0, 1.0);
    let cluster_score = (constellation.member_cluster_ids.len() as f32
        / config.min_clusters_for_stable.max(1) as f32)
        .clamp(0.0, 1.0);
    let binding_score = (constellation.member_binding_ids.len() as f32
        / config.min_bindings_for_stable.max(1) as f32)
        .clamp(0.0, 1.0);
    let contradiction_count = observation
        .accepted_bindings
        .iter()
        .filter(|candidate| binding_has_conflict(candidate))
        .count();
    let contradiction_penalty = (contradiction_count as f32 * 0.25).min(0.65);

    constellation.stability =
        (recurrence_score * 0.5 + binding_score * 0.3 + cluster_score * 0.2).clamp(0.0, 1.0);
    constellation.confidence = (positive_binding_score * 0.45
        + constellation.stability * 0.35
        + constellation.prediction_value * 0.2
        - contradiction_penalty)
        .clamp(0.0, 1.0);

    if evidence_suggests_split(observation) {
        constellation.state = ConstellationState::SplitNeeded;
        return;
    }
    if contradiction_count > 0 {
        constellation.state = ConstellationState::Ambiguous;
        return;
    }
    let promotable = constellation.member_cluster_ids.len() >= config.min_clusters_for_stable
        && constellation.member_binding_ids.len() >= config.min_bindings_for_stable
        && constellation.evidence_count >= config.min_evidence_for_stable
        && constellation.confidence >= config.promotion_confidence_threshold
        && constellation.prediction_value >= config.min_prediction_value_for_stable;
    constellation.state = if promotable {
        ConstellationState::Stable
    } else {
        ConstellationState::Candidate
    };
}

fn binding_has_conflict(candidate: &BindingCandidate) -> bool {
    candidate.evidence.iter().any(|evidence| {
        matches!(
            evidence.kind,
            BindingEvidenceKind::Contradiction | BindingEvidenceKind::SimultaneousConflict
        )
    })
}

fn evidence_suggests_split(observation: &ConstellationObservation) -> bool {
    observation.accepted_bindings.iter().any(|candidate| {
        candidate
            .evidence
            .iter()
            .any(|evidence| evidence.kind == BindingEvidenceKind::SimultaneousConflict)
    }) || observation.llm_notes.iter().any(|note| {
        let note = note.to_ascii_lowercase();
        note.contains("split")
            || note.contains("fused")
            || note.contains("fusion")
            || note.contains("two patterns")
    })
}

fn infer_constellation_kind(clusters: &[DiscoveredCluster]) -> Option<ConstellationKind> {
    let kinds = clusters
        .iter()
        .map(|cluster| cluster.kind.clone())
        .collect::<BTreeSet<_>>();
    if kinds.contains(&DiscoveredClusterKind::Face) || kinds.contains(&DiscoveredClusterKind::Voice)
    {
        Some(ConstellationKind::Person)
    } else if kinds.contains(&DiscoveredClusterKind::Action)
        || kinds.contains(&DiscoveredClusterKind::Outcome)
        || kinds.contains(&DiscoveredClusterKind::BodyState)
    {
        Some(ConstellationKind::ActionOutcome)
    } else if kinds.contains(&DiscoveredClusterKind::Place) {
        Some(ConstellationKind::Place)
    } else if kinds.contains(&DiscoveredClusterKind::Object)
        || kinds.contains(&DiscoveredClusterKind::Geometry)
        || kinds.contains(&DiscoveredClusterKind::RgbImage)
    {
        Some(ConstellationKind::Object)
    } else {
        None
    }
}

fn overlap_score(matched: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        (matched as f32 / total as f32).clamp(0.0, 1.0)
    }
}

fn stale_penalty(age_ms: u64, stale_after_ms: u64) -> f32 {
    if stale_after_ms == 0 || age_ms <= stale_after_ms {
        0.0
    } else {
        ((age_ms - stale_after_ms) as f32 / (stale_after_ms * 4) as f32).clamp(0.0, 0.6)
    }
}

fn intersection_count<T>(left: &[T], right: &[T]) -> usize
where
    T: Ord + Clone,
{
    let left = left.iter().cloned().collect::<BTreeSet<_>>();
    let right = right.iter().cloned().collect::<BTreeSet<_>>();
    left.intersection(&right).count()
}

fn merge_unique<T>(target: &mut Vec<T>, incoming: &[T])
where
    T: Ord + Clone,
{
    let mut seen = target.iter().cloned().collect::<BTreeSet<_>>();
    for item in incoming {
        if seen.insert(item.clone()) {
            target.push(item.clone());
        }
    }
    target.sort();
}

fn association_edge_id(from_id: &str, to_id: &str, relation: &AssociationRelation) -> String {
    format!(
        "association:{}:{}:{}",
        association_relation_slug(relation),
        stable_slug(from_id),
        stable_slug(to_id)
    )
}

fn association_relation_slug(relation: &AssociationRelation) -> &'static str {
    match relation {
        AssociationRelation::CoOccursWith => "co-occurs-with",
        AssociationRelation::Predicts => "predicts",
        AssociationRelation::Follows => "follows",
        AssociationRelation::Suppresses => "suppresses",
        AssociationRelation::Contradicts => "contradicts",
        AssociationRelation::Explains => "explains",
        AssociationRelation::Enables => "enables",
        AssociationRelation::Prevents => "prevents",
        AssociationRelation::PartOf => "part-of",
    }
}

fn dedupe_association_items(items: Vec<AssociationItem>) -> Vec<AssociationItem> {
    let mut by_id = BTreeMap::<String, AssociationItem>::new();
    for item in items {
        by_id
            .entry(item.id.clone())
            .and_modify(|existing| {
                existing.confidence = existing.confidence.max(item.confidence);
            })
            .or_insert(item);
    }
    by_id.into_values().collect()
}

fn canonical_association_pair<'a>(
    left: &'a AssociationItem,
    right: &'a AssociationItem,
) -> (&'a str, &'a str) {
    if left.id <= right.id {
        (&left.id, &right.id)
    } else {
        (&right.id, &left.id)
    }
}

fn sequence_relation(
    to: &AssociationItem,
    lag_ms: u64,
    config: &AssociationLearningConfig,
) -> AssociationRelation {
    if matches!(
        to.kind,
        AssociationItemKind::Outcome
            | AssociationItemKind::Prediction
            | AssociationItemKind::Surprise
            | AssociationItemKind::BodyState
    ) && lag_ms <= config.long_sequence_window_ms
    {
        AssociationRelation::Predicts
    } else if lag_ms <= config.short_sequence_window_ms {
        AssociationRelation::Follows
    } else {
        AssociationRelation::Follows
    }
}

fn lag_score_for_association(lag_ms: u64) -> f32 {
    match lag_ms {
        0..=500 => 1.0,
        501..=2_000 => 0.8,
        2_001..=10_000 => 0.55,
        _ => 0.2,
    }
}

fn approximate_mutual_information(p_b: f32, p_b_given_a: f32) -> f32 {
    let p_b = p_b.clamp(0.001, 0.999);
    let p_b_given_a = p_b_given_a.clamp(0.001, 0.999);
    if p_b_given_a <= p_b {
        return 0.0;
    }
    (p_b_given_a * (p_b_given_a / p_b).ln()).clamp(0.0, 1.0)
}

