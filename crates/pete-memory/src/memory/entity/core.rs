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

        let mut detector_observation_keys = BTreeSet::new();
        for detection in &now.objects.detections {
            let Some(label) = detection.labels.first() else {
                continue;
            };
            let observation = ObjectObservation {
                label: label.label.clone(),
                class: ObjectClass::Unknown,
                bearing_rad: ((detection.bbox.x as f32 + detection.bbox.width as f32 * 0.5)
                    / detection.image_width.max(1) as f32
                    - 0.5),
                distance_m: detection.position.as_ref().map(|position| position.depth_m),
                confidence: label.confidence,
                source: if detection.source_stream.contains("kinect") {
                    ObjectObservationSource::Kinect
                } else {
                    ObjectObservationSource::Unknown
                },
            };
            detector_observation_keys.insert((
                observation.label.clone(),
                observation.confidence.to_bits(),
                format!("{:?}", observation.source),
            ));
            let evidence_id = detection
                .track_id
                .as_deref()
                .unwrap_or(&detection.descendant_sensation_id);
            let id = format!("entity:visual-hypothesis:{}", stable_slug(evidence_id));
            if let Some(existing) = self.entities.get_mut(&id) {
                existing.merge_observation(&observation, now.t_ms, cell_key);
            } else {
                let mut hypothesis =
                    EntityHypothesis::from_observation(&observation, now.t_ms, cell_key);
                hypothesis.id = id.clone();
                // A detector label is a mutable hypothesis, never the entity's
                // permanent display identity.
                hypothesis.display_name = None;
                self.entities.insert(id, hypothesis);
            }
        }

        for observation in &now.objects.observations {
            if detector_observation_keys.contains(&(
                observation.label.clone(),
                observation.confidence.to_bits(),
                format!("{:?}", observation.source),
            )) {
                continue;
            }
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

        let mut current_entity_ids = now
            .objects
            .observations
            .iter()
            .filter(|observation| {
                !detector_observation_keys.contains(&(
                    observation.label.clone(),
                    observation.confidence.to_bits(),
                    format!("{:?}", observation.source),
                ))
            })
            .map(|observation| {
                format!(
                    "entity:{}:{}",
                    object_class_slug(&observation.class),
                    stable_slug(&observation.label)
                )
            })
            .collect::<BTreeSet<_>>();
        current_entity_ids.extend(now.objects.detections.iter().map(|detection| {
            let evidence_id = detection
                .track_id
                .as_deref()
                .unwrap_or(&detection.descendant_sensation_id);
            format!("entity:visual-hypothesis:{}", stable_slug(evidence_id))
        }));

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
