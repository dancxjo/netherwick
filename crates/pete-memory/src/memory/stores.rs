#[derive(Clone, Default)]
pub struct InMemoryExperienceStore {
    records: Arc<Mutex<Vec<MemoryRecord>>>,
    places: Arc<Mutex<PlaceMemory>>,
    entities: Arc<Mutex<EntityMemory>>,
}

impl InMemoryExperienceStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn snapshot(&self) -> Vec<MemoryRecord> {
        self.records.lock().expect("memory mutex poisoned").clone()
    }

    pub fn place_snapshot(&self) -> PlaceMemory {
        self.places
            .lock()
            .expect("place memory mutex poisoned")
            .clone()
    }

    pub fn place_report(&self) -> PlaceMemoryReport {
        self.place_snapshot().report()
    }

    pub fn entity_snapshot(&self) -> EntityMemory {
        self.entities
            .lock()
            .expect("entity memory mutex poisoned")
            .clone()
    }

    pub fn entity_report(&self) -> EntityMemoryReport {
        self.entity_snapshot().report()
    }

    pub fn last_social_interaction(&self, person_id: &PersonId) -> Option<InteractionState> {
        self.records
            .lock()
            .expect("memory mutex poisoned")
            .iter()
            .rev()
            .find_map(|record| {
                record
                    .social_world
                    .active_interaction
                    .iter()
                    .chain(record.social_world.recent_interactions.iter().rev())
                    .find(|interaction| interaction.participants.contains(person_id))
                    .cloned()
            })
    }

    pub fn completed_episodes(&self, kind: EpisodeKind) -> Vec<Episode> {
        let mut episodes = BTreeMap::new();
        for record in self.records.lock().expect("memory mutex poisoned").iter() {
            for episode in &record.temporal_context.recently_completed {
                if episode.kind == kind {
                    episodes.insert(episode.episode_id.clone(), episode.clone());
                }
            }
        }
        episodes.into_values().collect()
    }
}

#[async_trait]
impl MemoryStore for InMemoryExperienceStore {
    async fn store(&self, frame: &ExperienceFrame) -> Result<()> {
        self.store_record(memory_record_from_frame(frame)?).await
    }
}

impl InMemoryExperienceStore {
    async fn store_record(&self, record: MemoryRecord) -> Result<()> {
        self.records
            .lock()
            .expect("memory mutex poisoned")
            .push(record);
        Ok(())
    }
}

#[derive(Clone, Default)]
pub struct DurableExperienceStore {
    inner: InMemoryExperienceStore,
    vector_stores: Vec<Arc<dyn VectorStore>>,
    graph_stores: Vec<Arc<dyn GraphStore>>,
}

impl DurableExperienceStore {
    pub fn new(inner: InMemoryExperienceStore) -> Self {
        Self {
            inner,
            vector_stores: Vec::new(),
            graph_stores: Vec::new(),
        }
    }

    pub fn from_env() -> Self {
        let mut store = Self::new(InMemoryExperienceStore::new());
        if let Some(qdrant) = QdrantVectorStore::from_env() {
            store = store.with_vector_store(qdrant);
        }
        if let Some(neo4j) = Neo4jGraphStore::from_env() {
            store = store.with_graph_store(neo4j);
        }
        store
    }

    pub fn with_vector_store(mut self, store: impl VectorStore + 'static) -> Self {
        self.vector_stores.push(Arc::new(store));
        self
    }

    pub fn with_graph_store(mut self, store: impl GraphStore + 'static) -> Self {
        self.graph_stores.push(Arc::new(store));
        self
    }

    pub fn snapshot(&self) -> Vec<MemoryRecord> {
        self.inner.snapshot()
    }

    pub fn place_snapshot(&self) -> PlaceMemory {
        self.inner.place_snapshot()
    }

    pub fn place_report(&self) -> PlaceMemoryReport {
        self.inner.place_report()
    }

    pub fn entity_snapshot(&self) -> EntityMemory {
        self.inner.entity_snapshot()
    }

    pub fn entity_report(&self) -> EntityMemoryReport {
        self.inner.entity_report()
    }
}

#[async_trait]
impl MemoryStore for DurableExperienceStore {
    async fn store(&self, frame: &ExperienceFrame) -> Result<()> {
        let record = memory_record_from_frame(frame)?;
        self.inner.store_record(record.clone()).await?;
        for vector_store in &self.vector_stores {
            if let Err(error) = vector_store.upsert_vectors(&record).await {
                eprintln!("memory vector store write failed: {error:#}");
            }
        }
        for graph_store in &self.graph_stores {
            if let Err(error) = graph_store.upsert_graph(&record).await {
                eprintln!("memory graph store write failed: {error:#}");
            }
        }
        Ok(())
    }
}

#[async_trait]
impl Recall for DurableExperienceStore {
    async fn observe_now(&self, now: &Now) -> Result<()> {
        self.inner.observe_now(now).await
    }

    async fn observe_frame(&self, frame: &ExperienceFrame) -> Result<()> {
        self.inner.observe_frame(frame).await
    }

    async fn observe_transition(&self, transition: &ExperienceTransition) -> Result<()> {
        self.inner.observe_transition(transition).await
    }

    async fn loop_closure_candidates(
        &self,
        query: &RecallQuery,
        min_confidence: f32,
        limit: usize,
    ) -> Result<Vec<PlaceRecognitionCandidate>> {
        self.inner
            .loop_closure_candidates(query, min_confidence, limit)
            .await
    }

    async fn recall(&self, query: RecallQuery) -> Result<RecallBundle> {
        self.inner.recall(query).await
    }
}

#[async_trait]
impl Recall for InMemoryExperienceStore {
    async fn observe_now(&self, now: &Now) -> Result<()> {
        let cell_key = {
            let places = self.places.lock().expect("place memory mutex poisoned");
            Some(places.quantize(now.body.odometry.x_m, now.body.odometry.y_m))
        };
        self.places
            .lock()
            .expect("place memory mutex poisoned")
            .observe_now(now);
        self.entities
            .lock()
            .expect("entity memory mutex poisoned")
            .observe_now(now, cell_key);
        Ok(())
    }

    async fn observe_frame(&self, frame: &ExperienceFrame) -> Result<()> {
        let cell_key = {
            let places = self.places.lock().expect("place memory mutex poisoned");
            Some(places.quantize(frame.now.body.odometry.x_m, frame.now.body.odometry.y_m))
        };
        self.places
            .lock()
            .expect("place memory mutex poisoned")
            .observe_frame(frame);
        self.entities
            .lock()
            .expect("entity memory mutex poisoned")
            .observe_frame(frame, cell_key);
        Ok(())
    }

    async fn observe_transition(&self, transition: &ExperienceTransition) -> Result<()> {
        self.places
            .lock()
            .expect("place memory mutex poisoned")
            .observe_transition(transition);
        Ok(())
    }

    async fn loop_closure_candidates(
        &self,
        query: &RecallQuery,
        min_confidence: f32,
        limit: usize,
    ) -> Result<Vec<PlaceRecognitionCandidate>> {
        let places = self.places.lock().expect("place memory mutex poisoned");
        let current_key = query.pose.map(|pose| places.quantize(pose.x_m, pose.y_m));
        let mut query_vectors = query.scene_vectors.clone();
        if let Some(input) = query.place_recognition_input.as_ref() {
            query_vectors.extend(place_recognition_vectors_from_input(input));
        }
        let mut candidates =
            places.recognize_places(current_key, &query_vectors, min_confidence, limit);
        let mut entity_labels = query
            .place_recognition_input
            .as_ref()
            .map(|input| {
                input
                    .object_labels
                    .iter()
                    .chain(input.person_labels.iter())
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        entity_labels.sort();
        entity_labels.dedup();
        candidates.extend(places.recognize_entity_constellations(
            current_key,
            &entity_labels,
            min_confidence,
            limit,
        ));
        candidates.sort_by(|left, right| {
            right
                .confidence
                .partial_cmp(&left.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right.cell.last_seen_tick.cmp(&left.cell.last_seen_tick))
        });
        candidates.truncate(limit);
        Ok(candidates)
    }

    async fn recall(&self, query: RecallQuery) -> Result<RecallBundle> {
        let place_features = query
            .pose
            .map(|pose| {
                self.places
                    .lock()
                    .expect("place memory mutex poisoned")
                    .features_at(pose.x_m, pose.y_m)
            })
            .unwrap_or_default();
        let records = self.snapshot();
        let mut scored = records
            .into_iter()
            .filter_map(|record| score_record(&query, record))
            .collect::<Vec<_>>();
        scored.sort_by(|left, right| {
            right
                .0
                .partial_cmp(&left.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right.1.t_ms.cmp(&left.1.t_ms))
        });

        let mut hits = Vec::new();
        let mut recollections = Vec::new();
        let mut seen_actions = BTreeSet::new();
        let mut place_familiarity = 0.0f32;
        let mut place_danger = 0.0f32;
        let mut place_charge_value = 0.0f32;
        let mut face_familiarity = 0.0f32;
        let mut object_familiarity = 0.0f32;
        let mut voice_familiarity = 0.0f32;
        let mut remembered_warning = None;
        let mut best_remembered_action = None;
        let mut remembered_entities = Vec::new();
        let mut remembered_relationships = Vec::new();

        for (index, (score, record)) in scored.into_iter().take(5).enumerate() {
            hits.push(RecallHit {
                frame_id: Some(record.frame_id),
                score,
                summary: record.summary.clone(),
                warning: record.warning.clone(),
                graph_context: scored_entities(&record, score),
            });
            let scene_score = max_vector_similarity(
                query_scene_vectors(&query),
                record.scene_vectors.iter().collect(),
            );
            let face_score = max_vector_similarity(
                query_face_vectors(&query),
                record.face_vectors.iter().collect(),
            );
            let object_score = max_vector_similarity(
                query_object_vectors(&query),
                record.object_vectors.iter().collect(),
            );
            let voice_score = max_vector_similarity(
                query_voice_vectors(&query),
                record.voice_vectors.iter().collect(),
            );

            place_familiarity = place_familiarity.max(score).max(scene_score);
            if record.summary.to_ascii_lowercase().contains("danger") || record.warning.is_some() {
                place_danger = place_danger.max(score);
            }
            if matches!(record.active_goal, Some(Goal::Dock)) || record.summary.contains("charge") {
                place_charge_value = place_charge_value.max(score);
            }
            if has_face_query(&query) {
                face_familiarity = face_familiarity.max(score).max(face_score);
            }
            if has_object_query(&query) {
                object_familiarity = object_familiarity.max(score).max(object_score);
            }
            if has_voice_query(&query) {
                voice_familiarity = voice_familiarity.max(score).max(voice_score);
            }
            if remembered_warning.is_none() {
                remembered_warning = record.warning.clone();
            }
            if best_remembered_action.is_none() {
                if let Some(action) = &record.chosen_action {
                    let key = format!("{action:?}");
                    if seen_actions.insert(key) {
                        best_remembered_action = Some(action.clone());
                    }
                }
            }
            remembered_entities.extend(scored_entities(&record, score));
            remembered_relationships.extend(record.graph_relationships.clone());
            if let Some(experience) = record.experience.as_ref() {
                let original_vector_ids = recall_vector_ids(&record);
                let mut sensation = experience.to_recall_sensation_with_lineage(
                    query_pose_time_hint(&query, index as u64),
                    score,
                    "memory-recall",
                    Some(record.frame_id),
                    original_vector_ids.clone(),
                );
                let impression = experience.to_recall_impression(&sensation, score);
                sensation.impression = Some(impression);
                recollections.push(RecalledExperience {
                    score,
                    experience: experience.clone(),
                    sensation,
                    original_frame_id: Some(record.frame_id),
                    original_vector_ids,
                });
            }
        }
        remembered_entities = dedupe_entities(remembered_entities, 12);
        remembered_relationships = dedupe_relationships(remembered_relationships, 16);
        let graph_context_summary = graph_context_summary(&remembered_entities);

        let sense = MemorySense {
            place_familiarity: place_familiarity.max(place_features.current_place_familiarity),
            place_danger: place_danger.max(place_features.current_place_danger),
            place_charge_value: place_charge_value.max(place_features.current_place_charge),
            place_social_value: place_features.current_place_social,
            place_novelty: place_features.current_place_novelty,
            nearby_best_charge_direction_rad: place_features.nearby_best_charge_direction_rad,
            nearby_best_safe_direction_rad: place_features.nearby_best_safe_direction_rad,
            nearby_frontier_direction_rad: place_features.nearby_frontier_direction_rad,
            recent_trap_direction_rad: None,
            map_confidence: place_features.current_place_confidence,
            recent_trap_confidence: 0.0,
            places_visited: place_features.places_visited,
            face_familiarity,
            object_familiarity,
            voice_familiarity,
            similar_situation_count: hits.len().try_into().unwrap_or(u16::MAX),
            best_remembered_action,
            remembered_warning,
            remembered_entities,
            remembered_relationships,
            graph_context_summary,
        };
        let place_query_vectors = place_query_vectors_from_query(&query);
        let (semantic_map, place_recognition_candidates) = {
            let places = self.places.lock().expect("place memory mutex poisoned");
            let current_key = query.pose.map(|pose| places.quantize(pose.x_m, pose.y_m));
            let semantic_map = current_key.map(|key| {
                places.semantic_overlay_with_query(
                    Some(key),
                    &place_query_vectors,
                    PLACE_RECOGNITION_MIN_CONFIDENCE,
                )
            });
            let candidates = places.recognize_places(
                current_key,
                &place_query_vectors,
                PLACE_RECOGNITION_MIN_CONFIDENCE,
                5,
            );
            (semantic_map, candidates)
        };
        let first_person_summary = if hits.is_empty() {
            "I do not remember a similar situation yet.".to_string()
        } else {
            format!(
                "I remember {} similar moments. The closest one was: {}",
                hits.len(),
                hits[0].summary
            )
        };

        Ok(RecallBundle {
            hits,
            sense,
            first_person_summary,
            recollections,
            semantic_map,
            place_recognition_candidates,
        })
    }
}

pub fn memory_record_from_frame(frame: &ExperienceFrame) -> Result<MemoryRecord> {
    let warning = frame
        .memory_recall
        .iter()
        .find_map(|hit| hit.warning.clone())
        .or_else(|| frame.now.memory.remembered_warning.clone());
    let scene_vectors = scene_vectors_from_now(&frame.now, frame.id, frame.t_ms);
    let face_vectors = frame.now.face.vectors.clone();
    let object_vectors = frame.now.objects.vectors.clone();
    let voice_vectors = frame.now.voice.vectors.clone();
    let linked_experiences = experiences_with_memory_links(
        frame,
        &scene_vectors,
        &face_vectors,
        &object_vectors,
        &voice_vectors,
    );
    let (sensation_vectors, mut vector_payloads) = sensation_vectors_from_frame(frame);
    let experience_vectors = experience_vectors_from_frame(frame, &mut vector_payloads);
    let (graph_entities, graph_relationships) = graph_context_from_frame(
        frame,
        &linked_experiences,
        &scene_vectors,
        &face_vectors,
        &object_vectors,
        &voice_vectors,
    );
    Ok(MemoryRecord {
        frame_id: frame.id,
        t_ms: frame.t_ms,
        summary: frame.summary_text(),
        graph_entities,
        graph_relationships,
        scene_vectors,
        face_vectors,
        object_vectors,
        voice_vectors,
        sensation_vectors,
        experience_vectors,
        vector_payloads,
        battery: frame.now.body.battery_level,
        active_goal: RecallQuery::from_now(&frame.now).active_goal,
        chosen_action: frame.chosen_action.clone(),
        warning,
        experience: linked_experiences.last().cloned(),
        temporal_context: frame.now.world.temporal.clone(),
        social_world: frame.now.world.social.clone(),
        epistemic_state: frame.now.world.epistemic.clone(),
    })
}

pub fn attach_memory_links_to_frame(frame: &mut ExperienceFrame) {
    let scene_vectors = scene_vectors_from_now(&frame.now, frame.id, frame.t_ms);
    let face_vectors = frame.now.face.vectors.clone();
    let object_vectors = frame.now.objects.vectors.clone();
    let voice_vectors = frame.now.voice.vectors.clone();
    let links = memory_links_from_frame(
        frame,
        &scene_vectors,
        &face_vectors,
        &object_vectors,
        &voice_vectors,
    );
    for experience in &mut frame.experiences {
        merge_memory_links(&mut experience.memory_links, links.clone());
    }
}

fn experiences_with_memory_links(
    frame: &ExperienceFrame,
    scene_vectors: &[VectorArtifact],
    face_vectors: &[VectorArtifact],
    object_vectors: &[VectorArtifact],
    voice_vectors: &[VectorArtifact],
) -> Vec<Experience> {
    let links = memory_links_from_frame(
        frame,
        scene_vectors,
        face_vectors,
        object_vectors,
        voice_vectors,
    );
    frame
        .experiences
        .iter()
        .cloned()
        .map(|mut experience| {
            merge_memory_links(&mut experience.memory_links, links.clone());
            experience
        })
        .collect()
}

pub fn place_memory_report_from_frames(frames: &[ExperienceFrame]) -> PlaceMemoryReport {
    place_memory_from_frames(frames).report()
}

pub fn place_memory_from_frames(frames: &[ExperienceFrame]) -> PlaceMemory {
    let mut memory = PlaceMemory::new();
    for frame in frames {
        memory.observe_frame(frame);
    }
    memory
}

