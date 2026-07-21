fn top_cells(
    cells: &BTreeMap<PlaceCellKey, PlaceCell>,
    score: impl Fn(&PlaceCell) -> f32,
) -> Vec<PlaceCellSummary> {
    let mut scored = cells
        .values()
        .map(|cell| (score(cell), cell))
        .filter(|(score, _)| *score > 0.0)
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored
        .into_iter()
        .take(5)
        .map(|(score, cell)| summarize_cell(cell, score))
        .collect()
}

fn summarize_cell(cell: &PlaceCell, score: f32) -> PlaceCellSummary {
    PlaceCellSummary {
        x: cell.key.x,
        y: cell.key.y,
        center_x_m: cell.center_x_m,
        center_y_m: cell.center_y_m,
        score,
        visit_count: cell.visit_count,
        last_seen_tick: cell.last_seen_tick,
        confidence: cell.confidence,
        last_observed_objects: cell.last_observed_objects.clone(),
        associated_scene_vectors: cell.associated_scene_vectors.clone(),
        associated_face_vectors: cell.associated_face_vectors.clone(),
        associated_object_vectors: cell.associated_object_vectors.clone(),
        associated_voice_vectors: cell.associated_voice_vectors.clone(),
        successful_actions: cell.successful_actions.clone(),
        failed_actions: cell.failed_actions.clone(),
    }
}

fn recognition_kind(
    current_key: Option<PlaceCellKey>,
    candidate_key: PlaceCellKey,
    similarity: f32,
) -> PlaceRecognitionKind {
    if current_key == Some(candidate_key) || similarity >= 0.92 {
        PlaceRecognitionKind::SamePlace
    } else {
        PlaceRecognitionKind::SimilarPlace
    }
}

fn candidate_reason(
    similarity: f32,
    confidence: f32,
    current_key: Option<PlaceCellKey>,
    candidate_key: PlaceCellKey,
) -> String {
    let kind = if current_key == Some(candidate_key) {
        "same map cell"
    } else if similarity >= 0.92 {
        "high latent similarity"
    } else {
        "moderate latent similarity"
    };
    format!("{kind}; similarity={similarity:.3}; confidence={confidence:.3}")
}

fn score_record(query: &RecallQuery, record: MemoryRecord) -> Option<(f32, MemoryRecord)> {
    let query_tokens = tokenize(&query.now_text);
    let summary_tokens = tokenize(&record.summary);
    let overlap = token_overlap(&query_tokens, &summary_tokens);
    let battery_distance = (query.battery - record.battery).abs();
    let battery_score = (1.0 - battery_distance).clamp(0.0, 1.0);
    let goal_score = if query.active_goal.is_some() && query.active_goal == record.active_goal {
        1.0
    } else {
        0.0
    };
    let action_score =
        if query.proposed_action.is_some() && query.proposed_action == record.chosen_action {
            0.8
        } else {
            0.0
        };
    let vector_score = max_vector_similarity(query_all_vectors(query), record_all_vectors(&record));
    let score = (overlap * 0.4)
        + (battery_score * 0.15)
        + (goal_score * 0.15)
        + (action_score * 0.1)
        + (vector_score * 0.2);
    if score <= 0.05 {
        return None;
    }
    Some((score, record))
}

fn graph_context_from_frame(
    frame: &ExperienceFrame,
    experiences: &[Experience],
    scene_vectors: &[VectorArtifact],
    face_vectors: &[VectorArtifact],
    object_vectors: &[VectorArtifact],
    voice_vectors: &[VectorArtifact],
) -> (Vec<GraphEntity>, Vec<GraphEdge>) {
    let frame_id = frame.id.to_string();
    let experience_id = format!("experience:{frame_id}");
    let mut entities = vec![
        GraphEntity {
            id: format!("frame:{frame_id}"),
            labels: vec!["Frame".to_string(), "Memory".to_string()],
            summary: format!("runtime frame {frame_id}"),
            score: 1.0,
        },
        GraphEntity {
            id: experience_id.clone(),
            labels: vec!["Experience".to_string(), "Memory".to_string()],
            summary: frame.summary_text(),
            score: 1.0,
        },
    ];
    let mut relationships = vec![graph_edge(
        format!("frame:{frame_id}"),
        experience_id.clone(),
        "HAS_MEMORY_SUMMARY",
        None,
    )];

    let pose = frame.now.body.odometry;
    let place_id = place_id_for_pose(pose);
    entities.push(GraphEntity {
        id: place_id.clone(),
        labels: vec!["Place".to_string()],
        summary: format!("place near x={:.1}m y={:.1}m", pose.x_m, pose.y_m),
        score: 1.0,
    });
    relationships.push(graph_edge(
        experience_id.clone(),
        place_id,
        "OCCURRED_AT",
        None,
    ));

    append_world_model_memory(
        &frame.now.world.social,
        &frame.now.world.temporal,
        &experience_id,
        &mut entities,
        &mut relationships,
    );
    append_semantic_graph_memory(&frame.now.world.semantic, &mut entities, &mut relationships);

    for artifact in scene_vectors {
        let vector_id = vector_node_id(artifact);
        entities.push(vector_entity(artifact, "scene"));
        relationships.push(graph_edge(
            experience_id.clone(),
            vector_id,
            "HAS_SCENE_VECTOR",
            None,
        ));
    }

    for sensation in &frame.sensations {
        let sensation_id = sensation_node_id(sensation.id);
        entities.push(GraphEntity {
            id: sensation_id.clone(),
            labels: vec![
                "Sensation".to_string(),
                sensation.modality.as_str().to_string(),
                sensation.payload_kind.as_str().to_string(),
            ],
            summary: sensation
                .summary
                .clone()
                .unwrap_or_else(|| sensation.kind.clone()),
            score: sensation.metadata.confidence.unwrap_or(1.0),
        });
        relationships.push(graph_edge(
            format!("frame:{frame_id}"),
            sensation_id.clone(),
            "HAS_SENSATION",
            Some(sensation.kind.clone()),
        ));
        if let Some(parent_id) = sensation.parent_id {
            relationships.push(graph_edge(
                sensation_id.clone(),
                sensation_node_id(parent_id),
                "DERIVED_FROM_SENSATION",
                None,
            ));
        }
        if let Some(embedding) = &sensation.vector {
            let artifact = embodied_vector_artifact(
                SENSATION_VECTOR_COLLECTION,
                &format!("{}:sensation:{}", frame.id, sensation.id),
                embedding,
                frame.id,
                sensation.id.to_string(),
                sensation.occurred_at_ms,
            );
            entities.push(vector_entity(&artifact, "sensation"));
            relationships.push(graph_edge(
                sensation_id.clone(),
                vector_node_id(&artifact),
                "HAS_SENSATION_VECTOR",
                Some(format!("{} dimensions", embedding.dim)),
            ));
        }
        if let Some(impression) = &sensation.impression {
            let impression_id = impression_node_id(impression.id);
            entities.push(impression_entity(impression));
            relationships.push(graph_edge(
                sensation_id,
                impression_id,
                "HAS_IMPRESSION",
                Some(impression.text.clone()),
            ));
        }
    }

    for impression in &frame.impressions {
        entities.push(impression_entity(impression));
        for sensation_id in &impression.about {
            relationships.push(graph_edge(
                sensation_node_id(*sensation_id),
                impression_node_id(impression.id),
                "HAS_IMPRESSION",
                Some(impression.text.clone()),
            ));
        }
    }

    for experience in experiences {
        let canonical_experience_id = experience_node_id(experience.id);
        entities.push(GraphEntity {
            id: canonical_experience_id.clone(),
            labels: vec!["Experience".to_string(), "EmbodiedExperience".to_string()],
            summary: experience_summary_text(experience),
            score: experience.salience,
        });
        relationships.push(graph_edge(
            format!("frame:{frame_id}"),
            canonical_experience_id.clone(),
            "HAS_EXPERIENCE",
            Some(experience.kind.clone()),
        ));
        relationships.push(graph_edge(
            experience_id.clone(),
            canonical_experience_id.clone(),
            "SUMMARIZES_EXPERIENCE",
            None,
        ));
        let (artifact, _, _, _) = experience_vector_artifact(frame, experience);
        entities.push(vector_entity(&artifact, "experience"));
        relationships.push(graph_edge(
            canonical_experience_id.clone(),
            vector_node_id(&artifact),
            "HAS_FUSED_VECTOR",
            Some(format!("{} dimensions", artifact.vector.len())),
        ));
        for sensation_id in &experience.sensation_ids {
            relationships.push(graph_edge(
                canonical_experience_id.clone(),
                sensation_node_id(*sensation_id),
                "INTEGRATES_SENSATION",
                None,
            ));
        }
        for impression_id in &experience.impression_ids {
            relationships.push(graph_edge(
                canonical_experience_id.clone(),
                impression_node_id(*impression_id),
                "INTEGRATES_IMPRESSION",
                None,
            ));
        }
        if let Some(impression) = &experience.summary_impression {
            entities.push(impression_entity(impression));
            relationships.push(graph_edge(
                canonical_experience_id.clone(),
                impression_node_id(impression.id),
                "HAS_SUMMARY_IMPRESSION",
                Some(impression.text.clone()),
            ));
        }
        for link in &experience.memory_links {
            if let Some(entity) = memory_link_entity(link) {
                entities.push(entity);
            }
            relationships.push(memory_link_edge(canonical_experience_id.clone(), link));
        }
    }

    for (index, artifact) in face_vectors.iter().enumerate() {
        let person_id = artifact
            .source_id
            .clone()
            .unwrap_or_else(|| format!("person:face:{frame_id}:{index}"));
        entities.push(GraphEntity {
            id: person_id.clone(),
            labels: vec![
                "Person".to_string(),
                "FaceInstance".to_string(),
                "Entity".to_string(),
            ],
            summary: "person seen by face vector".to_string(),
            score: 1.0,
        });
        entities.push(vector_entity(artifact, "face"));
        relationships.push(graph_edge(
            experience_id.clone(),
            person_id.clone(),
            "SAW_PERSON",
            None,
        ));
        relationships.push(graph_edge(
            person_id,
            vector_node_id(artifact),
            "HAS_FACE_VECTOR",
            None,
        ));
    }

    for (index, artifact) in object_vectors.iter().enumerate() {
        let object_id = artifact
            .source_id
            .clone()
            .unwrap_or_else(|| format!("object:vector:{frame_id}:{index}"));
        entities.push(GraphEntity {
            id: object_id.clone(),
            labels: vec![
                "Object".to_string(),
                "ObjectInstance".to_string(),
                "Entity".to_string(),
            ],
            summary: "object seen by visual vector".to_string(),
            score: 1.0,
        });
        entities.push(vector_entity(artifact, "object"));
        relationships.push(graph_edge(
            experience_id.clone(),
            object_id.clone(),
            "SAW_OBJECT",
            None,
        ));
        relationships.push(graph_edge(
            object_id,
            vector_node_id(artifact),
            "HAS_OBJECT_VECTOR",
            None,
        ));
    }

    for (index, artifact) in voice_vectors.iter().enumerate() {
        let person_id = artifact
            .source_id
            .clone()
            .unwrap_or_else(|| format!("person:voice:{frame_id}:{index}"));
        entities.push(GraphEntity {
            id: person_id.clone(),
            labels: vec![
                "Person".to_string(),
                "VoiceSignature".to_string(),
                "Entity".to_string(),
            ],
            summary: "person heard by voice vector".to_string(),
            score: 1.0,
        });
        entities.push(vector_entity(artifact, "voice"));
        relationships.push(graph_edge(
            experience_id.clone(),
            person_id.clone(),
            "HEARD_PERSON",
            None,
        ));
        relationships.push(graph_edge(
            person_id,
            vector_node_id(artifact),
            "HAS_VOICE_VECTOR",
            None,
        ));
    }

    (
        dedupe_entities(entities, usize::MAX),
        dedupe_relationships(relationships, usize::MAX),
    )
}

fn append_semantic_graph_memory(
    semantic: &SemanticGraphSnapshot,
    entities: &mut Vec<GraphEntity>,
    relationships: &mut Vec<GraphEdge>,
) {
    for relation in semantic.relations.values() {
        for node in [&relation.subject, &relation.object] {
            entities.push(GraphEntity {
                id: node.stable_key(),
                labels: semantic_node_labels(node),
                summary: node.stable_key(),
                score: relation.confidence,
            });
        }
        relationships.push(GraphEdge {
            id: Some(relation.id.0.clone()),
            from: relation.subject.stable_key(),
            to: relation.object.stable_key(),
            relationship: format!("SEMANTIC_{:?}", relation.predicate).to_ascii_uppercase(),
            summary: Some(format!(
                "grounded semantic relation {:?} ({:?})",
                relation.predicate, relation.status
            )),
            score: relation.confidence,
            payload: serde_json::to_value(relation).unwrap_or(serde_json::Value::Null),
        });
    }
}

fn semantic_node_labels(node: &SemanticNodeRef) -> Vec<String> {
    let kind = match node {
        SemanticNodeRef::Entity(_) => "Entity",
        SemanticNodeRef::Place(_) => "Place",
        SemanticNodeRef::Person(_) => "Person",
        SemanticNodeRef::Action(_) => "Action",
        SemanticNodeRef::Skill(_) => "Skill",
        SemanticNodeRef::Behavior(_) => "Behavior",
        SemanticNodeRef::Goal(_) => "Goal",
        SemanticNodeRef::Drive(_) => "Drive",
        SemanticNodeRef::Outcome(_) => "Outcome",
        SemanticNodeRef::Property(_) => "Property",
        SemanticNodeRef::Concept(_) => "Concept",
        SemanticNodeRef::Episode(_) => "Episode",
    };
    vec!["SemanticNode".to_string(), kind.to_string()]
}

fn append_world_model_memory(
    social: &SocialWorldSnapshot,
    temporal: &TemporalContext,
    experience_id: &str,
    entities: &mut Vec<GraphEntity>,
    relationships: &mut Vec<GraphEdge>,
) {
    for person in social.people.values() {
        let name = person
            .preferred_name
            .as_ref()
            .map(|name| name.value.as_str())
            .or_else(|| {
                person
                    .best_identity()
                    .and_then(|identity| identity.display_name.as_deref())
            })
            .unwrap_or("unknown person");
        entities.push(GraphEntity {
            id: person.person_id.0.clone(),
            labels: vec![
                "Person".to_string(),
                "SocialPersonBelief".to_string(),
                "Entity".to_string(),
            ],
            summary: format!("social belief about {name}"),
            score: person.current_identity_confidence,
        });
        relationships.push(GraphEdge {
            id: None,
            from: experience_id.to_string(),
            to: person.person_id.0.clone(),
            relationship: if person.presence.present {
                "INTERACTED_WITH_OR_OBSERVED".to_string()
            } else {
                "REMEMBERS_PERSON".to_string()
            },
            summary: Some(format!(
                "presence={} identity_confidence={:.3}",
                person.presence.present, person.current_identity_confidence
            )),
            score: person
                .presence
                .confidence
                .max(person.current_identity_confidence),
            payload: json!({
                "present": person.presence.present,
                "presence_freshness": person.presence.freshness,
                "last_seen_at_ms": person.presence.last_seen_at_ms,
                "identity_hypotheses": person.identity_hypotheses,
            }),
        });
    }

    for relationship in social.relationships.values() {
        let relationship_id = format!("relationship:{}", relationship.relationship_id.0);
        let confidence = relationship
            .relationship_kinds
            .iter()
            .map(|kind| kind.confidence)
            .fold(0.0f32, f32::max);
        entities.push(GraphEntity {
            id: relationship_id.clone(),
            labels: vec!["Relationship".to_string(), "SocialBelief".to_string()],
            summary: format!("relationship belief for {}", relationship.person_id.0),
            score: confidence,
        });
        relationships.push(GraphEdge {
            id: None,
            from: relationship_id,
            to: relationship.person_id.0.clone(),
            relationship: "RELATES_TO_PERSON".to_string(),
            summary: Some(format!(
                "trust={:.3} affiliation={:.3}",
                relationship.trust, relationship.affiliation
            )),
            score: confidence,
            payload: json!({
                "relationship_kinds": relationship.relationship_kinds,
                "trust": relationship.trust,
                "affiliation": relationship.affiliation,
                "authority": relationship.caregiving_or_authority,
            }),
        });
    }

    for interaction in social
        .recent_interactions
        .iter()
        .chain(social.active_interaction.iter())
    {
        let interaction_id = interaction.interaction_id.0.clone();
        entities.push(GraphEntity {
            id: interaction_id.clone(),
            labels: vec!["Interaction".to_string(), "Episode".to_string()],
            summary: format!("social interaction in phase {:?}", interaction.phase),
            score: 1.0,
        });
        relationships.push(GraphEdge {
            id: None,
            from: experience_id.to_string(),
            to: interaction_id.clone(),
            relationship: "HAS_SOCIAL_INTERACTION".to_string(),
            summary: None,
            score: 1.0,
            payload: json!({
                "started_at_ms": interaction.started_at_ms,
                "last_activity_ms": interaction.last_activity_ms,
                "ended_at_ms": interaction.ended_at_ms,
                "phase": interaction.phase,
            }),
        });
        for participant in &interaction.participants {
            relationships.push(graph_edge(
                interaction_id.clone(),
                participant.0.clone(),
                "HAS_PARTICIPANT",
                None,
            ));
        }
    }

    for episode in temporal
        .recently_completed
        .iter()
        .chain(temporal.active_episodes.iter())
    {
        let episode_id = episode.episode_id.0.clone();
        entities.push(GraphEntity {
            id: episode_id.clone(),
            labels: vec!["Episode".to_string(), format!("{:?}", episode.kind)],
            summary: format!("{:?} episode", episode.kind),
            score: episode.confidence,
        });
        relationships.push(GraphEdge {
            id: None,
            from: experience_id.to_string(),
            to: episode_id.clone(),
            relationship: "HAS_TEMPORAL_EPISODE".to_string(),
            summary: episode.closure_reason.map(|reason| format!("{reason:?}")),
            score: episode.confidence,
            payload: json!({
                "clock_domain": episode.interval.domain,
                "start_ms": episode.interval.start_ms,
                "end_ms": episode.interval.end_ms,
                "active_goals": episode.active_goals,
                "significant_events": episode.significant_events,
            }),
        });
        for participant in &episode.participants {
            relationships.push(graph_edge(
                episode_id.clone(),
                participant.0.clone(),
                "HAS_PARTICIPANT",
                None,
            ));
        }
        for preceding in &episode.preceding_episode_refs {
            relationships.push(graph_edge(
                episode_id.clone(),
                preceding.0.clone(),
                "FOLLOWS_EPISODE",
                None,
            ));
        }
    }
}

fn place_id_for_pose(pose: Pose2) -> String {
    format!(
        "place:grid:{:.0}:{:.0}",
        (pose.x_m * 2.0).round(),
        (pose.y_m * 2.0).round()
    )
}

fn graph_edge(
    from: impl Into<String>,
    to: impl Into<String>,
    relationship: impl Into<String>,
    summary: Option<String>,
) -> GraphEdge {
    GraphEdge {
        id: None,
        from: from.into(),
        to: to.into(),
        relationship: relationship.into(),
        summary,
        score: 1.0,
        payload: serde_json::Value::Null,
    }
}

fn memory_link_edge(from: impl Into<String>, link: &MemoryLink) -> GraphEdge {
    GraphEdge {
        id: None,
        from: from.into(),
        to: link.target_id.clone(),
        relationship: link.relation.clone(),
        summary: link
            .payload
            .get("text")
            .and_then(|value| value.as_str())
            .map(str::to_string),
        score: link.score,
        payload: link.payload.clone(),
    }
}

fn memory_link_entity(link: &MemoryLink) -> Option<GraphEntity> {
    let target_kind = link
        .payload
        .get("target_kind")
        .and_then(|value| value.as_str())?;
    let text = link
        .payload
        .get("text")
        .and_then(|value| value.as_str())
        .unwrap_or(&link.target_id)
        .to_string();
    let labels = match target_kind {
        "place" => vec!["Place".to_string(), "Entity".to_string()],
        "object" => {
            let mut labels = vec!["Object".to_string(), "Entity".to_string()];
            if let Some(class) = link.payload.get("class").and_then(|value| value.as_str()) {
                labels.push(class.to_string());
            }
            labels
        }
        "person" => vec!["Person".to_string(), "Entity".to_string()],
        "surface" => vec!["Surface".to_string(), "Entity".to_string()],
        "experience" => vec!["Experience".to_string(), "Memory".to_string()],
        _ => return None,
    };
    Some(GraphEntity {
        id: link.target_id.clone(),
        labels,
        summary: text,
        score: link.score,
    })
}

fn vector_node_id(artifact: &VectorArtifact) -> String {
    format!("vector:{}:{}", artifact.collection, artifact.point_id)
}

fn sensation_node_id(id: uuid::Uuid) -> String {
    format!("sensation:{id}")
}

fn impression_node_id(id: uuid::Uuid) -> String {
    format!("impression:{id}")
}

fn experience_node_id(id: uuid::Uuid) -> String {
    format!("embodied_experience:{id}")
}

fn impression_entity(impression: &Impression) -> GraphEntity {
    GraphEntity {
        id: impression_node_id(impression.id),
        labels: vec!["Impression".to_string(), impression.kind.clone()],
        summary: impression.text.clone(),
        score: impression.confidence,
    }
}

fn experience_summary_text(experience: &Experience) -> String {
    experience
        .summary_impression
        .as_ref()
        .map(|impression| impression.text.clone())
        .unwrap_or_else(|| experience.text.clone())
}

fn vector_entity(artifact: &VectorArtifact, kind: &str) -> GraphEntity {
    GraphEntity {
        id: vector_node_id(artifact),
        labels: vec!["Vector".to_string()],
        summary: format!(
            "{kind} vector in {} with {} dimensions",
            artifact.collection,
            artifact.vector.len()
        ),
        score: 1.0,
    }
}

fn scored_entities(record: &MemoryRecord, score: f32) -> Vec<GraphEntity> {
    record
        .graph_entities
        .iter()
        .filter(|entity| !entity.has_label("Vector"))
        .map(|entity| {
            let mut entity = entity.clone();
            entity.score = score;
            entity
        })
        .collect()
}

fn dedupe_entities(entities: Vec<GraphEntity>, limit: usize) -> Vec<GraphEntity> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for entity in entities {
        if seen.insert(entity.id.clone()) {
            out.push(entity);
        }
        if out.len() >= limit {
            break;
        }
    }
    out
}

fn dedupe_relationships(relationships: Vec<GraphEdge>, limit: usize) -> Vec<GraphEdge> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for relationship in relationships {
        let key = graph_edge_id(&relationship);
        if seen.insert(key) {
            out.push(relationship);
        }
        if out.len() >= limit {
            break;
        }
    }
    out
}

fn graph_edge_id(edge: &GraphEdge) -> String {
    edge.id
        .as_deref()
        .filter(|id| !id.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| {
            // Character-length-prefix the structural fallback so identifiers
            // containing separators cannot alias another relationship triple,
            // and Neo4j can reproduce the same id with Cypher `size()`.
            format!(
                "graph-edge:{}:{}:{}:{}:{}:{}",
                edge.from.chars().count(),
                edge.from,
                edge.relationship.chars().count(),
                edge.relationship,
                edge.to.chars().count(),
                edge.to
            )
        })
}

fn graph_context_summary(entities: &[GraphEntity]) -> Option<String> {
    let people = entities
        .iter()
        .filter(|entity| entity.has_label("Person"))
        .count();
    let places = entities
        .iter()
        .filter(|entity| entity.has_label("Place"))
        .count();
    let experiences = entities
        .iter()
        .filter(|entity| entity.has_label("Experience"))
        .count();
    if people == 0 && places == 0 && experiences == 0 {
        return None;
    }
    Some(format!(
        "Graph recall: {people} person nodes, {places} place nodes, {experiences} experience nodes."
    ))
}

fn scene_vectors_from_now(now: &Now, frame_id: uuid::Uuid, t_ms: u64) -> Vec<VectorArtifact> {
    if !now.eye.scene_vectors.is_empty() {
        return now.eye.scene_vectors.clone();
    }
    now.eye
        .frames
        .last()
        .map(|vector| {
            VectorArtifact::new(
                SCENE_VECTOR_COLLECTION,
                format!(
                    "{}:scene:{}",
                    frame_id,
                    now.eye.frames.len().saturating_sub(1)
                ),
                vector.clone(),
            )
            .with_source_frame_id(frame_id.to_string())
            .with_occurred_at_ms(t_ms)
        })
        .into_iter()
        .collect()
}

fn sensation_vectors_from_frame(
    frame: &ExperienceFrame,
) -> (Vec<VectorArtifact>, BTreeMap<String, serde_json::Value>) {
    let mut payloads = BTreeMap::new();
    let artifacts = frame
        .sensations
        .iter()
        .filter_map(|sensation| {
            let embedding = sensation.vector.as_ref()?;
            let artifact = embodied_vector_artifact(
                SENSATION_VECTOR_COLLECTION,
                &format!("{}:sensation:{}", frame.id, sensation.id),
                embedding,
                frame.id,
                sensation.id.to_string(),
                sensation.occurred_at_ms,
            );
            payloads.insert(
                vector_payload_key(&artifact),
                json!({
                    "payload_kind": sensation.payload_kind.as_str(),
                    "modality": sensation.modality.as_str(),
                    "sensation_id": sensation.id.to_string(),
                    "parent_sensation_id": sensation.parent_id.map(|id| id.to_string()),
                    "source_sensation_id": embedding.source_sensation_id.to_string(),
                    "model_id": embedding.model_id,
                    "dim": embedding.dim,
                    "observed_at_ms": sensation.observed_at_ms,
                    "occurred_at_ms": sensation.occurred_at_ms,
                    "generated_at_ms": embedding.generated_at_ms,
                    "sensation_kind": sensation.kind,
                    "source": sensation.source,
                    "summary": sensation.summary,
                    "labels": sensation.metadata.labels,
                    "confidence": sensation.metadata.confidence,
                    "provenance": sensation.provenance,
                }),
            );
            Some(artifact)
        })
        .collect();
    (artifacts, payloads)
}

fn experience_vectors_from_frame(
    frame: &ExperienceFrame,
    payloads: &mut BTreeMap<String, serde_json::Value>,
) -> Vec<VectorArtifact> {
    frame
        .experiences
        .iter()
        .map(|experience| {
            let summary_impression = experience.summary_impression.as_ref();
            let (artifact, model_id, generated_at_ms, summary_text) =
                experience_vector_artifact(frame, experience);
            payloads.insert(
                vector_payload_key(&artifact),
                json!({
                    "experience_id": experience.id.to_string(),
                    "experience_kind": experience.kind,
                    "summary": experience.text,
                    "summary_impression_id": summary_impression.map(|impression| impression.id.to_string()),
                    "summary_impression_text": summary_text,
                    "impression_ids": experience.impression_ids.iter().map(ToString::to_string).collect::<Vec<_>>(),
                    "sensation_ids": experience.sensation_ids.iter().map(ToString::to_string).collect::<Vec<_>>(),
                    "model_id": model_id,
                    "dim": artifact.vector.len(),
                    "observed_at_ms": experience.observed_at_ms,
                    "occurred_at_ms": experience.occurred_at_ms,
                    "generated_at_ms": generated_at_ms,
                }),
            );
            artifact
        })
        .collect()
}

fn deterministic_text_vector(text: &str, dim: usize) -> Vec<f32> {
    let dim = dim.max(1);
    let mut vector = vec![0.0; dim];
    for token in text.split_whitespace() {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        token.to_ascii_lowercase().hash(&mut hasher);
        let hash = hasher.finish();
        let index = hash as usize % dim;
        let sign = if (hash >> 63) == 0 { 1.0 } else { -1.0 };
        vector[index] += sign;
    }
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

fn experience_vector_artifact(
    frame: &ExperienceFrame,
    experience: &Experience,
) -> (VectorArtifact, String, u64, String) {
    let summary_impression = experience.summary_impression.as_ref();
    let summary_text = summary_impression
        .map(|impression| impression.text.clone())
        .unwrap_or_else(|| experience.text.clone());
    let (vector, model_id, generated_at_ms) = if let Some(latent) = &frame.z {
        (
            latent.z.clone(),
            "pete.experience.latent".to_string(),
            latent.t_ms,
        )
    } else if let Some(embedding) =
        summary_impression.and_then(|impression| impression.vector.as_ref())
    {
        (
            embedding.vector.clone(),
            embedding.model_id.clone(),
            embedding.generated_at_ms,
        )
    } else {
        (
            deterministic_text_vector(&summary_text, 16),
            "pete.text.hashing.v1".to_string(),
            frame.t_ms,
        )
    };
    let artifact = VectorArtifact::new(
        EXPERIENCE_VECTOR_COLLECTION,
        format!("{}:experience:{}", frame.id, experience.id),
        vector,
    )
    .with_model(model_id.clone())
    .with_source_id(experience.id.to_string())
    .with_source_frame_id(frame.id.to_string())
    .with_occurred_at_ms(experience.occurred_at_ms);
    (artifact, model_id, generated_at_ms, summary_text)
}

fn memory_links_from_frame(
    frame: &ExperienceFrame,
    _scene_vectors: &[VectorArtifact],
    face_vectors: &[VectorArtifact],
    object_vectors: &[VectorArtifact],
    voice_vectors: &[VectorArtifact],
) -> Vec<MemoryLink> {
    let mut links = Vec::new();
    let pose = frame.now.body.odometry;
    let place_score = if frame.now.memory.places_visited > 0 {
        frame.now.memory.place_familiarity.max(0.75)
    } else {
        frame.now.memory.place_familiarity.max(0.5)
    };
    links.push(MemoryLink {
        target_id: place_id_for_pose(pose),
        relation: "occurred_at_place".to_string(),
        score: place_score.clamp(0.0, 1.0),
        payload: json!({
            "target_kind": "place",
            "text": format!("place near x={:.1}m y={:.1}m", pose.x_m, pose.y_m),
            "x_m": pose.x_m,
            "y_m": pose.y_m,
            "heading_rad": pose.heading_rad,
        }),
    });

    for observation in &frame.now.objects.observations {
        let target_id = object_observation_id(observation);
        links.push(MemoryLink {
            target_id,
            relation: "observed_object".to_string(),
            score: observation.confidence.clamp(0.0, 1.0),
            payload: json!({
                "target_kind": "object",
                "text": observation.label,
                "label": observation.label,
                "class": object_class_slug(&observation.class),
                "bearing_rad": observation.bearing_rad,
                "distance_m": observation.distance_m,
                "source": object_source_slug(&observation.source),
            }),
        });
    }

    for (index, artifact) in face_vectors.iter().enumerate() {
        let target_id = artifact
            .source_id
            .clone()
            .unwrap_or_else(|| format!("person:face:{}:{index}", frame.id));
        links.push(MemoryLink {
            target_id,
            relation: "saw_face".to_string(),
            score: 1.0,
            payload: json!({
                "target_kind": "person",
                "text": "face observed",
                "vector_id": vector_node_id(artifact),
                "collection": artifact.collection,
                "point_id": artifact.point_id,
            }),
        });
    }

    for (index, artifact) in object_vectors.iter().enumerate() {
        let target_id = artifact
            .source_id
            .clone()
            .unwrap_or_else(|| format!("object:vector:{}:{index}", frame.id));
        links.push(MemoryLink {
            target_id,
            relation: "saw_object_vector".to_string(),
            score: 1.0,
            payload: json!({
                "target_kind": "object",
                "text": "object visual vector observed",
                "vector_id": vector_node_id(artifact),
                "collection": artifact.collection,
                "point_id": artifact.point_id,
            }),
        });
    }

    for (index, artifact) in voice_vectors.iter().enumerate() {
        let target_id = artifact
            .source_id
            .clone()
            .unwrap_or_else(|| format!("person:voice:{}:{index}", frame.id));
        links.push(MemoryLink {
            target_id,
            relation: "heard_voice".to_string(),
            score: 1.0,
            payload: json!({
                "target_kind": "person",
                "text": "voice observed",
                "vector_id": vector_node_id(artifact),
                "collection": artifact.collection,
                "point_id": artifact.point_id,
            }),
        });
    }

    if let Some(surface_graph) = frame.now.extensions.get("surface.scene_graph") {
        links.extend(surface_memory_links(surface_graph));
    }

    links.extend(frame.recollections.iter().map(|recollection| MemoryLink {
        target_id: experience_node_id(recollection.experience.id),
        relation: "similar_to_experience".to_string(),
        score: recollection.score.clamp(0.0, 1.0),
        payload: json!({
            "target_kind": "experience",
            "text": recollection.experience.text,
            "original_frame_id": recollection.original_frame_id.map(|id| id.to_string()),
            "original_vector_ids": recollection.original_vector_ids,
        }),
    }));

    dedupe_memory_links(links)
}

fn surface_memory_links(surface_graph: &serde_json::Value) -> Vec<MemoryLink> {
    let mut links = Vec::new();
    if let Some(floor) = surface_graph.get("floor") {
        if let Some(link) = surface_link_from_value(floor, "near_surface") {
            links.push(link);
        }
    }
    if let Some(surfaces) = surface_graph
        .get("surfaces")
        .and_then(|value| value.as_array())
    {
        links.extend(
            surfaces
                .iter()
                .filter_map(|surface| surface_link_from_value(surface, "near_surface")),
        );
    }
    if let Some(clusters) = surface_graph
        .get("clusters")
        .and_then(|value| value.as_array())
    {
        links.extend(
            clusters
                .iter()
                .filter_map(|cluster| surface_link_from_value(cluster, "observed_surface_cluster")),
        );
    }
    links
}

fn surface_link_from_value(value: &serde_json::Value, relation: &str) -> Option<MemoryLink> {
    let id = value.get("id").and_then(|id| id.as_str())?;
    let confidence = value
        .get("confidence")
        .and_then(|confidence| confidence.as_f64())
        .unwrap_or(1.0) as f32;
    Some(MemoryLink {
        target_id: format!("surface:{id}"),
        relation: relation.to_string(),
        score: confidence.clamp(0.0, 1.0),
        payload: json!({
            "target_kind": "surface",
            "text": format!("surface {id}"),
            "surface_id": id,
            "kind": value.get("kind").cloned(),
        }),
    })
}

fn merge_memory_links(existing: &mut Vec<MemoryLink>, incoming: Vec<MemoryLink>) {
    let mut seen = existing
        .iter()
        .map(memory_link_key)
        .collect::<BTreeSet<_>>();
    for link in incoming {
        if seen.insert(memory_link_key(&link)) {
            existing.push(link);
        }
    }
}

fn dedupe_memory_links(links: Vec<MemoryLink>) -> Vec<MemoryLink> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for link in links {
        if seen.insert(memory_link_key(&link)) {
            out.push(link);
        }
    }
    out
}

fn memory_link_key(link: &MemoryLink) -> String {
    format!("{}:{}", link.relation, link.target_id)
}

fn object_observation_id(observation: &pete_now::ObjectObservation) -> String {
    format!(
        "object:{}:{}:{}",
        object_source_slug(&observation.source),
        object_class_slug(&observation.class),
        stable_slug(&observation.label)
    )
}

fn object_class_slug(class: &pete_now::ObjectClass) -> &'static str {
    match class {
        pete_now::ObjectClass::Obstacle => "obstacle",
        pete_now::ObjectClass::Charger => "charger",
        pete_now::ObjectClass::Person => "person",
        pete_now::ObjectClass::SoundSource => "sound_source",
        pete_now::ObjectClass::Landmark => "landmark",
        pete_now::ObjectClass::Unknown => "unknown",
    }
}

fn object_source_slug(source: &pete_now::ObjectObservationSource) -> &'static str {
    match source {
        pete_now::ObjectObservationSource::Sim => "sim",
        pete_now::ObjectObservationSource::Kinect => "kinect",
        pete_now::ObjectObservationSource::Captioner => "captioner",
        pete_now::ObjectObservationSource::CreateIr => "create_ir",
        pete_now::ObjectObservationSource::HumanLabel => "human_label",
        pete_now::ObjectObservationSource::Unknown => "unknown",
    }
}

fn stable_slug(value: &str) -> String {
    let mut slug = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    while slug.contains("--") {
        slug = slug.replace("--", "-");
    }
    slug.trim_matches('-').to_string()
}

fn embodied_vector_artifact(
    collection: &str,
    point_id: &str,
    embedding: &VectorEmbedding,
    frame_id: uuid::Uuid,
    source_id: String,
    occurred_at_ms: u64,
) -> VectorArtifact {
    VectorArtifact::new(collection, point_id, embedding.vector.clone())
        .with_model(embedding.model_id.clone())
        .with_source_id(source_id)
        .with_source_frame_id(frame_id.to_string())
        .with_occurred_at_ms(occurred_at_ms)
}

fn query_scene_vectors(query: &RecallQuery) -> Vec<&[f32]> {
    let mut vectors = query
        .scene_vectors
        .iter()
        .map(|artifact| artifact.vector.as_slice())
        .collect::<Vec<_>>();
    if let Some(vector) = &query.scene_vector {
        vectors.push(vector.as_slice());
    }
    vectors
}

fn place_query_vectors_from_query(query: &RecallQuery) -> Vec<VectorArtifact> {
    let mut vectors = query.scene_vectors.clone();
    if let Some(vector) = &query.scene_vector {
        vectors.push(VectorArtifact::new(
            SCENE_VECTOR_COLLECTION,
            "query:legacy-scene-vector",
            vector.clone(),
        ));
    }
    if let Some(input) = &query.place_recognition_input {
        vectors.extend(place_recognition_vectors_from_input(input));
    }
    vectors
}

