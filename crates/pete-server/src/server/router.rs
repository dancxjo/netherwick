pub fn live_view_router(state: LiveViewState) -> Router {
    state.observatory.start();
    Router::new()
        .route("/", get(live_view_page))
        .route("/now", get(get_live_now))
        .route("/models", get(get_models))
        .route("/view", get(live_view_page))
        .route("/view/snapshot", get(get_live_snapshot))
        .route("/view/vision", get(get_live_vision))
        .route("/view/embodied", get(get_live_embodied))
        .route("/view/embodied/graph", get(get_live_embodied_graph))
        .route("/api/experience/lineage", get(get_live_embodied_graph))
        .route("/debug/embodied", get(get_live_embodied))
        .route("/debug/embodied/graph", get(get_live_embodied_graph))
        .route("/view/scene", get(get_live_scene))
        .route("/view/map", get(get_live_map))
        .route("/view/behavior-nodes", get(get_behavior_nodes))
        .route("/view/behavior-nodes/{id}", post(post_behavior_node))
        .route(
            "/view/behavior-nodes/{id}/promote",
            post(post_promote_behavior_node),
        )
        .route("/view/cognitive", get(cognitive_view_page))
        .route("/view/observatory", get(observatory_page))
        .route("/api/cognitive/features", get(get_cognitive_features))
        .route("/api/cognitive/clusters", get(get_cognitive_clusters))
        .route("/api/cognitive/bindings", get(get_cognitive_bindings))
        .route("/api/cognitive/hypotheses", get(get_cognitive_hypotheses))
        .route(
            "/api/cognitive/constellations",
            get(get_cognitive_constellations),
        )
        .route(
            "/api/cognitive/associations",
            get(get_cognitive_associations),
        )
        .route("/api/cognitive/predictions", get(get_cognitive_predictions))
        .route("/api/cognitive/questions", get(get_cognitive_questions))
        .route("/api/cognitive/summary", get(get_cognitive_summary))
        .route("/view/3d", get(live_view_3d_page))
        .route("/view/capture-scene", get(get_capture_scene))
        .route("/stream/llm", get(get_llm_stream))
        .route("/view/retina-frame", post(post_retina_frame))
        .route("/view/retina/status", get(get_retina_status))
        .route("/view/retina/latest.png", get(get_retina_latest))
        .route("/view/training/latest", get(get_latest_training))
        .route("/view/inline-learning", get(get_inline_learning))
        .route("/view/inline-learning", post(post_inline_learning))
        .route("/view/calibration", post(post_calibration))
        .route("/memory/entities", get(get_entity_memory))
        .route("/api/observatory/history", get(get_observatory_history))
        .route("/api/observatory/health", get(get_observatory_health))
        .route(
            "/api/observatory/snapshots/{id}",
            get(get_observatory_now_snapshot),
        )
        .route(
            "/api/observatory/snapshot",
            get(get_observatory_now_at_or_before),
        )
        .route("/api/observatory/events/ws", get(get_observatory_events_ws))
        .route(
            "/api/observatory/provenance/{id}",
            get(get_observatory_provenance),
        )
        .route("/api/observatory/authority", get(get_observatory_authority))
        .route(
            "/api/observatory/calibration",
            get(get_observatory_calibration),
        )
        .route("/api/observatory/spatial", get(get_observatory_spatial))
        .route(
            "/api/observatory/component-health",
            get(get_observatory_component_health),
        )
        .route(
            "/api/observatory/diagnostic-export",
            get(get_observatory_diagnostic_export),
        )
        .route(
            "/api/observatory/diagnostic-verify",
            post(post_observatory_diagnostic_verify)
                .layer(DefaultBodyLimit::max(128 * 1024 * 1024)),
        )
        .route("/api/observatory/compare", get(get_observatory_compare))
        .nest_service(
            "/static",
            ServeDir::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("static")),
        )
        .with_state(state)
}

pub async fn serve_live_view(addr: SocketAddr, state: LiveViewState) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, live_view_router(state)).await
}

pub async fn serve_live_view_with_reign(
    addr: SocketAddr,
    live_state: LiveViewState,
    reign_state: ReignServerState,
) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let router = live_view_router(live_state).merge(reign_router(reign_state));
    axum::serve(listener, router).await
}

pub async fn serve_live_view_tls(
    addr: SocketAddr,
    state: LiveViewState,
    cert_path: impl AsRef<Path>,
    key_path: impl AsRef<Path>,
) -> std::io::Result<()> {
    install_rustls_crypto_provider();
    let config = RustlsConfig::from_pem_file(cert_path, key_path).await?;
    axum_server::bind_rustls(addr, config)
        .serve(live_view_router(state).into_make_service())
        .await
        .map_err(std::io::Error::other)
}

pub async fn serve_live_view_with_reign_tls(
    addr: SocketAddr,
    live_state: LiveViewState,
    reign_state: ReignServerState,
    cert_path: impl AsRef<Path>,
    key_path: impl AsRef<Path>,
) -> std::io::Result<()> {
    install_rustls_crypto_provider();
    let config = RustlsConfig::from_pem_file(cert_path, key_path).await?;
    let router = live_view_router(live_state).merge(reign_router(reign_state));
    axum_server::bind_rustls(addr, config)
        .serve(router.into_make_service())
        .await
        .map_err(std::io::Error::other)
}

fn install_rustls_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

async fn get_live_snapshot(
    State(state): State<LiveViewState>,
) -> Result<Json<LiveSnapshotResponse>, LiveViewError> {
    let snapshot = state
        .latest()
        .ok_or_else(|| LiveViewError::unavailable("no live world snapshot has arrived yet"))?;
    Ok(Json(LiveSnapshotResponse {
        t_ms: snapshot.body.last_update_ms,
        body: snapshot.body,
        range: snapshot.range,
        eye_frame: snapshot.eye_frame,
        gps: snapshot.gps,
        ear_pcm: snapshot.ear_pcm,
    }))
}

async fn get_live_vision(
    State(state): State<LiveViewState>,
) -> Result<Json<ObjectSense>, LiveViewError> {
    let snapshot = state
        .latest()
        .ok_or_else(|| LiveViewError::unavailable("no live world snapshot has arrived yet"))?;
    Ok(Json(snapshot.objects))
}

async fn get_live_embodied(
    State(state): State<LiveViewState>,
) -> Result<Json<EmbodiedContext>, LiveViewError> {
    state
        .latest_embodied_context()
        .map(Json)
        .ok_or_else(|| LiveViewError::unavailable("no embodied experience has arrived yet"))
}

async fn get_live_embodied_graph(
    State(state): State<LiveViewState>,
) -> Result<Json<EmbodiedLineageGraph>, LiveViewError> {
    let context = state
        .latest_embodied_context()
        .ok_or_else(|| LiveViewError::unavailable("no embodied experience has arrived yet"))?;
    Ok(Json(EmbodiedLineageGraph::from_context(&context)))
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EmbodiedLineageGraph {
    pub schema_version: u32,
    pub experience_id: Option<String>,
    pub summary: String,
    pub nodes: Vec<EmbodiedGraphNode>,
    pub edges: Vec<EmbodiedGraphEdge>,
    pub vector_metadata: Vec<EmbodiedGraphVector>,
    pub recent_memories: Vec<EmbodiedGraphMemory>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EmbodiedGraphNode {
    pub id: String,
    pub node_type: EmbodiedGraphNodeType,
    pub label: String,
    pub detail: Option<String>,
    pub entity_id: String,
    pub modality: Option<String>,
    pub payload_kind: Option<String>,
    pub derived: bool,
    pub vector_refs: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbodiedGraphNodeType {
    Sensation,
    Impression,
    Experience,
    Prediction,
    MemoryLink,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EmbodiedGraphEdge {
    pub from: String,
    pub to: String,
    pub relation: EmbodiedGraphEdgeType,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbodiedGraphEdgeType {
    ParentSensation,
    AboutSensation,
    SensationMember,
    ImpressionMember,
    SummarizesExperience,
    Predicts,
    MemoryLink,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EmbodiedGraphVector {
    pub index: usize,
    pub owner_node_id: String,
    pub vectorizer_id: String,
    pub model_id: String,
    pub model_label: String,
    pub dim: usize,
    pub modality: String,
    pub payload_kind: String,
    pub source_kind: String,
    pub source_sensation_id: String,
    pub purpose: String,
    pub collection: String,
    pub input_summary: String,
    pub is_fallback: bool,
    pub provenance: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EmbodiedGraphMemory {
    pub node_id: String,
    pub target_id: String,
    pub relation: String,
    pub score: f32,
    pub text: Option<String>,
}

impl EmbodiedLineageGraph {
    pub fn from_context(context: &EmbodiedContext) -> Self {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut vectors = Vec::new();
        let mut recent_memories = Vec::new();
        let experience_node_id = context.experience_id.map(|id| format!("experience:{id}"));

        if let Some(experience_id) = context.experience_id {
            nodes.push(EmbodiedGraphNode {
                id: format!("experience:{experience_id}"),
                node_type: EmbodiedGraphNodeType::Experience,
                label: "experience".to_string(),
                detail: non_empty(context.summary.clone()),
                entity_id: experience_id.to_string(),
                modality: None,
                payload_kind: None,
                derived: false,
                vector_refs: vector_refs_for_node(
                    &mut vectors,
                    format!("experience:{experience_id}"),
                    std::iter::empty(),
                ),
            });
        }

        let sensation_ids = context
            .sensations
            .iter()
            .map(|sensation| sensation.id)
            .collect::<BTreeSet<_>>();
        let impression_ids = context
            .impressions
            .iter()
            .map(|impression| impression.id)
            .collect::<BTreeSet<_>>();

        for sensation in &context.sensations {
            let node_id = format!("sensation:{}", sensation.id);
            let owned_vectors = context
                .sensation_vectors
                .iter()
                .filter(|vector| vector.source_sensation_id == sensation.id);
            let vector_refs = vector_refs_for_node(&mut vectors, node_id.clone(), owned_vectors);
            nodes.push(EmbodiedGraphNode {
                id: node_id.clone(),
                node_type: EmbodiedGraphNodeType::Sensation,
                label: sensation.kind.clone(),
                detail: sensation.summary.clone(),
                entity_id: sensation.id.to_string(),
                modality: Some(sensation.modality.as_str().to_string()),
                payload_kind: Some(sensation.payload_kind.as_str().to_string()),
                derived: sensation.parent_id.is_some(),
                vector_refs,
            });
            if let Some(experience_node_id) = &experience_node_id {
                edges.push(EmbodiedGraphEdge {
                    from: node_id,
                    to: experience_node_id.clone(),
                    relation: EmbodiedGraphEdgeType::SensationMember,
                });
            }
        }

        for edge in &context.lineage {
            if sensation_ids.contains(&edge.parent_id) && sensation_ids.contains(&edge.child_id) {
                edges.push(EmbodiedGraphEdge {
                    from: format!("sensation:{}", edge.parent_id),
                    to: format!("sensation:{}", edge.child_id),
                    relation: EmbodiedGraphEdgeType::ParentSensation,
                });
            }
        }

        for impression in &context.impressions {
            let node_id = format!("impression:{}", impression.id);
            let vector_refs = vector_refs_for_node(
                &mut vectors,
                node_id.clone(),
                impression.vector.as_ref().into_iter(),
            );
            nodes.push(EmbodiedGraphNode {
                id: node_id.clone(),
                node_type: EmbodiedGraphNodeType::Impression,
                label: impression.kind.clone(),
                detail: Some(impression.text.clone()),
                entity_id: impression.id.to_string(),
                modality: None,
                payload_kind: None,
                derived: false,
                vector_refs,
            });
            if let Some(sensation_id) = impression.sensation_id {
                if sensation_ids.contains(&sensation_id) {
                    edges.push(EmbodiedGraphEdge {
                        from: format!("sensation:{sensation_id}"),
                        to: node_id.clone(),
                        relation: EmbodiedGraphEdgeType::AboutSensation,
                    });
                }
            }
            if let (Some(experience_id), Some(experience_node_id)) =
                (context.experience_id, experience_node_id.as_ref())
            {
                if impression.experience_id.unwrap_or(experience_id) == experience_id
                    && impression_ids.contains(&impression.id)
                {
                    edges.push(EmbodiedGraphEdge {
                        from: node_id.clone(),
                        to: experience_node_id.clone(),
                        relation: EmbodiedGraphEdgeType::ImpressionMember,
                    });
                }
            }
        }

        for (index, prediction) in context.predictions.iter().enumerate() {
            let node_id = format!("prediction:{index}");
            let vector_refs = vector_refs_for_node(
                &mut vectors,
                node_id.clone(),
                prediction.vector.as_ref().into_iter(),
            );
            nodes.push(EmbodiedGraphNode {
                id: node_id.clone(),
                node_type: EmbodiedGraphNodeType::Prediction,
                label: format!("+{}ms", prediction.offset_ms),
                detail: Some(format!(
                    "{} ({:.0}% confidence)",
                    prediction.text,
                    prediction.confidence.clamp(0.0, 1.0) * 100.0
                )),
                entity_id: index.to_string(),
                modality: None,
                payload_kind: None,
                derived: false,
                vector_refs,
            });
            if let Some(experience_node_id) = &experience_node_id {
                edges.push(EmbodiedGraphEdge {
                    from: experience_node_id.clone(),
                    to: node_id,
                    relation: EmbodiedGraphEdgeType::Predicts,
                });
            }
        }

        for (index, link) in context.memory_links.iter().enumerate() {
            let node_id = format!("memory:{index}");
            nodes.push(EmbodiedGraphNode {
                id: node_id.clone(),
                node_type: EmbodiedGraphNodeType::MemoryLink,
                label: format!("{} {:.2}", link.relation, link.score),
                detail: link.text.clone().or_else(|| Some(link.target_id.clone())),
                entity_id: link.target_id.clone(),
                modality: Some("memory".to_string()),
                payload_kind: None,
                derived: false,
                vector_refs: Vec::new(),
            });
            if let Some(experience_node_id) = &experience_node_id {
                edges.push(EmbodiedGraphEdge {
                    from: experience_node_id.clone(),
                    to: node_id.clone(),
                    relation: EmbodiedGraphEdgeType::MemoryLink,
                });
            }
            recent_memories.push(EmbodiedGraphMemory {
                node_id,
                target_id: link.target_id.clone(),
                relation: link.relation.clone(),
                score: link.score,
                text: link.text.clone(),
            });
        }

        Self {
            schema_version: 1,
            experience_id: context.experience_id.map(|id| id.to_string()),
            summary: context.summary.clone(),
            nodes,
            edges,
            vector_metadata: vectors,
            recent_memories,
        }
    }
}

fn vector_refs_for_node<'a>(
    vectors: &mut Vec<EmbodiedGraphVector>,
    owner_node_id: String,
    source: impl Iterator<Item = &'a pete_experience::EmbodiedVectorRef>,
) -> Vec<usize> {
    source
        .map(|vector| {
            let index = vectors.len();
            vectors.push(EmbodiedGraphVector {
                index,
                owner_node_id: owner_node_id.clone(),
                vectorizer_id: vector.vectorizer_id.clone(),
                model_id: vector.model_id.clone(),
                model_label: vector.model_label.clone(),
                dim: vector.dim,
                modality: vector.modality.as_str().to_string(),
                payload_kind: vector.payload_kind.as_str().to_string(),
                source_kind: vector.source_kind.clone(),
                source_sensation_id: vector.source_sensation_id.to_string(),
                purpose: vector.purpose.clone(),
                collection: vector.collection.clone(),
                input_summary: vector.input_summary.clone(),
                is_fallback: vector.is_fallback,
                provenance: vector.provenance.clone(),
            });
            index
        })
        .collect()
}

fn non_empty(value: String) -> Option<String> {
    if value.trim().is_empty() {
        None
    } else {
        Some(value)
    }
}

async fn get_live_scene(
    State(state): State<LiveViewState>,
) -> Result<Json<LiveSceneResponse>, LiveViewError> {
    let snapshot = state
        .latest()
        .ok_or_else(|| LiveViewError::unavailable("no live world snapshot has arrived yet"))?;

    let rstate = state
        .retina_state
        .lock()
        .expect("retina state mutex poisoned");
    let connected = state.virtual_retina
        && rstate
            .last_received_at
            .map(|t| t.elapsed() < std::time::Duration::from_millis(1500))
            .unwrap_or(false);
    let retina_status = Some(RetinaStatusInfo {
        enabled: state.virtual_retina,
        connected,
        frames_received: rstate.frames_received,
        frames_written_to_ledger: rstate.frames_written_to_ledger,
    });

    let metadata = state.scene_metadata();
    let calibration = metadata
        .as_ref()
        .and_then(|metadata| metadata.sensor_calibration);
    let mut scene = snapshot_to_scene(
        &snapshot,
        metadata.as_ref(),
        state.session(),
        state.training_status(),
        state.prod_state(),
        state.behavior_nodes(),
        Some(&state.point_cloud_snapshot()),
        retina_status,
        state.hardware_control_status(),
    );
    scene.surface_perception = state.surface_perception(
        &snapshot,
        calibration,
        scene.action.final_selected_action.as_ref(),
    );
    Ok(Json(scene))
}

async fn get_live_map(
    State(state): State<LiveViewState>,
) -> Result<Json<LiveMapResponse>, LiveViewError> {
    let snapshot = state
        .latest()
        .ok_or_else(|| LiveViewError::unavailable("no live world snapshot has arrived yet"))?;
    let map = state.map_snapshot();
    let point_cloud = state.point_cloud_snapshot();
    let entity_report = state.entity_memory_report();
    Ok(Json(map_response_from_parts(
        &map,
        &point_cloud,
        &snapshot,
        state.scene_metadata().as_ref(),
        &entity_report,
    )))
}

async fn get_entity_memory(State(state): State<LiveViewState>) -> Json<EntityMemoryReport> {
    Json(state.entity_memory_report())
}

async fn get_cognitive_summary(
    State(state): State<LiveViewState>,
) -> Json<CognitiveDiagnosticsReport> {
    Json(state.cognitive_diagnostics_report())
}

async fn get_cognitive_features(
    State(state): State<LiveViewState>,
) -> Json<pete_memory::FeatureDiagnostics> {
    Json(state.cognitive_diagnostics_report().features)
}

async fn get_cognitive_clusters(
    State(state): State<LiveViewState>,
) -> Json<pete_memory::ClusterDiagnostics> {
    Json(state.cognitive_diagnostics_report().clusters)
}

async fn get_cognitive_bindings(
    State(state): State<LiveViewState>,
) -> Json<pete_memory::BindingDiagnostics> {
    Json(state.cognitive_diagnostics_report().bindings)
}

async fn get_cognitive_hypotheses(
    State(state): State<LiveViewState>,
) -> Json<pete_memory::HypothesisDiagnostics> {
    Json(state.cognitive_diagnostics_report().hypotheses)
}

async fn get_cognitive_constellations(
    State(state): State<LiveViewState>,
) -> Json<pete_memory::ConstellationDiagnostics> {
    Json(state.cognitive_diagnostics_report().constellations)
}

async fn get_cognitive_associations(
    State(state): State<LiveViewState>,
) -> Json<pete_memory::AssociationDiagnostics> {
    Json(state.cognitive_diagnostics_report().associations)
}

async fn get_cognitive_predictions(
    State(state): State<LiveViewState>,
) -> Json<pete_memory::PredictionDiagnostics> {
    Json(state.cognitive_diagnostics_report().predictions)
}

async fn get_cognitive_questions(
    State(state): State<LiveViewState>,
) -> Json<pete_memory::ActiveLearningDiagnostics> {
    Json(state.cognitive_diagnostics_report().active_learning)
}
