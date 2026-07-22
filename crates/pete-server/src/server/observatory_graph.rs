const DEFAULT_PROVENANCE_GRAPH_NODES: usize = 80;
const MAX_PROVENANCE_GRAPH_NODES: usize = 250;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenancePathMode {
    WhyBelieved,
    WhySelected,
    WhyRejected,
    Dependents,
    #[default]
    Full,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ProvenanceGraphQuery {
    #[serde(default)]
    pub mode: ProvenancePathMode,
    pub max_nodes: Option<usize>,
    #[serde(default)]
    pub include_artifacts: bool,
    #[serde(default)]
    pub include_assets: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceNodeKind {
    Event,
    MissingReference,
    ModelArtifact,
    ConfigurationArtifact,
    CalibrationEpoch,
    RawAsset,
    CaptureGap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceSemantics {
    DirectEvidence,
    Inference,
    Recall,
    LearnedOutput,
    Configuration,
    HumanInstruction,
    LlmHypothesis,
    RecordedEvent,
    Missing,
    Gap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProvenanceGraphNode {
    pub id: String,
    pub node_kind: ProvenanceNodeKind,
    pub semantics: ProvenanceSemantics,
    pub label: String,
    pub event_type: Option<BrainEventType>,
    pub source: Option<ProducerIdentity>,
    pub confidence: Option<f32>,
    pub uncertainty: Option<pete_events::Uncertainty>,
    pub freshness: Option<pete_events::Freshness>,
    pub trust: Option<TrustState>,
    pub disposition: Option<EventDisposition>,
    pub occurred_at_ms: Option<u64>,
    pub expandable: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceEdgeKind {
    Parent,
    Supports,
    Contradicts,
    Dependent,
    ProducedByModel,
    UsedConfiguration,
    CalibratedBy,
    RawPayload,
    InterruptedByGap,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceGraphEdge {
    pub from: String,
    pub to: String,
    pub relation: ProvenanceEdgeKind,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProvenanceGraphResponse {
    pub center_id: String,
    pub mode: ProvenancePathMode,
    pub nodes: Vec<ProvenanceGraphNode>,
    pub edges: Vec<ProvenanceGraphEdge>,
    pub truncated: bool,
    pub omitted_nodes: usize,
    pub missing_references: usize,
    pub capture_gaps: usize,
}

impl ProvenanceGraphQuery {
    fn max_nodes(&self) -> Result<usize, BrainEventQueryError> {
        let max_nodes = self.max_nodes.unwrap_or(DEFAULT_PROVENANCE_GRAPH_NODES);
        if max_nodes == 0 || max_nodes > MAX_PROVENANCE_GRAPH_NODES {
            return Err(BrainEventQueryError::new(format!(
                "max_nodes must be between 1 and {MAX_PROVENANCE_GRAPH_NODES}"
            )));
        }
        Ok(max_nodes)
    }
}

fn build_provenance_graph(
    events: &[BrainEvent],
    center_id: &str,
    query: &ProvenanceGraphQuery,
) -> Result<ProvenanceGraphResponse, BrainEventQueryError> {
    let max_nodes = query.max_nodes()?;
    let by_id: BTreeMap<&str, &BrainEvent> = events
        .iter()
        .map(|event| (event.event_id.0.as_str(), event))
        .collect();
    if !by_id.contains_key(center_id) {
        return Err(BrainEventQueryError::new(format!(
            "event {center_id} is not retained"
        )));
    }
    let mut reverse: BTreeMap<&str, Vec<&BrainEvent>> = BTreeMap::new();
    for event in events {
        for reference in event.causal_references() {
            reverse
                .entry(reference.target.event_id.0.as_str())
                .or_default()
                .push(event);
        }
    }

    let mut queue = VecDeque::from([center_id.to_string()]);
    let mut seen = BTreeSet::new();
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut omitted_nodes = 0;
    let mut missing_references = 0;
    while let Some(id) = queue.pop_front() {
        if !seen.insert(id.clone()) {
            continue;
        }
        if nodes.len() == max_nodes {
            omitted_nodes += 1;
            continue;
        }
        let Some(event) = by_id.get(id.as_str()).copied() else {
            nodes.push(missing_node(&id));
            missing_references += 1;
            continue;
        };
        nodes.push(event_node(event));
        let ancestors = !matches!(query.mode, ProvenancePathMode::Dependents);
        let descendants = matches!(query.mode, ProvenancePathMode::Dependents | ProvenancePathMode::Full);
        if ancestors {
            enqueue_links(event, query.mode, &mut queue, &mut edges);
        }
        if descendants {
            for dependent in reverse.get(id.as_str()).into_iter().flatten() {
                edges.push(ProvenanceGraphEdge {
                    from: id.clone(),
                    to: dependent.event_id.0.clone(),
                    relation: ProvenanceEdgeKind::Dependent,
                });
                queue.push_back(dependent.event_id.0.clone());
            }
        }
        if query.include_artifacts {
            append_artifacts(event, &mut nodes, &mut edges, max_nodes, &mut omitted_nodes);
        }
        for epoch in &event.calibration_epochs {
            append_auxiliary(
                event,
                format!("calibration:{epoch}"),
                epoch.clone(),
                ProvenanceNodeKind::CalibrationEpoch,
                ProvenanceSemantics::Configuration,
                ProvenanceEdgeKind::CalibratedBy,
                &mut nodes,
                &mut edges,
                max_nodes,
                &mut omitted_nodes,
            );
        }
        if query.include_assets {
            if let BrainEventPayload::Reference { reference } = &event.payload {
                append_auxiliary(
                    event,
                    format!("asset:{}", reference.id),
                    format!("{} ({})", reference.id, reference.media_type),
                    ProvenanceNodeKind::RawAsset,
                    ProvenanceSemantics::DirectEvidence,
                    ProvenanceEdgeKind::RawPayload,
                    &mut nodes,
                    &mut edges,
                    max_nodes,
                    &mut omitted_nodes,
                );
            }
        }
    }

    let event_times: Vec<u64> = nodes.iter().filter_map(|node| node.occurred_at_ms).collect();
    let min_time = event_times.iter().min().copied();
    let max_time = event_times.iter().max().copied();
    let mut capture_gaps = 0;
    for gap in events.iter().filter(|event| {
        event.event_type == BrainEventType::TransportGap
            && min_time.is_none_or(|min| event.times.occurred.t_ms >= min)
            && max_time.is_none_or(|max| event.times.occurred.t_ms <= max)
    }) {
        if nodes.len() == max_nodes {
            omitted_nodes += 1;
            continue;
        }
        nodes.push(ProvenanceGraphNode {
            node_kind: ProvenanceNodeKind::CaptureGap,
            semantics: ProvenanceSemantics::Gap,
            ..event_node(gap)
        });
        edges.push(ProvenanceGraphEdge {
            from: center_id.to_string(),
            to: gap.event_id.0.clone(),
            relation: ProvenanceEdgeKind::InterruptedByGap,
        });
        capture_gaps += 1;
    }
    edges.retain(|edge| {
        nodes.iter().any(|node| node.id == edge.from)
            && nodes.iter().any(|node| node.id == edge.to)
    });
    edges.sort_by(|left, right| {
        (&left.from, &left.to, edge_rank(left.relation))
            .cmp(&(&right.from, &right.to, edge_rank(right.relation)))
    });
    edges.dedup();
    Ok(ProvenanceGraphResponse {
        center_id: center_id.to_string(),
        mode: query.mode,
        truncated: omitted_nodes > 0,
        omitted_nodes,
        missing_references,
        capture_gaps,
        nodes,
        edges,
    })
}

fn enqueue_links(
    event: &BrainEvent,
    mode: ProvenancePathMode,
    queue: &mut VecDeque<String>,
    edges: &mut Vec<ProvenanceGraphEdge>,
) {
    let include_contradictions = matches!(
        mode,
        ProvenancePathMode::WhyBelieved | ProvenancePathMode::WhyRejected | ProvenancePathMode::Full
    );
    for (references, relation) in [
        (&event.links.parents, ProvenanceEdgeKind::Parent),
        (&event.links.supports, ProvenanceEdgeKind::Supports),
    ] {
        for reference in references {
            edges.push(ProvenanceGraphEdge {
                from: event.event_id.0.clone(),
                to: reference.event_id.0.clone(),
                relation,
            });
            queue.push_back(reference.event_id.0.clone());
        }
    }
    if include_contradictions {
        for reference in &event.links.contradicts {
            edges.push(ProvenanceGraphEdge {
                from: event.event_id.0.clone(),
                to: reference.event_id.0.clone(),
                relation: ProvenanceEdgeKind::Contradicts,
            });
            queue.push_back(reference.event_id.0.clone());
        }
    }
}

fn event_node(event: &BrainEvent) -> ProvenanceGraphNode {
    ProvenanceGraphNode {
        id: event.event_id.0.clone(),
        node_kind: ProvenanceNodeKind::Event,
        semantics: event_semantics(event),
        label: event.kind.clone(),
        event_type: Some(event.event_type),
        source: Some(event.producer.clone()),
        confidence: event.quality.confidence,
        uncertainty: event.quality.uncertainty.clone(),
        freshness: Some(event.quality.freshness.clone()),
        trust: Some(event.quality.trust),
        disposition: Some(event.disposition),
        occurred_at_ms: Some(event.times.occurred.t_ms),
        expandable: !event.links.parents.is_empty()
            || !event.links.supports.is_empty()
            || !event.links.contradicts.is_empty(),
    }
}

fn event_semantics(event: &BrainEvent) -> ProvenanceSemantics {
    let component = event.producer.component.to_ascii_lowercase();
    let kind = event.kind.to_ascii_lowercase();
    if event.producer.brain == Brain::Human {
        ProvenanceSemantics::HumanInstruction
    } else if component.contains("llm") || kind.contains("llm") {
        ProvenanceSemantics::LlmHypothesis
    } else if kind.contains("recall") || component.contains("memory") {
        ProvenanceSemantics::Recall
    } else if event
        .artifacts
        .iter()
        .any(|artifact| artifact.kind == ArtifactKind::Model)
    {
        ProvenanceSemantics::LearnedOutput
    } else if event.event_type == BrainEventType::Evidence {
        ProvenanceSemantics::DirectEvidence
    } else if matches!(
        event.event_type,
        BrainEventType::Interpretation | BrainEventType::BeliefUpdate
    ) {
        ProvenanceSemantics::Inference
    } else {
        ProvenanceSemantics::RecordedEvent
    }
}

fn missing_node(id: &str) -> ProvenanceGraphNode {
    ProvenanceGraphNode {
        id: id.to_string(),
        node_kind: ProvenanceNodeKind::MissingReference,
        semantics: ProvenanceSemantics::Missing,
        label: "missing or no longer retained".to_string(),
        event_type: None,
        source: None,
        confidence: None,
        uncertainty: None,
        freshness: None,
        trust: None,
        disposition: None,
        occurred_at_ms: None,
        expandable: false,
    }
}

fn append_artifacts(
    event: &BrainEvent,
    nodes: &mut Vec<ProvenanceGraphNode>,
    edges: &mut Vec<ProvenanceGraphEdge>,
    max_nodes: usize,
    omitted_nodes: &mut usize,
) {
    for artifact in &event.artifacts {
        let (node_kind, semantics, relation) = match artifact.kind {
            ArtifactKind::Model => (
                ProvenanceNodeKind::ModelArtifact,
                ProvenanceSemantics::LearnedOutput,
                ProvenanceEdgeKind::ProducedByModel,
            ),
            ArtifactKind::Configuration => (
                ProvenanceNodeKind::ConfigurationArtifact,
                ProvenanceSemantics::Configuration,
                ProvenanceEdgeKind::UsedConfiguration,
            ),
            _ => continue,
        };
        append_auxiliary(
            event,
            format!("artifact:{:?}:{}", artifact.kind, artifact.id).to_lowercase(),
            artifact.id.clone(),
            node_kind,
            semantics,
            relation,
            nodes,
            edges,
            max_nodes,
            omitted_nodes,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn append_auxiliary(
    event: &BrainEvent,
    id: String,
    label: String,
    node_kind: ProvenanceNodeKind,
    semantics: ProvenanceSemantics,
    relation: ProvenanceEdgeKind,
    nodes: &mut Vec<ProvenanceGraphNode>,
    edges: &mut Vec<ProvenanceGraphEdge>,
    max_nodes: usize,
    omitted_nodes: &mut usize,
) {
    if nodes.iter().any(|node| node.id == id) {
        edges.push(ProvenanceGraphEdge {
            from: event.event_id.0.clone(),
            to: id,
            relation,
        });
    } else if nodes.len() < max_nodes {
        nodes.push(ProvenanceGraphNode {
            id: id.clone(),
            node_kind,
            semantics,
            label,
            event_type: None,
            source: None,
            confidence: None,
            uncertainty: None,
            freshness: None,
            trust: None,
            disposition: None,
            occurred_at_ms: None,
            expandable: false,
        });
        edges.push(ProvenanceGraphEdge {
            from: event.event_id.0.clone(),
            to: id,
            relation,
        });
    } else {
        *omitted_nodes += 1;
    }
}

fn edge_rank(kind: ProvenanceEdgeKind) -> u8 {
    kind as u8
}

async fn get_observatory_provenance(
    State(state): State<LiveViewState>,
    AxumPath(event_id): AxumPath<String>,
    Query(query): Query<ProvenanceGraphQuery>,
) -> Result<Json<ProvenanceGraphResponse>, ObservatoryHttpError> {
    let history = state
        .observatory()
        .query(&BrainEventQuery {
            limit: Some(MAX_OBSERVATORY_QUERY_LIMIT),
            ..BrainEventQuery::default()
        })
        .map_err(|error| ObservatoryHttpError::bad_request(error.to_string()))?;
    let events: Vec<BrainEvent> = history
        .records
        .into_iter()
        .map(|record| match record {
            BrainEventStreamRecord::Event { envelope } => envelope.event,
            BrainEventStreamRecord::Gap { gap } => gap.event,
        })
        .collect();
    build_provenance_graph(&events, &event_id, &query)
        .map(Json)
        .map_err(|error| ObservatoryHttpError::bad_request(error.to_string()))
}
