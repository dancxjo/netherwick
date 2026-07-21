#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub frame_id: uuid::Uuid,
    pub t_ms: u64,
    pub summary: String,
    #[serde(default)]
    pub graph_entities: Vec<GraphEntity>,
    #[serde(default)]
    pub graph_relationships: Vec<GraphEdge>,
    #[serde(default)]
    pub scene_vectors: Vec<VectorArtifact>,
    #[serde(default)]
    pub face_vectors: Vec<VectorArtifact>,
    #[serde(default)]
    pub object_vectors: Vec<VectorArtifact>,
    #[serde(default)]
    pub voice_vectors: Vec<VectorArtifact>,
    #[serde(default)]
    pub sensation_vectors: Vec<VectorArtifact>,
    #[serde(default)]
    pub experience_vectors: Vec<VectorArtifact>,
    #[serde(default)]
    pub vector_payloads: BTreeMap<String, serde_json::Value>,
    pub battery: f32,
    pub active_goal: Option<Goal>,
    pub chosen_action: Option<ActionPrimitive>,
    pub warning: Option<String>,
    pub experience: Option<Experience>,
    #[serde(default)]
    pub temporal_context: TemporalContext,
    #[serde(default)]
    pub social_world: SocialWorldSnapshot,
    #[serde(default)]
    pub epistemic_state: EpistemicSnapshot,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QdrantConfig {
    pub url: String,
}

impl QdrantConfig {
    pub fn from_env() -> Option<Self> {
        std::env::var("PETE_QDRANT_URL")
            .ok()
            .filter(|url| !url.trim().is_empty())
            .map(|url| Self { url })
    }
}

#[derive(Clone)]
pub struct QdrantVectorStore {
    client: reqwest::Client,
    config: QdrantConfig,
}

impl QdrantVectorStore {
    pub fn new(config: QdrantConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
        }
    }

    pub fn from_env() -> Option<Self> {
        QdrantConfig::from_env().map(Self::new)
    }

    async fn ensure_collection(&self, collection: &str, vector_size: usize) -> Result<()> {
        if vector_size == 0 {
            return Ok(());
        }
        let url = format!(
            "{}/collections/{}",
            self.config.url.trim_end_matches('/'),
            collection
        );
        let response = self
            .client
            .put(url)
            .json(&json!({
                "vectors": {
                    "size": vector_size,
                    "distance": "Cosine"
                }
            }))
            .send()
            .await
            .context("creating qdrant collection")?;
        if response.status().is_success() || response.status() == StatusCode::CONFLICT {
            return Ok(());
        }
        Err(anyhow!(
            "qdrant collection create failed for {collection}: HTTP {}",
            response.status()
        ))
    }
}

#[async_trait]
impl VectorStore for QdrantVectorStore {
    async fn upsert_vectors(&self, record: &MemoryRecord) -> Result<()> {
        let mut by_collection: BTreeMap<&str, Vec<&VectorArtifact>> = BTreeMap::new();
        for artifact in record_all_vectors(record) {
            by_collection
                .entry(artifact.collection.as_str())
                .or_default()
                .push(artifact);
        }

        for (collection, artifacts) in by_collection {
            let Some(first) = artifacts.first() else {
                continue;
            };
            self.ensure_collection(collection, first.vector.len())
                .await?;
            let points = artifacts
                .into_iter()
                .filter(|artifact| !artifact.vector.is_empty())
                .map(|artifact| {
                    let mut payload = json!({
                        "collection": artifact.collection,
                        "point_id": artifact.point_id,
                        "frame_id": record.frame_id.to_string(),
                        "source_frame_id": artifact.source_frame_id,
                        "source_id": artifact.source_id,
                        "model": artifact.model,
                        "dim": artifact.vector.len(),
                        "occurred_at_ms": artifact.occurred_at_ms.or(Some(record.t_ms)),
                        "summary": record.summary,
                        "neo4j_node_id": vector_node_id(artifact),
                    });
                    if let Some(extra) = record.vector_payloads.get(&vector_payload_key(artifact)) {
                        merge_json_object(&mut payload, extra);
                    }
                    json!({
                        "id": stable_qdrant_point_id(&artifact.collection, &artifact.point_id),
                        "vector": artifact.vector,
                        "payload": payload
                    })
                })
                .collect::<Vec<_>>();
            if points.is_empty() {
                continue;
            }
            let url = format!(
                "{}/collections/{}/points?wait=true",
                self.config.url.trim_end_matches('/'),
                collection
            );
            let response = self
                .client
                .put(url)
                .json(&json!({ "points": points }))
                .send()
                .await
                .with_context(|| format!("upserting qdrant points into {collection}"))?;
            if !response.status().is_success() {
                return Err(anyhow!(
                    "qdrant upsert failed for {collection}: HTTP {}",
                    response.status()
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Neo4jConfig {
    pub http_url: String,
    pub user: String,
    pub password: String,
    pub database: String,
}

impl Neo4jConfig {
    pub fn from_env() -> Option<Self> {
        let user = std::env::var("PETE_NEO4J_USER").ok()?;
        let password = std::env::var("PETE_NEO4J_PASSWORD").ok()?;
        let http_url = std::env::var("PETE_NEO4J_HTTP_URL")
            .ok()
            .or_else(|| {
                std::env::var("PETE_NEO4J_URI")
                    .ok()
                    .and_then(|uri| neo4j_http_url_from_uri(&uri))
            })
            .unwrap_or_else(|| "http://localhost:7474".to_string());
        let database = std::env::var("PETE_NEO4J_DATABASE").unwrap_or_else(|_| "neo4j".to_string());
        Some(Self {
            http_url,
            user,
            password,
            database,
        })
    }
}

#[derive(Clone)]
pub struct Neo4jGraphStore {
    client: reqwest::Client,
    config: Neo4jConfig,
    legacy_related_migration: Arc<tokio::sync::OnceCell<()>>,
}

impl Neo4jGraphStore {
    pub fn new(config: Neo4jConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
            legacy_related_migration: Arc::new(tokio::sync::OnceCell::new()),
        }
    }

    pub fn from_env() -> Option<Self> {
        Neo4jConfig::from_env().map(Self::new)
    }

    async fn migrate_legacy_related_edges(&self) -> Result<()> {
        self.legacy_related_migration
            .get_or_try_init(|| async {
                self.run_cypher(NEO4J_LEGACY_RELATED_EDGE_MIGRATION_CYPHER, json!({}))
                    .await
            })
            .await?;
        Ok(())
    }

    async fn run_cypher(&self, statement: &str, parameters: serde_json::Value) -> Result<()> {
        let url = format!(
            "{}/db/{}/tx/commit",
            self.config.http_url.trim_end_matches('/'),
            self.config.database
        );
        let response = self
            .client
            .post(url)
            .basic_auth(&self.config.user, Some(&self.config.password))
            .json(&json!({
                "statements": [{
                    "statement": statement,
                    "parameters": parameters
                }]
            }))
            .send()
            .await
            .context("running neo4j cypher")?;
        if !response.status().is_success() {
            return Err(anyhow!("neo4j cypher failed: HTTP {}", response.status()));
        }
        let body = response
            .json::<serde_json::Value>()
            .await
            .context("reading neo4j response")?;
        let errors = body
            .get("errors")
            .and_then(|value| value.as_array())
            .cloned()
            .unwrap_or_default();
        if !errors.is_empty() {
            return Err(anyhow!("neo4j cypher errors: {errors:?}"));
        }
        Ok(())
    }

    async fn query_cypher(
        &self,
        statement: &str,
        parameters: serde_json::Value,
    ) -> Result<Vec<Vec<serde_json::Value>>> {
        let url = format!(
            "{}/db/{}/tx/commit",
            self.config.http_url.trim_end_matches('/'),
            self.config.database
        );
        let response = self
            .client
            .post(url)
            .basic_auth(&self.config.user, Some(&self.config.password))
            .json(&json!({
                "statements": [{
                    "statement": statement,
                    "parameters": parameters,
                    "resultDataContents": ["row"]
                }]
            }))
            .send()
            .await
            .context("querying neo4j cypher")?;
        if !response.status().is_success() {
            return Err(anyhow!("neo4j query failed: HTTP {}", response.status()));
        }
        let body = response
            .json::<serde_json::Value>()
            .await
            .context("reading neo4j query response")?;
        let errors = body
            .get("errors")
            .and_then(|value| value.as_array())
            .cloned()
            .unwrap_or_default();
        if !errors.is_empty() {
            return Err(anyhow!("neo4j query errors: {errors:?}"));
        }
        Ok(body
            .get("results")
            .and_then(|results| results.get(0))
            .and_then(|result| result.get("data"))
            .and_then(|data| data.as_array())
            .into_iter()
            .flatten()
            .filter_map(|item| item.get("row").and_then(|row| row.as_array()).cloned())
            .collect())
    }
}

#[async_trait]
impl GraphStore for Neo4jGraphStore {
    async fn upsert_graph(&self, record: &MemoryRecord) -> Result<()> {
        // Releases before stable edge ids merged RELATED edges by kind. Backfill
        // the surviving projection in one transaction before any new merge;
        // never delete historical relationships during an ordinary write.
        self.migrate_legacy_related_edges().await?;

        let entities = neo4j_entity_params(record);
        let relationships = neo4j_relationship_params(record);

        self.run_cypher(
            NEO4J_GRAPH_UPSERT_CYPHER,
            json!({
                "entities": entities,
                "relationships": relationships,
            }),
        )
        .await
    }
}

const NEO4J_LEGACY_RELATED_EDGE_MIGRATION_CYPHER: &str = r#"
MATCH (from:MemoryNode)-[legacy:RELATED]->(to:MemoryNode)
WHERE legacy.edge_id IS NULL
SET legacy.edge_id =
    CASE
        WHEN legacy.kind STARTS WITH 'SEMANTIC_'
             AND legacy.payload_json CONTAINS '"id":"'
        THEN split(split(legacy.payload_json, '"id":"')[1], '"')[0]
        ELSE 'graph-edge:'
             + toString(size(from.id)) + ':' + from.id + ':'
             + toString(size(legacy.kind)) + ':' + legacy.kind + ':'
             + toString(size(to.id)) + ':' + to.id
    END,
    legacy.edge_identity_migrated = true
"#;

const NEO4J_GRAPH_UPSERT_CYPHER: &str = r#"
UNWIND $entities AS entity
MERGE (n:MemoryNode {id: entity.id})
SET n.labels = entity.labels,
    n.summary = entity.summary,
    n.score = entity.score,
    n.frame_id = entity.frame_id,
    n.t_ms = entity.t_ms
WITH collect(n) AS ignored
UNWIND $relationships AS relationship
MATCH (from:MemoryNode {id: relationship.from})
MATCH (to:MemoryNode {id: relationship.to})
MERGE (from)-[r:RELATED {edge_id: relationship.edge_id}]->(to)
SET r.kind = relationship.kind,
    r.summary = relationship.summary,
    r.score = relationship.score,
    r.payload_json = relationship.payload_json,
    r.frame_id = relationship.frame_id,
    r.t_ms = relationship.t_ms
REMOVE r.payload
"#;

#[async_trait]
impl GraphIntelligence for Neo4jGraphStore {
    async fn upsert_intelligence(&self, document: &GraphIntelligenceDocument) -> Result<()> {
        let params = neo4j_intelligence_params(document);
        for statement in neo4j_intelligence_upsert_statements() {
            self.run_cypher(statement, params.clone()).await?;
        }
        Ok(())
    }

    async fn feature_or_cluster_intelligence(
        &self,
        node_id: &str,
        limit: usize,
    ) -> Result<FeatureClusterIntelligence> {
        let rows = self
            .query_cypher(
                r#"
MATCH (n {id: $id})
OPTIONAL MATCH (n)-[:MEMBER_OF]->(cluster:Cluster)
OPTIONAL MATCH (n)-[:BOUND_TO]-(bound:Cluster)
OPTIONAL MATCH (bc:BindingCandidate)-[:FROM|TO]->(n)
OPTIONAL MATCH (bc)-[:SUPPORTED_BY]->(support:Evidence)
OPTIONAL MATCH (bc)-[:REJECTED_BECAUSE]->(contradiction:Evidence)
OPTIONAL MATCH (co:Constellation)-[:HAS_MEMBER]->(n)
RETURN collect(DISTINCT cluster)[0..$limit],
       collect(DISTINCT bound)[0..$limit],
       collect(DISTINCT bc)[0..$limit],
       collect(DISTINCT support)[0..$limit],
       collect(DISTINCT contradiction)[0..$limit],
       collect(DISTINCT co)[0..$limit]
"#,
                json!({"id": node_id, "limit": limit as i64}),
            )
            .await?;
        let row = rows.first().cloned().unwrap_or_default();
        Ok(FeatureClusterIntelligence {
            node_id: node_id.to_string(),
            clusters: summaries_from_row(row.first()),
            bindings: summaries_from_row(row.get(1))
                .into_iter()
                .chain(summaries_from_row(row.get(2)))
                .collect(),
            supporting_evidence: summaries_from_row(row.get(3)),
            contradictions: summaries_from_row(row.get(4)),
            constellations: summaries_from_row(row.get(5)),
        })
    }

    async fn constellation_intelligence(
        &self,
        constellation_id: &str,
        limit: usize,
    ) -> Result<ConstellationIntelligence> {
        let rows = self
            .query_cypher(
                r#"
MATCH (co:Constellation {id: $id})
OPTIONAL MATCH (co)-[:HAS_MEMBER]->(member)
OPTIONAL MATCH (co)-[:SUPPORTED_BY]->(binding:BindingEdge)
OPTIONAL MATCH (co)<-[:TO]-(a:Association)-[:TO]->(prediction)
OPTIONAL MATCH (similar:Constellation)
WHERE similar.id <> co.id AND any(id IN co.member_cluster_ids WHERE id IN similar.member_cluster_ids)
OPTIONAL MATCH (co)<-[:CRITIQUES]-(lr:LlmReview)
RETURN co,
       collect(DISTINCT member)[0..$limit] + collect(DISTINCT binding)[0..$limit],
       collect(DISTINCT prediction)[0..$limit],
       collect(DISTINCT similar)[0..$limit],
       collect(DISTINCT lr)[0..$limit]
"#,
                json!({"id": constellation_id, "limit": limit as i64}),
            )
            .await?;
        let row = rows.first().cloned().unwrap_or_default();
        let constellation = row.first().map(summary_from_value).unwrap_or_default();
        let members = summaries_from_row(row.get(1));
        let known_member_ids = members
            .iter()
            .map(|member| member.id.clone())
            .collect::<BTreeSet<_>>();
        let expected = row
            .first()
            .and_then(|value| value.get("member_cluster_ids"))
            .and_then(|value| value.as_array())
            .into_iter()
            .flatten()
            .filter_map(|value| value.as_str().map(str::to_string))
            .collect::<Vec<_>>();
        Ok(ConstellationIntelligence {
            constellation_id: constellation_id.to_string(),
            state: constellation.state.clone().unwrap_or_default(),
            missing_members: expected
                .into_iter()
                .filter(|id| !known_member_ids.contains(id))
                .collect(),
            predictions: summaries_from_row(row.get(2)),
            similar_constellations: summaries_from_row(row.get(3)),
            contradictions: summaries_from_row(row.get(4)),
            stability: row
                .first()
                .and_then(|value| value.get("stability"))
                .and_then(value_as_f32)
                .unwrap_or_default(),
            reason: constellation.reason,
            members,
        })
    }

    async fn ambiguity_intelligence(
        &self,
        family_or_target_id: &str,
        limit: usize,
    ) -> Result<AmbiguityIntelligence> {
        let rows = self
            .query_cypher(
                r#"
MATCH (h:TrackingHypothesis)
WHERE h.family_id = $id OR h.target_id = $id OR h.id = $id
OPTIONAL MATCH (h)-[:SUPPORTS]->(bc:BindingCandidate)
OPTIONAL MATCH (bc)-[:SUPPORTED_BY|REJECTED_BECAUSE]->(e:Evidence)
OPTIONAL MATCH (h)<-[:CRITIQUES]-(lr:LlmReview)
RETURN collect(DISTINCT h)[0..$limit],
       collect(DISTINCT e)[0..$limit],
       collect(DISTINCT lr)[0..$limit]
"#,
                json!({"id": family_or_target_id, "limit": limit as i64}),
            )
            .await?;
        let row = rows.first().cloned().unwrap_or_default();
        let reviews = summaries_from_row(row.get(2));
        let question = reviews
            .iter()
            .find(|review| !review.reason.is_empty())
            .map(|review| review.reason.clone())
            .or_else(|| {
                Some(format!(
                    "Which hypothesis best explains {family_or_target_id}?"
                ))
            });
        Ok(AmbiguityIntelligence {
            target_id: family_or_target_id.to_string(),
            competing_hypotheses: summaries_from_row(row.first()),
            distinguishing_evidence: summaries_from_row(row.get(1)),
            contradictions: reviews.clone(),
            human_question: question,
        })
    }

    async fn action_outcome_intelligence(
        &self,
        action_id: &str,
        limit: usize,
    ) -> Result<ActionOutcomeIntelligence> {
        let rows = self
            .query_cypher(
                r#"
MATCH (a:ActionIntent {id: $id})
OPTIONAL MATCH (a)-[:RESULTED_IN]->(outcome:Outcome)
OPTIONAL MATCH (a)<-[:FROM]-(assoc:Association)-[:TO]->(next)
OPTIONAL MATCH (place:Place)<-[:FROM]-(risk:Association {relation: 'prevents'})
OPTIONAL MATCH (body:BodyState)<-[:FROM]-(prevent:Association {relation: 'prevents'})
RETURN collect(DISTINCT outcome)[0..$limit],
       collect(DISTINCT next)[0..$limit],
       collect(DISTINCT place)[0..$limit],
       collect(DISTINCT body)[0..$limit]
"#,
                json!({"id": action_id, "limit": limit as i64}),
            )
            .await?;
        let row = rows.first().cloned().unwrap_or_default();
        Ok(ActionOutcomeIntelligence {
            action_id: action_id.to_string(),
            outcomes: summaries_from_row(row.first()),
            usual_next: summaries_from_row(row.get(1)),
            risky_places: summaries_from_row(row.get(2)),
            preventing_body_states: summaries_from_row(row.get(3)),
        })
    }

    async fn local_community(
        &self,
        start_node_id: &str,
        max_depth: u32,
        min_weight: f32,
        limit: usize,
    ) -> Result<GraphCommunity> {
        let rows = self
            .query_cypher(
                r#"
MATCH path = (start {id: $id})-[rels*1..4]-(node)
WHERE length(path) <= $max_depth
WITH node, rels, length(path) AS depth,
     reduce(score = 0.0, r IN rels | score + coalesce(r.confidence, r.score, 0.1)) AS weight
WHERE weight >= $min_weight
RETURN node.id, labels(node), weight, depth, size(rels),
       coalesce(node.reason, node.summary, node.current_state, '')
ORDER BY weight DESC, depth ASC
LIMIT $limit
"#,
                json!({
                    "id": start_node_id,
                    "max_depth": max_depth.min(4) as i64,
                    "min_weight": min_weight,
                    "limit": limit as i64,
                }),
            )
            .await?;
        let members = rows
            .into_iter()
            .map(|row| GraphCommunityMember {
                node_id: row
                    .first()
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                labels: row
                    .get(1)
                    .and_then(|v| v.as_array())
                    .into_iter()
                    .flatten()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect(),
                score: row.get(2).and_then(value_as_f32).unwrap_or_default(),
                depth: row.get(3).and_then(|v| v.as_u64()).unwrap_or_default() as u32,
                recurrence: row.get(4).and_then(|v| v.as_u64()).unwrap_or_default() as u32,
                reason: row
                    .get(5)
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
            })
            .collect::<Vec<_>>();
        Ok(GraphCommunity {
            start_node_id: start_node_id.to_string(),
            max_depth,
            min_weight,
            summary: format!("{} strong nearby graph members", members.len()),
            members,
        })
    }

    async fn graph_recall(
        &self,
        query: GraphRecallQuery,
        limit: usize,
    ) -> Result<GraphRecallBundle> {
        let query_ids = graph_recall_query_ids(&query);
        let rows = self
            .query_cypher(
                r#"
MATCH (seed)
WHERE seed.id IN $ids
OPTIONAL MATCH (seed)-[*1..2]-(near:MemoryNode)
OPTIONAL MATCH (co:Constellation)
WHERE co.id IN $ids OR any(id IN co.member_cluster_ids WHERE id IN $ids)
OPTIONAL MATCH (seed)<-[:FROM]-(assoc:Association)-[:TO]->(outcome:Outcome)
OPTIONAL MATCH (seed)<-[:REVIEWS|CRITIQUES]-(review)
OPTIONAL MATCH (hr:HumanReview)-[:CONFIRMS]->(seed)
OPTIONAL MATCH (lr:LlmReview)-[:CRITIQUES]->(seed)
OPTIONAL MATCH (ai:ActionIntent)-[:RESULTED_IN]->(action_outcome:Outcome)
WHERE ai.id IN $ids
RETURN collect(DISTINCT near)[0..$limit],
       collect(DISTINCT co)[0..$limit],
       collect(DISTINCT outcome)[0..$limit],
       collect(DISTINCT review)[0..$limit],
       collect(DISTINCT hr)[0..$limit],
       collect(DISTINCT lr)[0..$limit],
       collect(DISTINCT action_outcome)[0..$limit]
"#,
                json!({"ids": query_ids, "limit": limit as i64}),
            )
            .await?;
        let row = rows.first().cloned().unwrap_or_default();
        let action_outcomes = summaries_from_row(row.get(6));
        Ok(GraphRecallBundle {
            query_ids: graph_recall_query_ids(&query),
            nearby_memories: summaries_from_row(row.first()),
            similar_constellations: summaries_from_row(row.get(1)),
            likely_outcomes: summaries_from_row(row.get(2)),
            previous_contradictions: summaries_from_row(row.get(3)),
            human_confirmations: summaries_from_row(row.get(4)),
            llm_critiques: summaries_from_row(row.get(5)),
            action_successes: action_outcomes
                .iter()
                .filter(|summary| summary.state.as_deref() == Some("succeeded"))
                .cloned()
                .collect(),
            action_failures: action_outcomes
                .into_iter()
                .filter(|summary| summary.state.as_deref() == Some("failed"))
                .collect(),
        })
    }

    async fn consistency_checks(&self, limit: usize) -> Result<Vec<GraphReviewRecord>> {
        let rows = self
            .query_cypher(
                r#"
MATCH (target)
WHERE (target:BindingCandidate AND target.decision IN ['reject', 'hold_ambiguous', 'ask_human'])
   OR (target:TrackingHypothesis AND target.current_state IN ['needs_review', 'rejected'])
   OR (target:Constellation AND target.current_state IN ['ambiguous', 'split_needed', 'merge_needed'])
   OR (target:Association AND coalesce(target.contradiction_count, 0) > 0)
   OR (target:Prediction)-[:FAILED_WITH]->(:Surprise)
RETURN target.id, labels(target)[0], coalesce(target.confidence, 0.5),
       coalesce(target.last_updated_ms, target.last_seen_ms, target.t_ms, 0),
       coalesce(target.reason, target.current_state, target.decision, 'suspicious graph state')
ORDER BY coalesce(target.last_updated_ms, target.last_seen_ms, target.t_ms, 0) DESC
LIMIT $limit
"#,
                json!({"limit": limit as i64}),
            )
            .await?;
        Ok(rows
            .into_iter()
            .enumerate()
            .map(|(index, row)| {
                let target_id = row.first().and_then(|v| v.as_str()).unwrap_or_default();
                let kind = row.get(1).and_then(|v| v.as_str()).unwrap_or("graph");
                GraphReviewRecord {
                    id: format!(
                        "graph-review:{}:{}",
                        stable_slug(kind),
                        stable_slug(target_id)
                    ),
                    target_id: target_id.to_string(),
                    review_kind: kind.to_string(),
                    severity: 1.0 - row.get(2).and_then(value_as_f32).unwrap_or(0.5),
                    confidence: row.get(2).and_then(value_as_f32).unwrap_or(0.5),
                    t_ms: row.get(3).and_then(|v| v.as_u64()).unwrap_or(index as u64),
                    reason: row
                        .get(4)
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    evidence_ids: Vec::new(),
                    state: "open".to_string(),
                }
            })
            .collect())
    }
}

fn neo4j_intelligence_upsert_statements() -> &'static [&'static str] {
    &[
        r#"
MERGE (doc:GraphIntelligenceWrite {id: $document.id})
SET doc.t_ms = $document.t_ms,
    doc.frame_id = $document.frame_id,
    doc.provenance = $document.provenance,
    doc.confidence = $document.confidence,
    doc.reason = $document.reason,
    doc.source_frame_ids = $document.source_frame_ids
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $features AS feature
MERGE (f:Feature {id: feature.id})
SET f.feature_type = feature.feature_type,
    f.modality = feature.modality,
    f.created_at_ms = feature.created_at_ms,
    f.confidence = feature.confidence,
    f.provenance_json = feature.provenance_json,
    f.source_frame = feature.source_frame,
    f.source_sensor = feature.source_sensor,
    f.vector_refs_json = feature.vector_refs_json,
    f.metadata_json = feature.metadata_json,
    f.current_state = feature.current_state,
    f.reason = feature.reason
MERGE (doc)-[:ASSERTS]->(f)
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $clusters AS cluster
MERGE (c:Cluster {id: cluster.id})
SET c.modality = cluster.modality,
    c.kind = cluster.kind,
    c.first_seen_ms = cluster.first_seen_ms,
    c.last_seen_ms = cluster.last_seen_ms,
    c.confidence = cluster.confidence,
    c.evidence_count = cluster.evidence_count,
    c.source_frame_id = cluster.source_frame_id,
    c.current_state = cluster.current_state,
    c.reason = cluster.reason,
    c.metadata_json = cluster.metadata_json
MERGE (doc)-[:ASSERTS]->(c)
"#,
        r#"
UNWIND $cluster_features AS rel
MATCH (f:Feature {id: rel.feature_id})
MATCH (c:Cluster {id: rel.cluster_id})
MERGE (f)-[r:MEMBER_OF]->(c)
SET r.confidence = rel.confidence,
    r.t_ms = rel.t_ms,
    r.provenance = rel.provenance,
    r.source_frame_ids = rel.source_frame_ids
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $binding_candidates AS candidate
MERGE (bc:BindingCandidate {id: candidate.id})
SET bc.left_cluster_id = candidate.left_cluster_id,
    bc.right_cluster_id = candidate.right_cluster_id,
    bc.relation = candidate.relation,
    bc.confidence = candidate.confidence,
    bc.evidence_count = candidate.evidence_count,
    bc.decision = candidate.decision,
    bc.current_state = candidate.current_state,
    bc.reason = candidate.reason,
    bc.t_ms = candidate.t_ms,
    bc.provenance = candidate.provenance,
    bc.source_frame_ids = candidate.source_frame_ids
MERGE (doc)-[:ASSERTS]->(bc)
WITH candidate, bc
OPTIONAL MATCH (left:Cluster {id: candidate.left_cluster_id})
OPTIONAL MATCH (right:Cluster {id: candidate.right_cluster_id})
FOREACH (_ IN CASE WHEN left IS NULL THEN [] ELSE [1] END | MERGE (bc)-[:FROM]->(left))
FOREACH (_ IN CASE WHEN right IS NULL THEN [] ELSE [1] END | MERGE (bc)-[:TO]->(right))
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $binding_edges AS edge
MERGE (be:BindingEdge {id: edge.id})
SET be.left_cluster_id = edge.left_cluster_id,
    be.right_cluster_id = edge.right_cluster_id,
    be.relation = edge.relation,
    be.confidence = edge.confidence,
    be.evidence_count = edge.evidence_count,
    be.last_seen_ms = edge.last_seen_ms,
    be.current_state = edge.current_state,
    be.reason = edge.reason
MERGE (doc)-[:ASSERTS]->(be)
WITH edge, be
OPTIONAL MATCH (left:Cluster {id: edge.left_cluster_id})
OPTIONAL MATCH (right:Cluster {id: edge.right_cluster_id})
FOREACH (_ IN CASE WHEN left IS NULL OR right IS NULL THEN [] ELSE [1] END |
    MERGE (left)-[r:BOUND_TO]->(right)
    SET r.binding_id = edge.id,
        r.relation = edge.relation,
        r.confidence = edge.confidence,
        r.evidence_count = edge.evidence_count,
        r.last_seen_ms = edge.last_seen_ms)
"#,
        r#"
UNWIND $candidate_edges AS rel
MATCH (bc:BindingCandidate {id: rel.candidate_id})
MATCH (be:BindingEdge {id: rel.binding_id})
MERGE (bc)-[r:PROPOSES]->(be)
SET r.confidence = rel.confidence,
    r.reason = rel.reason,
    r.t_ms = rel.t_ms
"#,
        r#"
UNWIND $candidate_evidence AS evidence
MERGE (e:Evidence {id: evidence.id})
SET e.kind = evidence.kind,
    e.score = evidence.score,
    e.reason = evidence.reason,
    e.t_ms = evidence.t_ms,
    e.current_state = evidence.current_state
WITH evidence, e
MATCH (bc:BindingCandidate {id: evidence.candidate_id})
FOREACH (_ IN CASE WHEN evidence.contradictory THEN [1] ELSE [] END | MERGE (bc)-[:REJECTED_BECAUSE]->(e))
FOREACH (_ IN CASE WHEN evidence.contradictory THEN [] ELSE [1] END | MERGE (bc)-[:SUPPORTED_BY]->(e))
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $tracking_hypotheses AS hyp
MERGE (h:TrackingHypothesis {id: hyp.id})
SET h.family_id = hyp.family_id,
    h.kind = hyp.kind,
    h.target_id = hyp.target_id,
    h.confidence = hyp.confidence,
    h.evidence_count = hyp.evidence_count,
    h.current_state = hyp.current_state,
    h.first_seen_ms = hyp.first_seen_ms,
    h.last_updated_ms = hyp.last_updated_ms,
    h.contradictions = hyp.contradictions,
    h.reason = hyp.reason
MERGE (doc)-[:ASSERTS]->(h)
"#,
        r#"
UNWIND $hypothesis_candidates AS rel
MATCH (h:TrackingHypothesis {id: rel.hypothesis_id})
MATCH (bc:BindingCandidate {id: rel.candidate_id})
MERGE (h)-[r:SUPPORTS]->(bc)
SET r.confidence = rel.confidence,
    r.t_ms = rel.t_ms
"#,
        r#"
UNWIND $hypothesis_competitions AS rel
MATCH (left:TrackingHypothesis {id: rel.left_id})
MATCH (right:TrackingHypothesis {id: rel.right_id})
MERGE (left)-[r:COMPETES_WITH]->(right)
SET r.family_id = rel.family_id,
    r.confidence = rel.confidence,
    r.t_ms = rel.t_ms
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $constellations AS constellation
MERGE (co:Constellation {id: constellation.id})
SET co.kind_hint = constellation.kind_hint,
    co.member_cluster_ids = constellation.member_cluster_ids,
    co.member_binding_ids = constellation.member_binding_ids,
    co.confidence = constellation.confidence,
    co.stability = constellation.stability,
    co.prediction_value = constellation.prediction_value,
    co.first_seen_ms = constellation.first_seen_ms,
    co.last_seen_ms = constellation.last_seen_ms,
    co.evidence_count = constellation.evidence_count,
    co.current_state = constellation.current_state,
    co.reason = constellation.reason,
    co.notes = constellation.notes
MERGE (doc)-[:ASSERTS]->(co)
"#,
        r#"
UNWIND $constellation_members AS rel
MATCH (co:Constellation {id: rel.constellation_id})
OPTIONAL MATCH (c:Cluster {id: rel.member_id})
OPTIONAL MATCH (be:BindingEdge {id: rel.member_id})
FOREACH (_ IN CASE WHEN c IS NULL THEN [] ELSE [1] END | MERGE (co)-[r:HAS_MEMBER]->(c) SET r.confidence = rel.confidence, r.t_ms = rel.t_ms)
FOREACH (_ IN CASE WHEN be IS NULL THEN [] ELSE [1] END | MERGE (co)-[r:SUPPORTED_BY]->(be) SET r.confidence = rel.confidence, r.t_ms = rel.t_ms)
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $associations AS assoc
MERGE (a:Association {id: assoc.id})
SET a.from_id = assoc.from_id,
    a.to_id = assoc.to_id,
    a.relation = assoc.relation,
    a.confidence = assoc.confidence,
    a.evidence_count = assoc.evidence_count,
    a.prediction_gain = assoc.prediction_gain,
    a.contradiction_count = assoc.contradiction_count,
    a.first_seen_ms = assoc.first_seen_ms,
    a.last_seen_ms = assoc.last_seen_ms,
    a.current_state = assoc.current_state,
    a.reason = assoc.reason,
    a.examples_json = assoc.examples_json
MERGE (doc)-[:ASSERTS]->(a)
WITH assoc, a
OPTIONAL MATCH (from {id: assoc.from_id})
OPTIONAL MATCH (to {id: assoc.to_id})
FOREACH (_ IN CASE WHEN from IS NULL THEN [] ELSE [1] END | MERGE (a)-[:FROM]->(from))
FOREACH (_ IN CASE WHEN to IS NULL THEN [] ELSE [1] END | MERGE (a)-[:TO]->(to))
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $action_intents AS action
MERGE (ai:ActionIntent {id: action.id})
SET ai.action_json = action.action_json,
    ai.frame_id = action.frame_id,
    ai.t_ms = action.t_ms,
    ai.confidence = action.confidence,
    ai.current_state = action.current_state,
    ai.reason = action.reason
MERGE (doc)-[:ASSERTS]->(ai)
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $outcomes AS outcome
MERGE (o:Outcome {id: outcome.id})
SET o.frame_id = outcome.frame_id,
    o.t_ms = outcome.t_ms,
    o.reward = outcome.reward,
    o.success = outcome.success,
    o.confidence = outcome.confidence,
    o.current_state = outcome.current_state,
    o.reason = outcome.reason
MERGE (doc)-[:ASSERTS]->(o)
"#,
        r#"
UNWIND $action_outcomes AS rel
MATCH (ai:ActionIntent {id: rel.action_id})
MATCH (o:Outcome {id: rel.outcome_id})
MERGE (ai)-[r:RESULTED_IN]->(o)
SET r.confidence = rel.confidence,
    r.t_ms = rel.t_ms,
    r.reason = rel.reason
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $predictions AS prediction
MERGE (p:Prediction {id: prediction.id})
SET p.target_id = prediction.target_id,
    p.predicted = prediction.predicted,
    p.confidence = prediction.confidence,
    p.t_ms = prediction.t_ms,
    p.current_state = prediction.current_state,
    p.reason = prediction.reason
MERGE (doc)-[:ASSERTS]->(p)
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $surprises AS surprise
MERGE (s:Surprise {id: surprise.id})
SET s.target_id = surprise.target_id,
    s.observed = surprise.observed,
    s.surprise = surprise.surprise,
    s.confidence = surprise.confidence,
    s.t_ms = surprise.t_ms,
    s.reason = surprise.reason
MERGE (doc)-[:ASSERTS]->(s)
"#,
        r#"
UNWIND $prediction_failures AS rel
MATCH (p:Prediction {id: rel.prediction_id})
MATCH (s:Surprise {id: rel.surprise_id})
MERGE (p)-[r:FAILED_WITH]->(s)
SET r.confidence = rel.confidence,
    r.t_ms = rel.t_ms,
    r.reason = rel.reason
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $llm_reviews AS review
MERGE (lr:LlmReview {id: review.id})
SET lr.target_id = review.target_id,
    lr.target_kind = review.target_kind,
    lr.confidence = review.confidence,
    lr.t_ms = review.t_ms,
    lr.critique = review.critique,
    lr.contradictions = review.contradictions,
    lr.suggested_questions = review.suggested_questions,
    lr.current_state = review.current_state
MERGE (doc)-[:ASSERTS]->(lr)
WITH review, lr
OPTIONAL MATCH (target {id: review.target_id})
FOREACH (_ IN CASE WHEN target IS NULL THEN [] ELSE [1] END | MERGE (lr)-[:CRITIQUES]->(target))
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $human_reviews AS review
MERGE (hr:HumanReview {id: review.id})
SET hr.target_id = review.target_id,
    hr.target_kind = review.target_kind,
    hr.confidence = review.confidence,
    hr.t_ms = review.t_ms,
    hr.confirmation = review.confirmation,
    hr.reviewer = review.reviewer,
    hr.current_state = review.current_state
MERGE (doc)-[:ASSERTS]->(hr)
WITH review, hr
OPTIONAL MATCH (target {id: review.target_id})
FOREACH (_ IN CASE WHEN target IS NULL THEN [] ELSE [1] END | MERGE (hr)-[:CONFIRMS]->(target))
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $review_records AS review
MERGE (gr:GraphReview {id: review.id})
SET gr.target_id = review.target_id,
    gr.review_kind = review.review_kind,
    gr.severity = review.severity,
    gr.confidence = review.confidence,
    gr.t_ms = review.t_ms,
    gr.reason = review.reason,
    gr.evidence_ids = review.evidence_ids,
    gr.current_state = review.current_state
MERGE (doc)-[:ASSERTS]->(gr)
WITH review, gr
OPTIONAL MATCH (target {id: review.target_id})
FOREACH (_ IN CASE WHEN target IS NULL THEN [] ELSE [1] END | MERGE (gr)-[:REVIEWS]->(target))
"#,
    ]
}

fn neo4j_intelligence_params(document: &GraphIntelligenceDocument) -> serde_json::Value {
    let document_meta = json!({
        "id": document.id,
        "t_ms": document.t_ms,
        "frame_id": document.frame_id,
        "provenance": document.provenance,
        "confidence": document.confidence,
        "reason": document.reason,
        "source_frame_ids": document.source_frame_ids,
    });
    json!({
        "document": document_meta,
        "features": document.features.iter().map(|feature| json!({
            "id": feature.id.to_string(),
            "feature_type": format!("{:?}", feature.feature_type),
            "modality": feature.modality.as_str(),
            "created_at_ms": feature.created_at_ms,
            "confidence": feature.confidence,
            "provenance_json": json_string(&feature.provenance),
            "source_frame": feature.source_frame,
            "source_sensor": feature.source_sensor,
            "vector_refs_json": json_string(&feature.vector_refs),
            "metadata_json": json_string(&feature.metadata),
            "current_state": "observed",
            "reason": document.reason,
        })).collect::<Vec<_>>(),
        "clusters": document.clusters.iter().map(|cluster| json!({
            "id": cluster.id,
            "modality": cluster.modality.as_str(),
            "kind": format!("{:?}", cluster.kind),
            "first_seen_ms": cluster.first_seen_ms,
            "last_seen_ms": cluster.last_seen_ms,
            "confidence": cluster.confidence,
            "evidence_count": cluster.feature_ids.len() as u32,
            "source_frame_id": cluster.source_frame_id,
            "current_state": "active",
            "reason": document.reason,
            "metadata_json": json_string(&cluster.metadata),
        })).collect::<Vec<_>>(),
        "cluster_features": document.clusters.iter().flat_map(|cluster| {
            cluster.feature_ids.iter().map(|feature_id| json!({
                "cluster_id": cluster.id,
                "feature_id": feature_id.to_string(),
                "confidence": cluster.confidence,
                "t_ms": cluster.last_seen_ms,
                "provenance": document.provenance,
                "source_frame_ids": document.source_frame_ids,
            }))
        }).collect::<Vec<_>>(),
        "binding_candidates": document.binding_candidates.iter().map(|candidate| {
            let id = binding_candidate_id(candidate);
            json!({
                "id": id,
                "left_cluster_id": candidate.left_cluster_id,
                "right_cluster_id": candidate.right_cluster_id,
                "relation": binding_relation_slug(&candidate.relation),
                "confidence": candidate.confidence,
                "evidence_count": candidate.evidence.len() as u32,
                "decision": binding_decision_slug(&candidate.decision),
                "current_state": binding_decision_slug(&candidate.decision),
                "reason": candidate.reason,
                "t_ms": document.t_ms,
                "provenance": document.provenance,
                "source_frame_ids": document.source_frame_ids,
            })
        }).collect::<Vec<_>>(),
        "binding_edges": document.binding_edges.iter().map(|edge| {
            let id = binding_edge_id(edge);
            json!({
                "id": id,
                "left_cluster_id": edge.left_cluster_id,
                "right_cluster_id": edge.right_cluster_id,
                "relation": binding_relation_slug(&edge.relation),
                "confidence": edge.confidence,
                "evidence_count": edge.evidence_count,
                "last_seen_ms": edge.last_seen_ms,
                "current_state": if edge.is_strong() { "accepted" } else { "provisional" },
                "reason": format!("{} evidence events", edge.evidence_count),
            })
        }).collect::<Vec<_>>(),
        "candidate_edges": document.binding_candidates.iter().map(|candidate| json!({
            "candidate_id": binding_candidate_id(candidate),
            "binding_id": binding_edge_id_from_parts(&candidate.left_cluster_id, &candidate.right_cluster_id, &candidate.relation),
            "confidence": candidate.confidence,
            "reason": candidate.reason,
            "t_ms": document.t_ms,
        })).collect::<Vec<_>>(),
        "candidate_evidence": document.binding_candidates.iter().flat_map(|candidate| {
            let candidate_id = binding_candidate_id(candidate);
            candidate.evidence.iter().enumerate().map(move |(index, evidence)| json!({
                "id": format!("evidence:{}:{}", stable_slug(&candidate_id), index),
                "candidate_id": candidate_id,
                "kind": binding_evidence_slug(&evidence.kind),
                "score": evidence.score,
                "reason": evidence.reason,
                "t_ms": document.t_ms,
                "current_state": if binding_evidence_is_contradictory(evidence) { "contradictory" } else { "supporting" },
                "contradictory": binding_evidence_is_contradictory(evidence),
            }))
        }).collect::<Vec<_>>(),
        "tracking_hypotheses": document.tracking_hypotheses.iter().map(|hypothesis| json!({
            "id": hypothesis.id,
            "family_id": hypothesis.family_id,
            "kind": format!("{:?}", hypothesis.kind),
            "target_id": hypothesis.target_id,
            "confidence": hypothesis.confidence,
            "evidence_count": hypothesis.evidence.len() as u32,
            "current_state": format!("{:?}", hypothesis.state).to_lowercase(),
            "first_seen_ms": hypothesis.first_seen_ms,
            "last_updated_ms": hypothesis.last_updated_ms,
            "contradictions": hypothesis.contradictions,
            "reason": hypothesis.contradictions.first().cloned().unwrap_or_else(|| "tracking hypothesis evidence".to_string()),
        })).collect::<Vec<_>>(),
        "hypothesis_candidates": document.tracking_hypotheses.iter().flat_map(|hypothesis| {
            hypothesis.binding_candidate_ids.iter().map(|candidate_id| json!({
                "hypothesis_id": hypothesis.id,
                "candidate_id": candidate_id,
                "confidence": hypothesis.confidence,
                "t_ms": hypothesis.last_updated_ms,
            }))
        }).collect::<Vec<_>>(),
        "hypothesis_competitions": hypothesis_competition_params(&document.tracking_hypotheses),
        "constellations": document.constellations.iter().map(|constellation| json!({
            "id": constellation.id,
            "kind_hint": constellation.kind_hint,
            "member_cluster_ids": constellation.member_cluster_ids,
            "member_binding_ids": constellation.member_binding_ids,
            "confidence": constellation.confidence,
            "stability": constellation.stability,
            "prediction_value": constellation.prediction_value,
            "first_seen_ms": constellation.first_seen_ms,
            "last_seen_ms": constellation.last_seen_ms,
            "evidence_count": constellation.evidence_count,
            "current_state": constellation_state_slug(&constellation.state),
            "reason": constellation.notes.first().cloned().unwrap_or_else(|| "constellation evidence".to_string()),
            "notes": constellation.notes,
        })).collect::<Vec<_>>(),
        "constellation_members": document.constellations.iter().flat_map(|constellation| {
            constellation.member_cluster_ids.iter().chain(constellation.member_binding_ids.iter()).map(|member_id| json!({
                "constellation_id": constellation.id,
                "member_id": member_id,
                "confidence": constellation.confidence,
                "t_ms": constellation.last_seen_ms,
            }))
        }).collect::<Vec<_>>(),
        "associations": document.associations.iter().map(|edge| json!({
            "id": edge.id,
            "from_id": edge.from_id,
            "to_id": edge.to_id,
            "relation": association_relation_slug(&edge.relation),
            "confidence": edge.confidence,
            "evidence_count": edge.evidence_count,
            "prediction_gain": edge.prediction_gain,
            "contradiction_count": edge.contradiction_count,
            "first_seen_ms": edge.first_seen_ms,
            "last_seen_ms": edge.last_seen_ms,
            "current_state": if edge.contradiction_count > 0 { "needs_review" } else { "active" },
            "reason": edge.examples.last().map(|example| example.reason.clone()).unwrap_or_else(|| "association evidence".to_string()),
            "examples_json": json_string(&edge.examples),
        })).collect::<Vec<_>>(),
        "action_intents": document.action_intents.iter().map(|action| json!({
            "id": action.id,
            "action_json": json_string(&action.action),
            "frame_id": action.frame_id,
            "t_ms": action.t_ms,
            "confidence": action.confidence,
            "current_state": action.state,
            "reason": action.reason,
        })).collect::<Vec<_>>(),
        "outcomes": document.outcomes.iter().map(|outcome| json!({
            "id": outcome.id,
            "frame_id": outcome.frame_id,
            "t_ms": outcome.t_ms,
            "reward": outcome.reward,
            "success": outcome.success,
            "confidence": outcome.confidence,
            "current_state": outcome.state,
            "reason": outcome.reason,
        })).collect::<Vec<_>>(),
        "action_outcomes": document.action_intents.iter().zip(document.outcomes.iter()).map(|(action, outcome)| json!({
            "action_id": action.id,
            "outcome_id": outcome.id,
            "confidence": action.confidence.min(outcome.confidence),
            "t_ms": outcome.t_ms,
            "reason": outcome.reason,
        })).collect::<Vec<_>>(),
        "predictions": document.predictions.iter().map(|prediction| json!({
            "id": prediction.id,
            "target_id": prediction.target_id,
            "predicted": prediction.predicted,
            "confidence": prediction.confidence,
            "t_ms": prediction.t_ms,
            "current_state": prediction.state,
            "reason": prediction.reason,
        })).collect::<Vec<_>>(),
        "surprises": document.surprises.iter().map(|surprise| json!({
            "id": surprise.id,
            "target_id": surprise.target_id,
            "observed": surprise.observed,
            "surprise": surprise.surprise,
            "confidence": surprise.confidence,
            "t_ms": surprise.t_ms,
            "reason": surprise.reason,
        })).collect::<Vec<_>>(),
        "prediction_failures": document.predictions.iter().flat_map(|prediction| {
            document.surprises.iter().filter(move |surprise| surprise.target_id == prediction.target_id).map(move |surprise| json!({
                "prediction_id": prediction.id,
                "surprise_id": surprise.id,
                "confidence": prediction.confidence.min(surprise.confidence),
                "t_ms": surprise.t_ms,
                "reason": surprise.reason,
            }))
        }).collect::<Vec<_>>(),
        "llm_reviews": document.llm_reviews.iter().map(|review| json!({
            "id": review.id,
            "target_id": review.target_id,
            "target_kind": format!("{:?}", review.target_kind),
            "confidence": review.confidence,
            "t_ms": review.t_ms,
            "critique": review.critique,
            "contradictions": review.contradictions,
            "suggested_questions": review.suggested_questions,
            "current_state": if review.contradictions.is_empty() { "open" } else { "needs_review" },
        })).collect::<Vec<_>>(),
        "human_reviews": document.human_reviews.iter().map(|review| json!({
            "id": review.id,
            "target_id": review.target_id,
            "target_kind": format!("{:?}", review.target_kind),
            "confidence": review.confidence,
            "t_ms": review.t_ms,
            "confirmation": review.confirmation,
            "reviewer": review.reviewer,
            "current_state": "confirmed",
        })).collect::<Vec<_>>(),
        "review_records": document.review_records.iter().map(|review| json!({
            "id": review.id,
            "target_id": review.target_id,
            "review_kind": review.review_kind,
            "severity": review.severity,
            "confidence": review.confidence,
            "t_ms": review.t_ms,
            "reason": review.reason,
            "evidence_ids": review.evidence_ids,
            "current_state": review.state,
        })).collect::<Vec<_>>(),
    })
}

fn hypothesis_competition_params(hypotheses: &[TrackingHypothesis]) -> Vec<serde_json::Value> {
    let mut by_family = BTreeMap::<String, Vec<&TrackingHypothesis>>::new();
    for hypothesis in hypotheses {
        by_family
            .entry(hypothesis.family_id.clone())
            .or_default()
            .push(hypothesis);
    }
    by_family
        .into_iter()
        .flat_map(|(family_id, hypotheses)| {
            let mut params = Vec::new();
            for left in 0..hypotheses.len() {
                for right in (left + 1)..hypotheses.len() {
                    params.push(json!({
                        "left_id": hypotheses[left].id,
                        "right_id": hypotheses[right].id,
                        "family_id": family_id,
                        "confidence": hypotheses[left].confidence.min(hypotheses[right].confidence),
                        "t_ms": hypotheses[left].last_updated_ms.max(hypotheses[right].last_updated_ms),
                    }));
                }
            }
            params
        })
        .collect()
}

fn json_string(value: &impl Serialize) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".to_string())
}

fn summaries_from_row(value: Option<&serde_json::Value>) -> Vec<GraphFactSummary> {
    value
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .map(summary_from_value)
        .filter(|summary| !summary.id.is_empty())
        .collect()
}

fn summary_from_value(value: &serde_json::Value) -> GraphFactSummary {
    GraphFactSummary {
        id: value
            .get("id")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string(),
        kind: value
            .get("kind")
            .or_else(|| value.get("kind_hint"))
            .or_else(|| value.get("feature_type"))
            .or_else(|| value.get("relation"))
            .and_then(|value| value.as_str())
            .unwrap_or("graph_fact")
            .to_string(),
        relation: value
            .get("relation")
            .and_then(|value| value.as_str())
            .map(str::to_string),
        confidence: value
            .get("confidence")
            .or_else(|| value.get("score"))
            .and_then(value_as_f32)
            .unwrap_or_default(),
        evidence_count: value
            .get("evidence_count")
            .and_then(|value| value.as_u64())
            .unwrap_or_default() as u32,
        t_ms: value
            .get("t_ms")
            .or_else(|| value.get("last_seen_ms"))
            .or_else(|| value.get("last_updated_ms"))
            .and_then(|value| value.as_u64())
            .unwrap_or_default(),
        state: value
            .get("current_state")
            .or_else(|| value.get("decision"))
            .and_then(|value| value.as_str())
            .map(str::to_string),
        reason: value
            .get("reason")
            .or_else(|| value.get("summary"))
            .or_else(|| value.get("critique"))
            .or_else(|| value.get("confirmation"))
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string(),
    }
}

fn value_as_f32(value: &serde_json::Value) -> Option<f32> {
    value.as_f64().map(|value| value as f32)
}

fn graph_recall_query_ids(query: &GraphRecallQuery) -> Vec<String> {
    query
        .active_feature_ids
        .iter()
        .chain(query.active_cluster_ids.iter())
        .chain(query.active_constellation_ids.iter())
        .chain(query.action_ids.iter())
        .chain(query.place_ids.iter())
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn binding_edge_id(edge: &BindingEdge) -> String {
    binding_edge_id_from_parts(
        &edge.left_cluster_id,
        &edge.right_cluster_id,
        &edge.relation,
    )
}

fn binding_edge_id_from_parts(left: &str, right: &str, relation: &BindingRelation) -> String {
    format!(
        "binding-edge:{}:{}:{}",
        stable_slug(left),
        binding_relation_slug(relation),
        stable_slug(right)
    )
}

fn binding_relation_slug(relation: &BindingRelation) -> &'static str {
    match relation {
        BindingRelation::CooccursInTime => "cooccurs_in_time",
        BindingRelation::CooccursInEstimatedSpace => "cooccurs_in_estimated_space",
        BindingRelation::MovesTogether => "moves_together",
        BindingRelation::PredictsSameFutureEvents => "predicts_same_future_events",
        BindingRelation::NamedBy => "named_by",
        BindingRelation::ProjectsTo => "projects_to",
        BindingRelation::HasColorAtPose => "has_color_at_pose",
        BindingRelation::LikelySameEntity => "likely_same_entity",
        BindingRelation::ExplainsOutcome => "explains_outcome",
        BindingRelation::Contradicts => "contradicts",
        BindingRelation::RequiresReview => "requires_review",
    }
}

fn binding_decision_slug(decision: &BindingDecision) -> &'static str {
    match decision {
        BindingDecision::Accept => "accept",
        BindingDecision::Reject => "reject",
        BindingDecision::HoldAmbiguous => "hold_ambiguous",
        BindingDecision::AskHuman => "ask_human",
        BindingDecision::CollectMoreEvidence => "collect_more_evidence",
    }
}

fn binding_evidence_slug(kind: &BindingEvidenceKind) -> &'static str {
    match kind {
        BindingEvidenceKind::TemporalOverlap => "temporal_overlap",
        BindingEvidenceKind::SpatialOverlap => "spatial_overlap",
        BindingEvidenceKind::VectorSimilarity => "vector_similarity",
        BindingEvidenceKind::ProjectionAgreement => "projection_agreement",
        BindingEvidenceKind::PoseAgreement => "pose_agreement",
        BindingEvidenceKind::RepeatedCooccurrence => "repeated_cooccurrence",
        BindingEvidenceKind::SingleCandidateContext => "single_candidate_context",
        BindingEvidenceKind::HumanConfirmed => "human_confirmed",
        BindingEvidenceKind::LlmSuggested => "llm_suggested",
        BindingEvidenceKind::Contradiction => "contradiction",
        BindingEvidenceKind::SimultaneousConflict => "simultaneous_conflict",
    }
}

fn binding_evidence_is_contradictory(evidence: &BindingEvidence) -> bool {
    matches!(
        evidence.kind,
        BindingEvidenceKind::Contradiction | BindingEvidenceKind::SimultaneousConflict
    )
}

fn constellation_state_slug(state: &ConstellationState) -> &'static str {
    match state {
        ConstellationState::Candidate => "candidate",
        ConstellationState::Stable => "stable",
        ConstellationState::Ambiguous => "ambiguous",
        ConstellationState::SplitNeeded => "split_needed",
        ConstellationState::MergeNeeded => "merge_needed",
        ConstellationState::Retired => "retired",
    }
}

fn neo4j_entity_params(record: &MemoryRecord) -> Vec<serde_json::Value> {
    record
        .graph_entities
        .iter()
        .map(|entity| {
            json!({
                "id": entity.id,
                "labels": entity.labels,
                "summary": entity.summary,
                "score": entity.score,
                "frame_id": record.frame_id.to_string(),
                "t_ms": record.t_ms,
            })
        })
        .collect()
}

fn neo4j_relationship_params(record: &MemoryRecord) -> Vec<serde_json::Value> {
    record
        .graph_relationships
        .iter()
        .map(|edge| {
            json!({
                "edge_id": graph_edge_id(edge),
                "from": edge.from,
                "to": edge.to,
                "kind": edge.relationship,
                "summary": edge.summary,
                "score": edge.score,
                "payload_json": edge.payload.to_string(),
                "frame_id": record.frame_id.to_string(),
                "t_ms": record.t_ms,
            })
        })
        .collect()
}

