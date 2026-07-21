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
