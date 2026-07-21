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
