impl<L, M, R, C, S, A> MinimalRuntime<L, M, R, C, S, A>
where
    L: LedgerWriter + Sync,
    M: MemoryStore,
    R: Recall + Sync,
    C: Conductor,
    S: SafetyLayer,
    A: LlmAgent + 'static,
{
    /// Poll a previous request and enqueue the current immutable view.
    ///
    /// `JoinHandle::is_finished` is deliberately checked before awaiting it,
    /// making the only await here a ready-value extraction rather than model
    /// or network I/O. Provider output is reduced to `LlmSense` and typed
    /// evidence. Decisions cross only as discarded advisory telemetry;
    /// conscious commands and executable actions never cross this boundary.
    async fn advance_cognition(
        &mut self,
        now: &Now,
        impressions: &[Impression],
        embodied: &pete_experience::EmbodiedContext,
        latent: &ExperienceLatent,
        futures: &[FuturePrediction],
        recall_summary: &str,
        notes: &mut Vec<String>,
        brain_events: &mut Vec<BrainEvent>,
        frame_id: Uuid,
    ) -> Option<AcceptedLlmCognition> {
        if now.t_ms > self.cognition.last_sense_valid_until_ms {
            self.cognition.last_sense = pete_now::LlmSense::default();
        }
        let mut accepted = None;
        if self
            .cognition
            .pending
            .as_ref()
            .is_some_and(|pending| now.t_ms > pending.deadline_ms)
        {
            let pending = self.cognition.pending.take().expect("expired task");
            pending.task.abort();
            self.cognition.next_request_at_ms = now.t_ms.saturating_add(COGNITION_COOLDOWN_MS);
            self.cognition.last_outcome = Some(CognitionOutcome::Expired);
            brain_events.push(higher_brain_response_event(
                &pending.request_event_id,
                &pending.snapshot_ref,
                pending.requested_at_ms,
                now.t_ms,
                EventDisposition::Expired,
                serde_json::json!({"reason": "deadline expired"}),
            ));
        }
        if self
            .cognition
            .pending
            .as_ref()
            .is_some_and(|pending| pending.task.is_finished())
        {
            let pending = self.cognition.pending.take().expect("finished task");
            self.cognition.next_request_at_ms = now.t_ms.saturating_add(COGNITION_COOLDOWN_MS);
            let PendingLlmCognition {
                request_event_id,
                snapshot_ref,
                requested_at_ms,
                deadline_ms,
                task,
            } = pending;
            match task.await {
                Err(error) => {
                    let outcome = if error.is_cancelled() {
                        CognitionOutcome::Cancelled
                    } else {
                        CognitionOutcome::Failed(error.to_string())
                    };
                    self.cognition.last_outcome = Some(outcome);
                    brain_events.push(higher_brain_response_event(
                        &request_event_id,
                        &snapshot_ref,
                        requested_at_ms,
                        now.t_ms,
                        if error.is_cancelled() {
                            EventDisposition::Rejected
                        } else {
                            EventDisposition::Unavailable
                        },
                        serde_json::json!({"error": error.to_string()}),
                    ));
                }
                Ok(Err(error)) => {
                    self.cognition.last_outcome = Some(CognitionOutcome::Failed(error.to_string()));
                    brain_events.push(higher_brain_response_event(
                        &request_event_id,
                        &snapshot_ref,
                        requested_at_ms,
                        now.t_ms,
                        EventDisposition::Unavailable,
                        serde_json::json!({"error": error.to_string()}),
                    ));
                }
                Ok(Ok((_reflection, _result))) if now.t_ms > deadline_ms => {
                    self.cognition.last_outcome = Some(CognitionOutcome::Expired);
                    brain_events.push(higher_brain_response_event(
                        &request_event_id,
                        &snapshot_ref,
                        requested_at_ms,
                        now.t_ms,
                        EventDisposition::Expired,
                        serde_json::json!({"reason": "response arrived after deadline"}),
                    ));
                }
                Ok(Ok((reflection, result))) => {
                    self.cognition.last_sense = result.sense.clone();
                    self.cognition.last_sense_valid_until_ms =
                        now.t_ms.saturating_add(COGNITION_DEADLINE_MS);
                    self.cognition.last_outcome = Some(CognitionOutcome::Accepted);
                    let response_event = higher_brain_response_event(
                        &request_event_id,
                        &snapshot_ref,
                        requested_at_ms,
                        now.t_ms,
                        EventDisposition::Accepted,
                        serde_json::json!({
                            "sense": result.sense,
                            "decision": result.decision,
                            "conscious_command": result.conscious_command,
                            "combobulation": reflection,
                        }),
                    );
                    brain_events.push(response_event);
                    accepted = Some(AcceptedLlmCognition {
                        reflection,
                        tick: result,
                        snapshot_ref,
                        requested_at_ms,
                        observed_at_ms: now.t_ms,
                    });
                }
            }
        }

        if self.cognition.provider_declared_available
            && self.cognition.pending.is_none()
            && now.t_ms >= self.cognition.next_request_at_ms
        {
            let llm = Arc::clone(&self.llm);
            let request_now = now.clone();
            let request_impressions = impressions.to_vec();
            let request_embodied = embodied.clone();
            let request_latent = latent.clone();
            let request_futures = futures.to_vec();
            let request_recall = recall_summary.to_string();
            let task = tokio::spawn(async move {
                let mut agent = llm.lock().await;
                let reflection = agent
                    .combobulate(
                        &request_now,
                        &request_impressions,
                        Some(&request_embodied),
                        &request_latent,
                        &request_futures,
                        &request_recall,
                    )
                    .await?;
                let awareness = reflection.as_ref().map(|value| value.summary.as_str());
                let tick = agent
                    .maybe_tick(
                        &request_now,
                        Some(&request_embodied),
                        &request_latent,
                        &request_futures,
                        &request_recall,
                        awareness,
                    )
                    .await?;
                Ok((reflection, tick))
            });
            let snapshot_ref = frame_id.to_string();
            let request_event = higher_brain_request_event(frame_id, now.t_ms);
            let request_event_id = request_event.event_id.clone();
            brain_events.push(request_event);
            self.cognition.pending = Some(PendingLlmCognition {
                request_event_id,
                snapshot_ref,
                requested_at_ms: now.t_ms,
                deadline_ms: now.t_ms.saturating_add(COGNITION_DEADLINE_MS),
                task,
            });
        } else if !self.cognition.provider_declared_available && self.cognition.pending.is_none() {
            brain_events.push(higher_brain_unavailable_event(
                frame_id,
                now.t_ms,
                self.cognition.provider_unavailable_reason.as_deref(),
            ));
        }
        if let Some(outcome) = self.cognition.last_outcome.as_ref() {
            notes.push(match outcome {
                CognitionOutcome::Accepted => "LlmProviderOutcome: accepted".to_string(),
                CognitionOutcome::Expired => "LlmProviderOutcome: expired".to_string(),
                CognitionOutcome::Cancelled => "LlmProviderOutcome: cancelled".to_string(),
                CognitionOutcome::Failed(error) => format!("LlmProviderOutcome: failed: {error}"),
            });
        }
        accepted
    }

    pub fn apply_behavior_node_update(&mut self, node_id: &str, update: &BehaviorNodeUpdate) {
        self.models.apply_behavior_node_update(node_id, update);
    }

}
