impl<L, M, R, C, S, A> MinimalRuntime<L, M, R, C, S, A>
where
    L: LedgerWriter + Sync,
    M: MemoryStore,
    R: Recall + Sync,
    C: Conductor,
    S: SafetyLayer,
    A: LlmAgent + 'static,
{
    fn run_event_scripts(
        &mut self,
        now: &mut Now,
        notes: &mut Vec<String>,
        proposed_actions: &mut Vec<ActionPrimitive>,
    ) -> Result<(Option<ActionPrimitive>, Vec<ErasedBehaviorRunRecord>)> {
        let mut behavior_runs = Vec::new();
        let forced_action = None;
        let mut safe_sequences = serde_json::Map::new();

        if let Some(input) = robot_initialized_event_input(now) {
            let run = self
                .models
                .behaviors
                .event_robot_initialized
                .infer_with_teacher_source(&input, now.t_ms, TrainingSource::HardcodedTeacher)?;
            let sequence = safety_trace_script_actions(&mut self.safety, now, &run.chosen);
            for action in run
                .chosen
                .actions
                .iter()
                .filter_map(script_action_to_primitive)
            {
                proposed_actions.push(action);
            }
            safe_sequences.insert(
                "robot-initialized".to_string(),
                serde_json::to_value(&sequence)?,
            );
            notes.push("EventScript:on(robot-initialized) emitted bring-up sequence".to_string());
            behavior_runs.push(run.record.erase());
        }

        if now.body.flags.bump_left || now.body.flags.bump_right {
            let input = BumpEventInput {
                t_ms: now.t_ms,
                bump_left: now.body.flags.bump_left,
                bump_right: now.body.flags.bump_right,
            };
            let run = self.models.behaviors.event_bump.infer_with_teacher_source(
                &input,
                now.t_ms,
                TrainingSource::HardcodedTeacher,
            )?;
            let sequence = safety_trace_script_actions(&mut self.safety, now, &run.chosen);
            if let Some(first) = first_motor_script_action(&run.chosen) {
                proposed_actions.push(first);
            }
            safe_sequences.insert("bump".to_string(), serde_json::to_value(&sequence)?);
            notes.push(
                "EventScript:on(bump) emitted random lament -> Stop -> Rotate(180) -> Go"
                    .to_string(),
            );
            let mut record = run.record;
            record.selected_output = Some(EventScriptOutput {
                actions: sequence
                    .actions
                    .iter()
                    .map(|action| action.requested.clone())
                    .collect(),
            });
            record.confidence = Some(if sequence.actions.iter().any(|action| action.vetoed) {
                0.5
            } else {
                1.0
            });
            behavior_runs.push(record.erase());
        }

        if !safe_sequences.is_empty() {
            now.extensions.insert(
                "event_scripts".to_string(),
                serde_json::Value::Object(safe_sequences),
            );
            notes.push("EventScript: safety filtered every emitted action".to_string());
        }

        Ok((forced_action, behavior_runs))
    }

}
