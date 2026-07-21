fn scenario_map_memory_decision_report(
    decision: &serde_json::Value,
) -> Option<ScenarioMapMemoryDecisionReport> {
    let signal = decision.get("signal")?.as_str()?.to_string();
    let reason = decision.get("reason")?.as_str()?.to_string();
    let chosen_action = decision
        .get("chosen_action")
        .or_else(|| decision.get("selected_action"))
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok());
    Some(ScenarioMapMemoryDecisionReport {
        signal,
        signal_value: decision
            .get("signal_value")
            .and_then(|value| value.as_f64())
            .map(|value| value as f32),
        signal_confidence: decision
            .get("signal_confidence")
            .or_else(|| decision.get("confidence"))
            .and_then(|value| value.as_f64())
            .unwrap_or(0.0) as f32,
        chosen_action,
        reason,
        reason_string: decision
            .get("reason_string")
            .and_then(|value| value.as_str())
            .map(str::to_string),
        safety_overrode: decision
            .get("safety_overrode")
            .and_then(|value| value.as_bool())
            .unwrap_or(false),
    })
}

#[derive(Clone, Debug, Default)]
struct ScenarioEpisodeMemoryBuilder {
    max_places_visited: usize,
    charge_memory_ticks: usize,
    charge_opportunity_ticks: usize,
    danger_memory_ticks: usize,
    danger_opportunity_ticks: usize,
    social_memory_ticks: usize,
    social_opportunity_ticks: usize,
    first_novelty: Option<f32>,
    final_novelty: Option<f32>,
}

impl ScenarioEpisodeMemoryBuilder {
    fn observe(&mut self, snapshot: &WorldSnapshot, tick: &RuntimeTick) {
        let memory = &tick.frame.now.memory;
        self.max_places_visited = self.max_places_visited.max(memory.places_visited as usize);
        self.first_novelty.get_or_insert(memory.place_novelty);
        self.final_novelty = Some(memory.place_novelty);

        let charger_near = snapshot.body.charging
            || sim_world_score(snapshot, 3).max(sim_world_score(snapshot, 4)) >= 0.3;
        if charger_near {
            self.charge_opportunity_ticks = self.charge_opportunity_ticks.saturating_add(1);
            if memory.place_charge_value >= 0.3 {
                self.charge_memory_ticks = self.charge_memory_ticks.saturating_add(1);
            }
        }

        let danger_near = snapshot.body.flags.bump_left
            || snapshot.body.flags.bump_right
            || snapshot.body.flags.wall
            || snapshot.body.flags.cliff_left
            || snapshot.body.flags.cliff_front_left
            || snapshot.body.flags.cliff_front_right
            || snapshot.body.flags.cliff_right
            || snapshot
                .range
                .nearest_m
                .map(|nearest| nearest <= 0.35)
                .unwrap_or(false);
        if danger_near {
            self.danger_opportunity_ticks = self.danger_opportunity_ticks.saturating_add(1);
            if memory.place_danger >= 0.3 {
                self.danger_memory_ticks = self.danger_memory_ticks.saturating_add(1);
            }
        }

        let social_seen = !snapshot.face.vectors.is_empty()
            || !snapshot.voice.vectors.is_empty()
            || !snapshot.kinect.skeletons.is_empty();
        if social_seen {
            self.social_opportunity_ticks = self.social_opportunity_ticks.saturating_add(1);
            if memory.place_social_value >= 0.3 {
                self.social_memory_ticks = self.social_memory_ticks.saturating_add(1);
            }
        }
    }

    fn finish(self) -> ScenarioEpisodeMemoryReport {
        ScenarioEpisodeMemoryReport {
            places_visited: self.max_places_visited,
            charge_memory_ticks: self.charge_memory_ticks,
            charge_opportunity_ticks: self.charge_opportunity_ticks,
            charge_memory_hit_rate: hit_rate(
                self.charge_memory_ticks,
                self.charge_opportunity_ticks,
            ),
            danger_memory_ticks: self.danger_memory_ticks,
            danger_opportunity_ticks: self.danger_opportunity_ticks,
            danger_memory_hit_rate: hit_rate(
                self.danger_memory_ticks,
                self.danger_opportunity_ticks,
            ),
            social_memory_ticks: self.social_memory_ticks,
            social_opportunity_ticks: self.social_opportunity_ticks,
            social_memory_hit_rate: hit_rate(
                self.social_memory_ticks,
                self.social_opportunity_ticks,
            ),
            first_novelty: self.first_novelty,
            final_novelty: self.final_novelty,
            novelty_decayed: self
                .first_novelty
                .zip(self.final_novelty)
                .map(|(first, final_value)| final_value <= first)
                .unwrap_or(false),
        }
    }
}
