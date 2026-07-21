#[derive(Clone, Debug, Default)]
pub struct ReignQueue {
    pending: VecDeque<ReignInput>,
    latest: Option<ReignInput>,
    clear_sequence: u64,
}

impl ReignQueue {
    pub fn push(&mut self, input: ReignInput) {
        self.latest = Some(input.clone());
        self.pending.push_back(input);
    }

    pub fn latest_active(&self, now_ms: TimeMs) -> Option<ReignInput> {
        self.pending
            .iter()
            .rev()
            .find(|input| input.expires_at_ms > now_ms)
            .cloned()
    }

    pub fn drain_expired(&mut self, now_ms: TimeMs) {
        self.pending.retain(|input| input.expires_at_ms > now_ms);
        if self
            .latest
            .as_ref()
            .map(|input| input.expires_at_ms <= now_ms)
            .unwrap_or(false)
        {
            self.latest = self.latest_active(now_ms);
        }
    }

    pub fn clear(&mut self) {
        self.pending.clear();
        self.latest = None;
        self.clear_sequence = self.clear_sequence.saturating_add(1);
    }

    pub fn sense(&self, now_ms: TimeMs) -> ReignSense {
        let latest = self.latest_active(now_ms);
        let active = latest.is_some();
        ReignSense {
            active,
            mode: latest.as_ref().map(|input| input.mode.clone()),
            last_command_age_ms: latest
                .as_ref()
                .map(|input| now_ms.saturating_sub(input.issued_at_ms)),
            human_override_pressure: latest
                .as_ref()
                .map(|input| input.priority.clamp(0.0, 1.0))
                .unwrap_or(0.0),
            latest,
            pending_count: self
                .pending
                .iter()
                .filter(|input| input.expires_at_ms > now_ms)
                .count(),
            clear_sequence: self.clear_sequence,
        }
    }
}

fn mechanical_reign_action(
    input: &Option<ReignInput>,
    selector_mode: ActionSelectorMode,
) -> Option<ActionPrimitive> {
    let input = input.as_ref()?;
    let goal_mode = selector_mode == ActionSelectorMode::Goal;
    let mechanical = matches!(input.mode, pete_actions::ReignMode::Direct)
        || (!goal_mode && matches!(input.mode, pete_actions::ReignMode::Assist));
    if !mechanical {
        return None;
    }
    input.command.to_action()
}

fn reign_input_drives_sim_directly(input: &ReignInput) -> bool {
    matches!(
        input.mode,
        pete_actions::ReignMode::Direct | pete_actions::ReignMode::Assist
    ) && input.command.to_action().is_some()
}
