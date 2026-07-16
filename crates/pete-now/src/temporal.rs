use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{EntityId, EvidenceRef};

const RECENT_EPISODE_LIMIT: usize = 8;

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum ClockDomain {
    #[default]
    Monotonic,
    WallClock,
    Event,
    Observation,
    Recall,
    Replay,
    Predicted,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypedTimestamp {
    pub domain: ClockDomain,
    pub ms: u64,
}

impl TypedTimestamp {
    pub const fn monotonic(ms: u64) -> Self {
        Self {
            domain: ClockDomain::Monotonic,
            ms,
        }
    }

    pub fn elapsed_since(self, earlier: Self) -> Option<u64> {
        (self.domain == earlier.domain).then(|| self.ms.saturating_sub(earlier.ms))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeInterval {
    pub domain: ClockDomain,
    pub start_ms: u64,
    pub end_ms: Option<u64>,
    #[serde(default)]
    pub uncertainty_ms: u64,
}

impl TimeInterval {
    pub fn open(domain: ClockDomain, start_ms: u64) -> Self {
        Self {
            domain,
            start_ms,
            end_ms: None,
            uncertainty_ms: 0,
        }
    }

    pub fn contains(&self, timestamp: TypedTimestamp) -> Option<bool> {
        (timestamp.domain == self.domain).then(|| {
            timestamp.ms >= self.start_ms && self.end_ms.is_none_or(|end_ms| timestamp.ms <= end_ms)
        })
    }

    pub fn duration_ms(&self, now: TypedTimestamp) -> Option<u64> {
        (now.domain == self.domain)
            .then(|| self.end_ms.unwrap_or(now.ms).saturating_sub(self.start_ms))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalRelation {
    #[default]
    OccurredDuring,
    Before,
    After,
    Overlaps,
    Continues,
    Ended,
    ExpectedBy,
    Repeats,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TemporalBelief {
    pub interval: TimeInterval,
    pub relation: TemporalRelation,
    pub subject: String,
    pub confidence: f32,
    #[serde(default)]
    pub provenance: Vec<EvidenceRef>,
}

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum EpisodeKind {
    Charging,
    Conversation,
    Recovery,
    Exploration,
    Task,
    #[default]
    Other,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpisodeClosureReason {
    ExplicitCompletion,
    StateEnded,
    Inactivity,
    GoalChanged,
    ReplayBoundary,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EpisodeId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Episode {
    pub episode_id: EpisodeId,
    pub kind: EpisodeKind,
    pub interval: TimeInterval,
    #[serde(default)]
    pub participants: Vec<EntityId>,
    #[serde(default)]
    pub places: Vec<String>,
    #[serde(default)]
    pub active_goals: Vec<String>,
    #[serde(default)]
    pub significant_events: Vec<String>,
    #[serde(default)]
    pub preceding_episode_refs: Vec<EpisodeId>,
    pub closure_reason: Option<EpisodeClosureReason>,
    pub confidence: f32,
    #[serde(default)]
    pub provenance: Vec<EvidenceRef>,
}

impl Episode {
    pub fn is_open(&self) -> bool {
        self.interval.end_ms.is_none()
    }

    pub fn elapsed_ms(&self, monotonic_now_ms: u64) -> Option<u64> {
        self.interval
            .duration_ms(TypedTimestamp::monotonic(monotonic_now_ms))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PendingTemporalExpectation {
    pub subject: String,
    pub expected_interval: TimeInterval,
    pub confidence: f32,
    #[serde(default)]
    pub provenance: Vec<EvidenceRef>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TemporalContext {
    pub monotonic_now: TypedTimestamp,
    pub wall_clock_now: Option<TypedTimestamp>,
    pub replay_now: Option<TypedTimestamp>,
    pub current_episode: Option<EpisodeId>,
    #[serde(default)]
    pub active_episodes: Vec<Episode>,
    #[serde(default)]
    pub recently_completed: Vec<Episode>,
    #[serde(default)]
    pub ongoing_durations_ms: BTreeMap<String, u64>,
    #[serde(default)]
    pub pending_expectations: Vec<PendingTemporalExpectation>,
    #[serde(default)]
    pub current_temporal_beliefs: Vec<TemporalBelief>,
}

impl TemporalContext {
    pub fn active_episode(&self, kind: EpisodeKind) -> Option<&Episode> {
        self.active_episodes
            .iter()
            .find(|episode| episode.kind == kind)
    }

    pub fn elapsed_for(&self, subject: &str) -> Option<u64> {
        self.ongoing_durations_ms.get(subject).copied()
    }

    pub fn last_completed(&self, kind: EpisodeKind) -> Option<&Episode> {
        self.recently_completed
            .iter()
            .rev()
            .find(|episode| episode.kind == kind)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimedPrediction<T> {
    pub value: T,
    pub horizon: TimeInterval,
    pub confidence: f32,
    #[serde(default)]
    pub provenance: Vec<EvidenceRef>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct TemporalIntegrator {
    sequence: u64,
    active: BTreeMap<EpisodeKind, Episode>,
    completed: Vec<Episode>,
    active_goal: Option<(String, u64)>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct TemporalUpdateInput {
    pub monotonic_now_ms: u64,
    pub wall_clock_unix_ms: Option<u64>,
    pub replay_now_ms: Option<u64>,
    pub charging: bool,
    pub contact_or_recovery: bool,
    pub active_goal: Option<String>,
    pub interaction_id: Option<String>,
    pub interaction_participants: Vec<EntityId>,
    pub expectations: Vec<PendingTemporalExpectation>,
    pub temporal_beliefs: Vec<TemporalBelief>,
}

impl TemporalIntegrator {
    pub(crate) fn update(&mut self, input: TemporalUpdateInput) -> TemporalContext {
        match (&self.active_goal, &input.active_goal) {
            (Some((current, _)), Some(next)) if current == next => {}
            (_, Some(next)) => {
                self.active_goal = Some((next.clone(), input.monotonic_now_ms));
            }
            (_, None) => self.active_goal = None,
        }
        self.transition(
            EpisodeKind::Charging,
            input.charging,
            &input,
            Vec::new(),
            "charging",
        );
        self.transition(
            EpisodeKind::Conversation,
            input.interaction_id.is_some(),
            &input,
            input.interaction_participants.clone(),
            "social_interaction",
        );
        self.transition(
            EpisodeKind::Recovery,
            input.contact_or_recovery,
            &input,
            Vec::new(),
            "contact_or_recovery",
        );
        self.transition(
            EpisodeKind::Exploration,
            input.active_goal.as_deref() == Some("explore"),
            &input,
            Vec::new(),
            "goal:explore",
        );
        self.transition(
            EpisodeKind::Task,
            input.active_goal.as_deref() == Some("follow_task"),
            &input,
            Vec::new(),
            "goal:follow_task",
        );

        for episode in self.active.values_mut() {
            if let Some(goal) = input.active_goal.as_ref() {
                if !episode.active_goals.contains(goal) {
                    episode.active_goals.push(goal.clone());
                }
            }
        }

        let active_episodes = self.active.values().cloned().collect::<Vec<_>>();
        let current_episode = [
            EpisodeKind::Conversation,
            EpisodeKind::Recovery,
            EpisodeKind::Charging,
            EpisodeKind::Task,
            EpisodeKind::Exploration,
        ]
        .into_iter()
        .find_map(|kind| {
            self.active
                .get(&kind)
                .map(|episode| episode.episode_id.clone())
        });
        let ongoing_durations_ms = active_episodes
            .iter()
            .filter_map(|episode| {
                episode
                    .elapsed_ms(input.monotonic_now_ms)
                    .map(|elapsed| (format!("episode:{}", episode.episode_id.0), elapsed))
            })
            .chain(self.active_goal.as_ref().map(|(goal, entered_at_ms)| {
                (
                    format!("goal:{goal}"),
                    input.monotonic_now_ms.saturating_sub(*entered_at_ms),
                )
            }))
            .collect();
        let recent_start = self.completed.len().saturating_sub(RECENT_EPISODE_LIMIT);
        TemporalContext {
            monotonic_now: TypedTimestamp::monotonic(input.monotonic_now_ms),
            wall_clock_now: input.wall_clock_unix_ms.map(|ms| TypedTimestamp {
                domain: ClockDomain::WallClock,
                ms,
            }),
            replay_now: input.replay_now_ms.map(|ms| TypedTimestamp {
                domain: ClockDomain::Replay,
                ms,
            }),
            current_episode,
            active_episodes,
            recently_completed: self.completed[recent_start..].to_vec(),
            ongoing_durations_ms,
            pending_expectations: input.expectations,
            current_temporal_beliefs: input.temporal_beliefs,
        }
    }

    fn transition(
        &mut self,
        kind: EpisodeKind,
        active: bool,
        input: &TemporalUpdateInput,
        participants: Vec<EntityId>,
        event: &str,
    ) {
        if active {
            if let Some(episode) = self.active.get_mut(&kind) {
                for participant in participants {
                    if !episode.participants.contains(&participant) {
                        episode.participants.push(participant);
                    }
                }
                return;
            }
            self.sequence = self.sequence.saturating_add(1);
            let preceding_episode_refs = self
                .completed
                .last()
                .map(|episode| vec![episode.episode_id.clone()])
                .unwrap_or_default();
            self.active.insert(
                kind,
                Episode {
                    episode_id: EpisodeId(format!(
                        "episode:{}:{}:{}",
                        episode_kind_key(kind),
                        input.monotonic_now_ms,
                        self.sequence
                    )),
                    kind,
                    interval: TimeInterval::open(ClockDomain::Monotonic, input.monotonic_now_ms),
                    participants,
                    active_goals: input.active_goal.iter().cloned().collect(),
                    significant_events: vec![event.to_string()],
                    preceding_episode_refs,
                    confidence: 1.0,
                    ..Episode::default()
                },
            );
        } else if let Some(mut episode) = self.active.remove(&kind) {
            episode.interval.end_ms = Some(input.monotonic_now_ms);
            episode.closure_reason = Some(EpisodeClosureReason::StateEnded);
            episode
                .significant_events
                .push("episode_closed".to_string());
            self.completed.push(episode);
        }
    }
}

fn episode_kind_key(kind: EpisodeKind) -> &'static str {
    match kind {
        EpisodeKind::Charging => "charging",
        EpisodeKind::Conversation => "conversation",
        EpisodeKind::Recovery => "recovery",
        EpisodeKind::Exploration => "exploration",
        EpisodeKind::Task => "task",
        EpisodeKind::Other => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unlike_clock_domains_are_not_compared() {
        assert_eq!(
            TypedTimestamp {
                domain: ClockDomain::WallClock,
                ms: 20,
            }
            .elapsed_since(TypedTimestamp::monotonic(10)),
            None
        );
    }

    #[test]
    fn charging_and_conversation_episodes_close_deterministically() {
        let mut integrator = TemporalIntegrator::default();
        let charging = integrator.update(TemporalUpdateInput {
            monotonic_now_ms: 100,
            charging: true,
            ..TemporalUpdateInput::default()
        });
        assert!(charging.active_episode(EpisodeKind::Charging).is_some());

        let conversation = integrator.update(TemporalUpdateInput {
            monotonic_now_ms: 200,
            charging: true,
            interaction_id: Some("interaction:1".to_string()),
            interaction_participants: vec![EntityId("person:alex".to_string())],
            ..TemporalUpdateInput::default()
        });
        assert!(conversation
            .active_episode(EpisodeKind::Conversation)
            .is_some());

        let closed = integrator.update(TemporalUpdateInput {
            monotonic_now_ms: 500,
            ..TemporalUpdateInput::default()
        });
        assert_eq!(closed.recently_completed.len(), 2);
        assert!(closed
            .recently_completed
            .iter()
            .all(|episode| episode.interval.end_ms == Some(500)));
    }

    #[test]
    fn active_goal_duration_uses_monotonic_time_when_wall_clock_moves_backward() {
        let mut integrator = TemporalIntegrator::default();
        integrator.update(TemporalUpdateInput {
            monotonic_now_ms: 100,
            wall_clock_unix_ms: Some(10_000),
            active_goal: Some("explore".to_string()),
            ..TemporalUpdateInput::default()
        });
        let later = integrator.update(TemporalUpdateInput {
            monotonic_now_ms: 350,
            wall_clock_unix_ms: Some(9_000),
            active_goal: Some("explore".to_string()),
            ..TemporalUpdateInput::default()
        });
        assert_eq!(later.elapsed_for("goal:explore"), Some(250));
        assert_eq!(later.monotonic_now, TypedTimestamp::monotonic(350));
        assert_eq!(
            later.wall_clock_now,
            Some(TypedTimestamp {
                domain: ClockDomain::WallClock,
                ms: 9_000,
            })
        );
    }
}
