#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NowReconstruction {
    pub t_ms: TimeMs,
    pub body: Option<BodySense>,
    pub memory: Option<MemorySense>,
    pub drives: Option<DriveSense>,
    pub prediction_summary: Option<String>,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default)]
pub struct FeatureExperienceEncoder {
    vectorizers: Vec<BaselineSenseVectorizer>,
}

impl FeatureExperienceEncoder {
    pub fn new() -> Self {
        Self {
            vectorizers: vec![
                BaselineSenseVectorizer::Body,
                BaselineSenseVectorizer::Memory,
                BaselineSenseVectorizer::Drives,
                BaselineSenseVectorizer::Predictions,
                BaselineSenseVectorizer::Surprise,
                BaselineSenseVectorizer::Safety,
                BaselineSenseVectorizer::Reign,
                BaselineSenseVectorizer::Audio,
                BaselineSenseVectorizer::Asr,
                BaselineSenseVectorizer::Range,
                BaselineSenseVectorizer::KinectIr,
                BaselineSenseVectorizer::CreateIr,
            ],
        }
    }
}

impl ExperienceEncoder for FeatureExperienceEncoder {
    fn encode(&mut self, now: &Now) -> Result<ExperienceLatent> {
        let mut z = Vec::new();
        for vectorizer in &self.vectorizers {
            z.extend(vectorizer.encode(now));
        }
        if z.is_empty() {
            z.push(0.0);
        }
        Ok(ExperienceLatent {
            t_ms: now.t_ms,
            z,
            reconstruction_error: 0.0,
            prediction_error: now.surprise.prediction_error,
            confidence: 0.65,
        })
    }
}

impl LatentEncoder for FeatureExperienceEncoder {
    fn encoder_kind(&self) -> &'static str {
        "online-evolved-filters"
    }

    fn encode_input(
        &mut self,
        input: &ExperienceEncodeInput,
        t_ms: TimeMs,
    ) -> Result<ExperienceLatent> {
        let z = input
            .flat_features()
            .into_iter()
            .map(sanitize_feature)
            .collect::<Vec<_>>();
        Ok(ExperienceLatent {
            t_ms,
            z,
            reconstruction_error: 0.0,
            prediction_error: 0.0,
            confidence: 0.55,
        })
    }
}

#[derive(Clone, Debug)]
pub struct RandomProjectionExperienceEncoder {
    z_dim: usize,
    seed: u64,
}

impl RandomProjectionExperienceEncoder {
    pub fn new(z_dim: usize, seed: u64) -> Self {
        Self {
            z_dim: z_dim.max(1),
            seed,
        }
    }

    fn weight(&self, output_index: usize, input_index: usize) -> f32 {
        let mut x = self.seed
            ^ ((output_index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            ^ ((input_index as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9));
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        match x % 6 {
            0 | 1 => -1.0,
            2 | 3 => 0.0,
            _ => 1.0,
        }
    }
}

impl LatentEncoder for RandomProjectionExperienceEncoder {
    fn encoder_kind(&self) -> &'static str {
        "random-projection"
    }

    fn encode_input(
        &mut self,
        input: &ExperienceEncodeInput,
        t_ms: TimeMs,
    ) -> Result<ExperienceLatent> {
        let features = input.flat_features();
        let scale = (features.len().max(1) as f32).sqrt();
        let mut z = vec![0.0; self.z_dim];
        for (out_index, out) in z.iter_mut().enumerate() {
            let sum = features
                .iter()
                .enumerate()
                .map(|(in_index, value)| {
                    sanitize_feature(*value) * self.weight(out_index, in_index)
                })
                .sum::<f32>();
            *out = (sum / scale).tanh();
        }
        Ok(ExperienceLatent {
            t_ms,
            z,
            reconstruction_error: 1.0,
            prediction_error: 0.0,
            confidence: 0.35,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodebookQuantizer {
    pub codes: Vec<Vec<f32>>,
    pub usage: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodebookUsageReport {
    pub code_count: usize,
    pub used_codes: usize,
    pub dead_codes: usize,
    pub usage: Vec<u64>,
}

impl CodebookQuantizer {
    pub fn from_latents(latents: &[Vec<f32>], code_count: usize) -> Self {
        let code_count = code_count.max(1);
        let mut codes = Vec::new();
        for index in 0..code_count {
            let code = latents
                .get(index.saturating_mul(latents.len().max(1)) / code_count)
                .cloned()
                .unwrap_or_default();
            codes.push(code);
        }
        Self {
            codes,
            usage: vec![0; code_count],
        }
    }

    pub fn encode(&mut self, latent: &[f32]) -> usize {
        let (index, _) = self
            .codes
            .iter()
            .enumerate()
            .map(|(index, code)| (index, normalized_distance(code, latent)))
            .min_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or((0, 0.0));
        if let Some(count) = self.usage.get_mut(index) {
            *count = count.saturating_add(1);
        }
        index
    }

    pub fn decode(&self, code_id: usize) -> Vec<f32> {
        self.codes.get(code_id).cloned().unwrap_or_default()
    }

    pub fn report(&self) -> CodebookUsageReport {
        let used_codes = self.usage.iter().filter(|count| **count > 0).count();
        CodebookUsageReport {
            code_count: self.codes.len(),
            used_codes,
            dead_codes: self.codes.len().saturating_sub(used_codes),
            usage: self.usage.clone(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PartialExperienceDecoder;

impl ExperienceDecoder for PartialExperienceDecoder {
    fn decode(&mut self, latent: &ExperienceLatent) -> Result<NowReconstruction> {
        let body = (!latent.z.is_empty()).then(|| BodySense {
            battery_level: latent.z.first().copied().unwrap_or(1.0).clamp(0.0, 1.0),
            charging: latent.z.get(1).copied().unwrap_or(0.0) >= 0.5,
            ..BodySense::default()
        });
        let drives = (latent.z.len() >= 27).then(|| DriveSense {
            battery_hunger: latent
                .z
                .get(21)
                .copied()
                .unwrap_or_default()
                .clamp(0.0, 1.0),
            danger_avoidance: latent
                .z
                .get(22)
                .copied()
                .unwrap_or_default()
                .clamp(0.0, 1.0),
            curiosity: latent
                .z
                .get(23)
                .copied()
                .unwrap_or_default()
                .clamp(0.0, 1.0),
            social_interest: latent
                .z
                .get(24)
                .copied()
                .unwrap_or_default()
                .clamp(0.0, 1.0),
            fatigue: latent
                .z
                .get(25)
                .copied()
                .unwrap_or_default()
                .clamp(0.0, 1.0),
            uncertainty_pressure: latent
                .z
                .get(26)
                .copied()
                .unwrap_or_default()
                .clamp(0.0, 1.0),
        });
        Ok(NowReconstruction {
            t_ms: latent.t_ms,
            body,
            memory: None,
            drives,
            prediction_summary: Some(format!(
                "Partial reconstruction from {} latent features.",
                latent.z.len()
            )),
            confidence: latent.confidence * (1.0 - latent.reconstruction_error).clamp(0.0, 1.0),
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct StasisFuturePredictor;

impl FuturePredictor for StasisFuturePredictor {
    fn predict(
        &mut self,
        latent: &ExperienceLatent,
        _action: &ActionPrimitive,
        offset_ms: TimeMs,
    ) -> Result<FuturePrediction> {
        let seconds = offset_ms as f32 / 1_000.0;
        Ok(FuturePrediction {
            offset_ms,
            predicted_z: latent.z.clone(),
            confidence: (latent.confidence * (-0.18 * seconds).exp()).clamp(0.05, 1.0),
            summary: Some("I expect the situation to remain mostly stable.".to_string()),
        })
    }
}

pub trait SurpriseComputer {
    fn compute(
        &self,
        predicted: &[FuturePrediction],
        actual_z: &ExperienceLatent,
        actual_now: &Now,
    ) -> SurpriseSense;
}

#[derive(Clone, Debug, Default)]
pub struct BaselineSurpriseComputer;

impl SurpriseComputer for BaselineSurpriseComputer {
    fn compute(
        &self,
        predicted: &[FuturePrediction],
        actual_z: &ExperienceLatent,
        actual_now: &Now,
    ) -> SurpriseSense {
        let nearest = predicted
            .iter()
            .min_by_key(|prediction| prediction.offset_ms);
        let prediction_error = nearest
            .map(|prediction| normalized_distance(&prediction.predicted_z, &actual_z.z))
            .unwrap_or(0.0);
        let mut total = prediction_error;
        if actual_now.body.flags.bump_left || actual_now.body.flags.bump_right {
            total += 0.25;
        }
        if cliff_detected(actual_now) || actual_now.body.flags.wheel_drop {
            total += 0.45;
        }
        if actual_now.body.charging
            && !predicted
                .iter()
                .any(|prediction| prediction.summary_contains("charge"))
        {
            total += 0.12;
        }
        if actual_now
            .extensions
            .get("safety.vetoed")
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
        {
            total += 0.2;
        }
        SurpriseSense {
            schema_version: 1,
            total: total.clamp(0.0, 1.0),
            prediction_error,
        }
    }
}

pub trait RewardComputer {
    fn compute(
        &self,
        before: &Now,
        action: Option<&ActionPrimitive>,
        after: &Now,
        surprise: &SurpriseSense,
    ) -> Reward;
}

#[derive(Clone, Debug, Default)]
pub struct BaselineRewardComputer;

impl RewardComputer for BaselineRewardComputer {
    fn compute(
        &self,
        before: &Now,
        action: Option<&ActionPrimitive>,
        after: &Now,
        surprise: &SurpriseSense,
    ) -> Reward {
        let battery_delta = after.body.battery_level - before.body.battery_level;
        let mut value = battery_delta.max(0.0) * (1.0 - before.body.battery_level).clamp(0.0, 1.0);
        let hazard = after.body.flags.bump_left
            || after.body.flags.bump_right
            || cliff_detected(after)
            || after.body.flags.wheel_drop;
        if !before.body.charging && after.body.charging && before.body.battery_level < 0.35 {
            value += 0.35;
        }
        if !hazard {
            value += 0.01;
        }
        if after.body.flags.bump_left || after.body.flags.bump_right {
            value -= 0.25;
        }
        if cliff_detected(after) || after.body.flags.wheel_drop {
            value -= 0.8;
        }
        if battery_delta < 0.0 {
            value += battery_delta * 0.2;
        }
        if matches!(
            action,
            Some(ActionPrimitive::Go { .. } | ActionPrimitive::Explore { .. })
        ) && after.body.velocity.forward_m_s.abs() < 0.01
            && before.body.velocity.forward_m_s.abs() < 0.01
        {
            value -= 0.08;
        }
        if after
            .extensions
            .get("safety.vetoed")
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
        {
            value -= 0.12;
        }
        let hazard_surprise = (surprise.total - surprise.prediction_error).max(0.0);
        value -= hazard_surprise * 0.03;
        if !hazard {
            value += discovery_bonus(before, action, after, surprise);
        }
        Reward { value }
    }
}

fn discovery_bonus(
    before: &Now,
    action: Option<&ActionPrimitive>,
    after: &Now,
    surprise: &SurpriseSense,
) -> f32 {
    let novelty = after.memory.place_novelty.clamp(0.0, 1.0);
    let novelty_delta = (after.memory.place_novelty - before.memory.place_novelty)
        .max(0.0)
        .clamp(0.0, 1.0);
    let newly_visited_places = after
        .memory
        .places_visited
        .saturating_sub(before.memory.places_visited)
        .min(3) as f32;
    let dx = after.body.odometry.x_m - before.body.odometry.x_m;
    let dy = after.body.odometry.y_m - before.body.odometry.y_m;
    let distance_m = (dx * dx + dy * dy).sqrt().clamp(0.0, 0.25);
    let motion_bonus = if matches!(
        action,
        Some(
            ActionPrimitive::Go { .. }
                | ActionPrimitive::Turn { .. }
                | ActionPrimitive::Inspect { .. }
                | ActionPrimitive::Explore { .. }
        )
    ) {
        distance_m * 0.16
    } else {
        0.0
    };
    let prediction_bonus = if matches!(
        action,
        Some(
            ActionPrimitive::Go { .. }
                | ActionPrimitive::Turn { .. }
                | ActionPrimitive::Inspect { .. }
                | ActionPrimitive::Explore { .. }
        )
    ) {
        surprise.prediction_error.clamp(0.0, 1.0) * 0.03
    } else {
        0.0
    };

    (novelty * 0.04
        + novelty_delta * 0.06
        + newly_visited_places * 0.03
        + motion_bonus
        + prediction_bonus)
        .clamp(0.0, 0.12)
}

#[derive(Clone, Debug)]
enum BaselineSenseVectorizer {
    Body,
    Memory,
    Drives,
    Predictions,
    Surprise,
    Safety,
    Reign,
    Audio,
    Asr,
    Range,
    KinectIr,
    CreateIr,
}

impl SenseVectorizer for BaselineSenseVectorizer {
    fn sense_name(&self) -> &'static str {
        match self {
            Self::Body => "body",
            Self::Memory => "memory",
            Self::Drives => "drives",
            Self::Predictions => "predictions",
            Self::Surprise => "surprise",
            Self::Safety => "safety",
            Self::Reign => "reign",
            Self::Audio => "audio",
            Self::Asr => "asr",
            Self::Range => "range",
            Self::KinectIr => "kinect_ir",
            Self::CreateIr => "create_ir",
        }
    }

    fn schema_version(&self) -> u32 {
        1
    }

    fn encode(&self, now: &Now) -> Vec<f32> {
        match self {
            Self::Body => vec![
                now.body.battery_level.clamp(0.0, 1.0),
                bool01(now.body.charging),
                bool01(now.body.flags.bump_left || now.body.flags.bump_right),
                bool01(cliff_detected(now)),
                bool01(now.body.flags.wheel_drop),
                bool01(now.body.flags.wall || now.body.flags.virtual_wall),
                now.body.velocity.forward_m_s.clamp(-1.0, 1.0),
                now.body.velocity.turn_rad_s.clamp(-1.0, 1.0),
                now.body.health.strain.clamp(0.0, 1.0),
                now.body.health.health.clamp(0.0, 1.0),
                now.body.cliff_sensors.left.clamp(0.0, 1.0),
                now.body.cliff_sensors.front_left.clamp(0.0, 1.0),
                now.body.cliff_sensors.front_right.clamp(0.0, 1.0),
                now.body.cliff_sensors.right.clamp(0.0, 1.0),
            ],
            Self::Memory => vec![
                now.memory.place_familiarity.clamp(0.0, 1.0),
                now.memory.place_danger.clamp(0.0, 1.0),
                now.memory.place_charge_value.clamp(0.0, 1.0),
                now.memory.face_familiarity.clamp(0.0, 1.0),
                now.memory.voice_familiarity.clamp(0.0, 1.0),
                (now.memory.similar_situation_count as f32 / 32.0).clamp(0.0, 1.0),
                bool01(now.memory.remembered_warning.is_some()),
                graph_label_pressure(now, "Person"),
                graph_label_pressure(now, "Place"),
                graph_label_pressure(now, "Experience"),
                (now.memory.remembered_relationships.len() as f32 / 32.0).clamp(0.0, 1.0),
                bool01(now.memory.graph_context_summary.is_some()),
            ],
            Self::Drives => vec![
                now.drives.battery_hunger.clamp(0.0, 1.0),
                now.drives.danger_avoidance.clamp(0.0, 1.0),
                now.drives.curiosity.clamp(0.0, 1.0),
                now.drives.social_interest.clamp(0.0, 1.0),
                now.drives.fatigue.clamp(0.0, 1.0),
                now.drives.uncertainty_pressure.clamp(0.0, 1.0),
            ],
            Self::Predictions => vec![
                now.predictions.uncertainty.clamp(0.0, 1.0),
                (now.predictions.expected_events.len() as f32 / 8.0).clamp(0.0, 1.0),
            ],
            Self::Surprise => vec![
                now.surprise.total.clamp(0.0, 1.0),
                now.surprise.prediction_error.clamp(0.0, 1.0),
            ],
            Self::Safety => vec![bool01(
                now.extensions
                    .get("safety.vetoed")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false),
            )],
            Self::Reign => vec![
                bool01(now.reign.active),
                now.reign.human_override_pressure.clamp(0.0, 1.0),
                (now.reign.pending_count as f32 / 8.0).clamp(0.0, 1.0),
            ],
            Self::Audio => vec![bool01(
                now.ear
                    .transcript
                    .as_ref()
                    .map(|text| !text.trim().is_empty())
                    .unwrap_or(false),
            )],
            Self::Asr => asr_features(now),
            Self::Range => vec![now
                .range
                .nearest_m
                .map(|m| (1.0 / (1.0 + m)).clamp(0.0, 1.0))
                .unwrap_or(0.0)],
            Self::KinectIr => kinect_ir_features(now),
            Self::CreateIr => vec![
                bool01(now.body.infrared_character != 0),
                f32::from(now.body.infrared_character) / f32::from(u8::MAX),
            ],
        }
    }
}

fn normalized_distance(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().max(b.len());
    if len == 0 {
        return 0.0;
    }
    let sum = (0..len)
        .map(|idx| {
            let delta =
                a.get(idx).copied().unwrap_or_default() - b.get(idx).copied().unwrap_or_default();
            delta * delta
        })
        .sum::<f32>();
    (sum.sqrt() / (len as f32).sqrt()).clamp(0.0, 1.0)
}

fn bool01(value: bool) -> f32 {
    if value {
        1.0
    } else {
        0.0
    }
}

fn graph_label_pressure(now: &Now, label: &str) -> f32 {
    now.memory
        .remembered_entities
        .iter()
        .filter(|entity| entity.has_label(label))
        .map(|entity| entity.score.clamp(0.0, 1.0))
        .fold(0.0f32, f32::max)
}

pub fn action_features(action: Option<&ActionPrimitive>) -> Vec<f32> {
    danger_action_features(action)
}

fn danger_action_features(action: Option<&ActionPrimitive>) -> Vec<f32> {
    let motor = action_to_motor_command(action);
    let mut out = vec![
        bool01(action.is_none()),
        bool01(matches!(action, Some(ActionPrimitive::Stop))),
        bool01(matches!(action, Some(ActionPrimitive::Go { .. }))),
        bool01(matches!(action, Some(ActionPrimitive::Turn { .. }))),
        bool01(matches!(action, Some(ActionPrimitive::Explore { .. }))),
        bool01(matches!(action, Some(ActionPrimitive::Approach { .. }))),
        bool01(matches!(action, Some(ActionPrimitive::Dock))),
        bool01(matches!(action, Some(ActionPrimitive::Inspect { .. }))),
        bool01(matches!(action, Some(ActionPrimitive::Speak { .. }))),
        bool01(matches!(action, Some(ActionPrimitive::Chirp { .. }))),
        motor.forward.clamp(-1.0, 1.0),
        motor.turn.clamp(-1.0, 1.0),
    ];
    match action {
        Some(ActionPrimitive::Go {
            intensity,
            duration_ms,
        }) => {
            out.push(intensity.clamp(0.0, 1.0));
            out.push((*duration_ms as f32 / 5_000.0).clamp(0.0, 1.0));
            out.push(0.0);
        }
        Some(ActionPrimitive::Turn {
            direction,
            intensity,
            duration_ms,
        }) => {
            out.push(intensity.clamp(0.0, 1.0));
            out.push((*duration_ms as f32 / 5_000.0).clamp(0.0, 1.0));
            out.push(match direction {
                TurnDir::Left => 1.0,
                TurnDir::Right => -1.0,
            });
        }
        Some(ActionPrimitive::Explore { style, duration_ms }) => {
            out.push(match style {
                ExploreStyle::Wander => 0.25,
                ExploreStyle::RandomWalk => 0.5,
                ExploreStyle::WallFollow => 0.9,
            });
            out.push((*duration_ms as f32 / 5_000.0).clamp(0.0, 1.0));
            out.push(0.0);
        }
        _ => {
            out.push(0.0);
            out.push(0.0);
            out.push(0.0);
        }
    }
    out
}

fn danger_body_features(now: &Now) -> Vec<f32> {
    vec![
        now.body.battery_level.clamp(0.0, 1.0),
        bool01(now.body.charging),
        bool01(now.body.flags.bump_left || now.body.flags.bump_right),
        bool01(cliff_detected(now)),
        bool01(now.body.flags.wheel_drop),
        bool01(now.body.flags.wall || now.body.flags.virtual_wall),
        now.body.velocity.forward_m_s.clamp(-1.0, 1.0),
        now.body.velocity.turn_rad_s.clamp(-1.0, 1.0),
        now.body.health.strain.clamp(0.0, 1.0),
        now.body.health.health.clamp(0.0, 1.0),
        now.body.cliff_sensors.left.clamp(0.0, 1.0),
        now.body.cliff_sensors.front_left.clamp(0.0, 1.0),
        now.body.cliff_sensors.front_right.clamp(0.0, 1.0),
        now.body.cliff_sensors.right.clamp(0.0, 1.0),
        now.range
            .nearest_m
            .map(|m| (1.0 / (1.0 + m)).clamp(0.0, 1.0))
            .unwrap_or(0.0),
        now.memory.place_danger.clamp(0.0, 1.0),
        now.surprise.total.clamp(0.0, 1.0),
        now.surprise.prediction_error.clamp(0.0, 1.0),
        bool01(
            now.extensions
                .get("safety.vetoed")
                .and_then(|value| value.as_bool())
                .unwrap_or(false),
        ),
    ]
}

fn charge_body_features(now: &Now) -> Vec<f32> {
    let mut out = vec![
        now.body.battery_level.clamp(0.0, 1.0),
        bool01(now.body.charging),
        now.drives.battery_hunger.clamp(0.0, 1.0),
        now.body.velocity.forward_m_s.clamp(-1.0, 1.0),
        now.body.velocity.turn_rad_s.clamp(-1.0, 1.0),
        now.range
            .nearest_m
            .map(|m| (1.0 / (1.0 + m)).clamp(0.0, 1.0))
            .unwrap_or(0.0),
        extension_value(now, "sim.world", 3)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0),
        extension_value(now, "sim.world", 4)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0),
    ];
    out.extend(kinect_ir_features(now));
    out
}

fn charge_memory_features(now: &Now) -> Vec<f32> {
    vec![
        now.memory.place_charge_value.clamp(0.0, 1.0),
        now.memory.place_familiarity.clamp(0.0, 1.0),
        now.memory.place_danger.clamp(0.0, 1.0),
        (now.memory.similar_situation_count as f32 / 32.0).clamp(0.0, 1.0),
        bool01(matches!(
            now.memory.best_remembered_action,
            Some(ActionPrimitive::Dock)
        )),
    ]
}

fn action_value_body_features(now: &Now) -> Vec<f32> {
    let mut out = danger_body_features(now);
    out.extend(charge_body_features(now));
    out
}

fn action_value_drive_features(now: &Now) -> Vec<f32> {
    vec![
        now.drives.battery_hunger.clamp(0.0, 1.0),
        now.drives.danger_avoidance.clamp(0.0, 1.0),
        now.drives.curiosity.clamp(0.0, 1.0),
        now.drives.social_interest.clamp(0.0, 1.0),
        now.drives.fatigue.clamp(0.0, 1.0),
        now.drives.uncertainty_pressure.clamp(0.0, 1.0),
    ]
}

fn action_value_memory_features(now: &Now, action: Option<&ActionPrimitive>) -> Vec<f32> {
    let best_matches = match (&now.memory.best_remembered_action, action) {
        (Some(best), Some(action)) => best == action,
        _ => false,
    };
    vec![
        now.memory.place_familiarity.clamp(0.0, 1.0),
        now.memory.place_danger.clamp(0.0, 1.0),
        now.memory.place_charge_value.clamp(0.0, 1.0),
        now.memory.face_familiarity.clamp(0.0, 1.0),
        now.memory.voice_familiarity.clamp(0.0, 1.0),
        (now.memory.similar_situation_count as f32 / 32.0).clamp(0.0, 1.0),
        bool01(best_matches),
        bool01(now.memory.remembered_warning.is_some()),
    ]
}

fn action_value_prediction_features(now: &Now) -> Vec<f32> {
    let danger = now.predictions.danger_model.map(|value| DangerOutput {
        bump_risk: value.bump_risk,
        cliff_risk: value.cliff_risk,
        wheel_drop_risk: value.wheel_drop_risk,
        stuck_risk: value.stuck_risk,
        confidence: value.confidence,
    });
    let charge = now.predictions.charge_model.map(|value| ChargeOutput {
        charge_probability: value.charge_probability,
        expected_battery_delta: value.expected_battery_delta,
        dock_likelihood: value.dock_likelihood,
        confidence: value.confidence,
    });
    action_value_prediction_features_from_outputs(now, danger, charge)
}

fn action_value_prediction_features_from_outputs(
    now: &Now,
    danger: Option<DangerOutput>,
    charge: Option<ChargeOutput>,
) -> Vec<f32> {
    let danger = danger.unwrap_or_default();
    let charge = charge.unwrap_or_default();
    vec![
        danger.bump_risk.clamp(0.0, 1.0),
        danger.cliff_risk.clamp(0.0, 1.0),
        danger.wheel_drop_risk.clamp(0.0, 1.0),
        danger.stuck_risk.clamp(0.0, 1.0),
        danger.confidence.clamp(0.0, 1.0),
        charge.charge_probability.clamp(0.0, 1.0),
        charge.expected_battery_delta.clamp(-1.0, 1.0),
        charge.dock_likelihood.clamp(0.0, 1.0),
        charge.confidence.clamp(0.0, 1.0),
        now.predictions.uncertainty.clamp(0.0, 1.0),
        (now.predictions.expected_events.len() as f32 / 8.0).clamp(0.0, 1.0),
        bool01(
            now.extensions
                .get("safety.vetoed")
                .and_then(|value| value.as_bool())
                .unwrap_or(false),
        ),
    ]
}

fn eye_next_features(now: &Now) -> Vec<f32> {
    let Some(frame) = now.eye.frames.last() else {
        return vec![0.0; 16];
    };
    if frame.is_empty() {
        return vec![0.0; 16];
    }
    let mut out = Vec::with_capacity(16);
    let chunk = (frame.len() / 16).max(1);
    for part in frame.chunks(chunk).take(16) {
        let avg = part.iter().copied().map(sanitize_feature).sum::<f32>() / part.len() as f32;
        out.push(avg.clamp(0.0, 1.0));
    }
    out.resize(16, 0.0);
    out
}

fn ear_next_features(now: &Now) -> Vec<f32> {
    let mut out = Vec::with_capacity(16);
    if let Some(features) = ear_frame_features(now) {
        let chunk = (features.len() / 8).max(1);
        for part in features.chunks(chunk).take(8) {
            let avg = part.iter().copied().map(sanitize_feature).sum::<f32>() / part.len() as f32;
            out.push(avg.clamp(-1.0, 1.0));
        }
    }
    out.resize(8, 0.0);
    out.extend(asr_features(now));
    out
}

fn asr_features(now: &Now) -> Vec<f32> {
    let transcript = now
        .ear
        .asr
        .committed_transcript
        .as_deref()
        .or(now.ear.asr.transcript.as_deref())
        .or(now.ear.asr.possible_transcript.as_deref())
        .or(now.ear.transcript.as_deref())
        .map(str::trim)
        .filter(|text| !text.is_empty());
    let Some(transcript) = transcript else {
        return vec![0.0; 8];
    };

    let word_count = now
        .ear
        .asr
        .word_count
        .map(f32::from)
        .unwrap_or_else(|| count_transcript_words(transcript) as f32);
    let char_count = transcript.chars().count() as f32;
    let punctuation_count = transcript
        .chars()
        .filter(|ch| matches!(ch, '.' | ',' | '?' | '!' | ';' | ':'))
        .count() as f32;
    let duration_ms = now
        .ear
        .asr
        .duration_ms
        .or_else(|| Some(now.ear.asr.end_ms?.saturating_sub(now.ear.asr.start_ms?)))
        .unwrap_or_default();
    let sequence_span = match (now.ear.asr.sequence_start, now.ear.asr.sequence_end) {
        (Some(start), Some(end)) => end.saturating_sub(start).saturating_add(1),
        _ => 0,
    };

    vec![
        1.0,
        bool01(now.ear.asr.is_final || now.ear.asr.committed_transcript.is_some()),
        now.ear.asr.confidence.clamp(0.0, 1.0),
        (word_count / 32.0).clamp(0.0, 1.0),
        (char_count / 160.0).clamp(0.0, 1.0),
        (duration_ms as f32 / 20_000.0).clamp(0.0, 1.0),
        (sequence_span as f32 / 128.0).clamp(0.0, 1.0),
        (punctuation_count / 8.0).clamp(0.0, 1.0),
    ]
}

fn count_transcript_words(transcript: &str) -> usize {
    transcript
        .split_whitespace()
        .filter(|word| word.chars().any(|ch| ch.is_alphanumeric()))
        .count()
}

fn kinect_ir_features(now: &Now) -> Vec<f32> {
    if now.kinect.ir.is_empty() {
        return vec![0.0, 0.0, 0.0, 0.0];
    }
    let len = now.kinect.ir.len() as f32;
    let sum = now.kinect.ir.iter().copied().sum::<f32>();
    let max = now
        .kinect
        .ir
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let bright = now.kinect.ir.iter().filter(|value| **value >= 0.7).count() as f32 / len;
    vec![
        (sum / len).clamp(0.0, 1.0),
        max.clamp(0.0, 1.0),
        bright.clamp(0.0, 1.0),
        (len / 1024.0).clamp(0.0, 1.0),
    ]
}

fn extension_value(now: &Now, name: &str, index: usize) -> Option<f32> {
    now.extensions
        .get(name)?
        .get("values")?
        .as_array()?
        .get(index)?
        .as_f64()
        .map(|value| value as f32)
}

fn cliff_detected(now: &Now) -> bool {
    now.body.flags.cliff_left
        || now.body.flags.cliff_front_left
        || now.body.flags.cliff_front_right
        || now.body.flags.cliff_right
        || now.body.cliff_sensors.max() >= 0.5
}

fn sanitize_feature(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn unit_to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

trait PredictionSummaryExt {
    fn summary_contains(&self, needle: &str) -> bool;
}

impl PredictionSummaryExt for FuturePrediction {
    fn summary_contains(&self, needle: &str) -> bool {
        self.summary
            .as_ref()
            .map(|summary| summary.to_lowercase().contains(needle))
            .unwrap_or(false)
    }
}
