pub trait NeuralModel<I, O> {
    fn predict(&self, input: I) -> Result<O>;
}

pub trait OnlineTrainer<I, O> {
    fn train_step(&mut self, sample: TrainingSample<I, O>) -> Result<TrainStats>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TrainStats {
    pub loss: f32,
    pub samples_seen: u64,
    pub improved: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DangerTrainStats {
    pub loss: f32,
    pub samples_seen: u64,
    pub improved: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DangerShadowMetric {
    pub observed_at_ms: u64,
    pub hardcoded: DangerOutput,
    pub model: DangerOutput,
    pub target: DangerTarget,
    pub loss: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChargeTrainStats {
    pub loss: f32,
    pub samples_seen: u64,
    pub improved: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChargeShadowMetric {
    pub observed_at_ms: u64,
    pub hardcoded: ChargeOutput,
    pub model: ChargeOutput,
    pub target: ChargeTarget,
    pub loss: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionValueTrainStats {
    pub loss: f32,
    pub samples_seen: u64,
    pub improved: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionValueShadowMetric {
    pub observed_at_ms: u64,
    pub hardcoded: ActionValueOutput,
    pub model: ActionValueOutput,
    pub target: ActionValueTarget,
    pub loss: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FutureShadowMetric {
    pub t_ms: TimeMs,
    pub offset_ms: TimeMs,
    pub hardcoded_error: f32,
    pub model_error: f32,
    pub selected_error: f32,
    pub model_loss: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EyeNextTrainStats {
    pub loss: f32,
    pub samples_seen: u64,
    pub improved: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EyeNextShadowMetric {
    pub observed_at_ms: u64,
    pub hardcoded: EyeNextOutput,
    pub model: EyeNextOutput,
    pub target: EyeNextTarget,
    pub loss: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EarNextTrainStats {
    pub loss: f32,
    pub samples_seen: u64,
    pub improved: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EarNextShadowMetric {
    pub observed_at_ms: u64,
    pub hardcoded: EarNextOutput,
    pub model: EarNextOutput,
    pub target: EarNextTarget,
    pub loss: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperienceAutoencoderTrainStats {
    pub loss: f32,
    pub samples_seen: u64,
    pub improved: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperienceAutoencoderShadowMetric {
    pub t_ms: TimeMs,
    pub baseline_z_norm: f32,
    pub model_z_norm: f32,
    pub z_disagreement: f32,
    pub reconstruction_loss: f32,
    pub selected: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperienceAutoencoderPrediction {
    pub encoded: ExperienceEncodeOutput,
    pub decoded: ExperienceDecodeOutput,
}

#[derive(Clone, Debug, Default)]
pub struct HardcodedDangerPredictor;

impl HardcodedDangerPredictor {
    pub fn predict_from_now(&self, now: &Now, input: &DangerInput) -> DangerOutput {
        let mut bump_risk = 0.0_f32;
        let nearest_m = now.range.nearest_m;
        if let Some(distance) = nearest_m {
            if distance <= 0.12 {
                bump_risk = bump_risk.max(0.9);
            } else if distance <= 0.25 {
                bump_risk = bump_risk.max(0.65);
            } else if distance <= 0.45 {
                bump_risk = bump_risk.max(0.35);
            }
        }
        if now.body.flags.bump_left || now.body.flags.bump_right {
            bump_risk = 1.0;
        }
        bump_risk = bump_risk.max(surface_anticipated_collision_risk(now));

        let cliff_risk = if now.body.flags.cliff_left
            || now.body.flags.cliff_front_left
            || now.body.flags.cliff_front_right
            || now.body.flags.cliff_right
            || now.body.cliff_sensors.max() >= 0.5
        {
            1.0
        } else {
            0.0
        };
        let wheel_drop_risk = if now.body.flags.wheel_drop { 1.0 } else { 0.0 };
        let commanded_forward = input.action_features.get(10).copied().unwrap_or(0.0) > 0.01;
        let explore = input.action_features.get(4).copied().unwrap_or(0.0) > 0.5;
        let stuck_risk =
            if (commanded_forward || explore) && now.body.velocity.forward_m_s.abs() < 0.01 {
                0.85
            } else {
                0.0
            };
        let safety_vetoed = now
            .extensions
            .get("safety.vetoed")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);

        DangerOutput {
            bump_risk,
            cliff_risk,
            wheel_drop_risk,
            stuck_risk,
            confidence: if safety_vetoed || cliff_risk >= 1.0 || wheel_drop_risk >= 1.0 {
                1.0
            } else if nearest_m.is_some() || bump_risk > 0.0 || stuck_risk > 0.0 {
                0.75
            } else {
                0.6
            },
        }
    }
}

impl NeuralModel<(DangerInput, Now), DangerOutput> for HardcodedDangerPredictor {
    fn predict(&self, input: (DangerInput, Now)) -> Result<DangerOutput> {
        Ok(self.predict_from_now(&input.1, &input.0))
    }
}

#[derive(Clone, Debug, Default)]
pub struct HardcodedChargePredictor;

impl HardcodedChargePredictor {
    pub fn predict_from_now(&self, now: &Now, input: &ChargeInput) -> ChargeOutput {
        let already_charging = now.body.charging;
        let low_battery = now.body.battery_level < 0.35;
        let dock_action = input.action_features.get(6).copied().unwrap_or(0.0) > 0.5;
        let memory_charge = now.memory.place_charge_value.clamp(0.0, 1.0);
        let charger_near = input.body_features.get(6).copied().unwrap_or(0.0);
        let charger_visible = input.body_features.get(7).copied().unwrap_or(0.0);
        let ir_bright = input.body_features.get(9).copied().unwrap_or(0.0);

        let mut charge_probability = 0.05_f32;
        let mut expected_battery_delta = -0.002_f32;
        let mut dock_likelihood = 0.05_f32;
        let mut confidence = 0.55_f32;

        if already_charging {
            charge_probability = 0.95;
            expected_battery_delta = 0.03;
            dock_likelihood = 0.9;
            confidence = 0.95;
        }
        if memory_charge >= 0.7 {
            charge_probability = charge_probability.max(0.65);
            dock_likelihood = dock_likelihood.max(0.65);
            confidence = confidence.max(0.7);
        } else if memory_charge >= 0.4 {
            charge_probability = charge_probability.max(0.45);
            dock_likelihood = dock_likelihood.max(0.45);
        }
        if dock_action && low_battery && (already_charging || charger_near >= 0.5) {
            charge_probability = charge_probability.max(0.55);
            expected_battery_delta = expected_battery_delta.max(0.015);
            dock_likelihood = dock_likelihood.max(0.7);
            confidence = confidence.max(0.7);
        }
        if charger_near >= 0.5 {
            charge_probability = charge_probability.max(0.85);
            expected_battery_delta = expected_battery_delta.max(0.025);
            dock_likelihood = dock_likelihood.max(0.9);
            confidence = confidence.max(0.85);
        } else if charger_visible >= 0.5 {
            charge_probability = charge_probability.max(0.80);
            expected_battery_delta = expected_battery_delta.max(0.015);
            dock_likelihood = dock_likelihood.max(0.35);
            confidence = confidence.max(0.80);
        } else if ir_bright >= 0.7 {
            charge_probability = charge_probability.max(0.35);
            dock_likelihood = dock_likelihood.max(0.35);
        }

        ChargeOutput {
            charge_probability: charge_probability.clamp(0.0, 1.0),
            expected_battery_delta: expected_battery_delta.clamp(-1.0, 1.0),
            dock_likelihood: dock_likelihood.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

impl NeuralModel<(ChargeInput, Now), ChargeOutput> for HardcodedChargePredictor {
    fn predict(&self, input: (ChargeInput, Now)) -> Result<ChargeOutput> {
        Ok(self.predict_from_now(&input.1, &input.0))
    }
}

#[derive(Clone, Debug, Default)]
pub struct HardcodedActionValuePredictor;

impl HardcodedActionValuePredictor {
    pub fn predict_from_now(&self, now: &Now, input: &ActionValueInput) -> ActionValueOutput {
        let battery_low = now.body.battery_level < 0.35;
        let dock_action = input.action_features.get(6).copied().unwrap_or(0.0) > 0.5;
        let go_action = input.action_features.get(2).copied().unwrap_or(0.0) > 0.5;
        let turn_action = input.action_features.get(3).copied().unwrap_or(0.0) > 0.5;
        let explore_action = input.action_features.get(4).copied().unwrap_or(0.0) > 0.5;
        let approach_action = input.action_features.get(5).copied().unwrap_or(0.0) > 0.5;
        let stop_action = input.action_features.get(1).copied().unwrap_or(0.0) > 0.5;
        let forward = input
            .action_features
            .get(10)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let bump_risk = input.prediction_features.first().copied().unwrap_or(0.0);
        let cliff_risk = input.prediction_features.get(1).copied().unwrap_or(0.0);
        let wheel_drop_risk = input.prediction_features.get(2).copied().unwrap_or(0.0);
        let stuck_risk = input.prediction_features.get(3).copied().unwrap_or(0.0);
        let danger_confidence = input.prediction_features.get(4).copied().unwrap_or(0.0);
        let charge_probability = input.prediction_features.get(5).copied().unwrap_or(0.0);
        let expected_battery_delta = input.prediction_features.get(6).copied().unwrap_or(0.0);
        let dock_likelihood = input.prediction_features.get(7).copied().unwrap_or(0.0);
        let prediction_uncertainty = input.prediction_features.get(9).copied().unwrap_or(0.0);
        let safety_veto_likely = input.prediction_features.get(11).copied().unwrap_or(0.0);
        let surface_collision_risk = surface_anticipated_collision_risk(now);
        let memory_danger = input.memory_features.get(1).copied().unwrap_or(0.0);
        let memory_charge = input.memory_features.get(2).copied().unwrap_or(0.0);
        let remembered_best = input.memory_features.get(6).copied().unwrap_or(0.0);
        let has_warning = input.memory_features.get(7).copied().unwrap_or(0.0);

        let mut value = 0.0_f32;
        value += charge_probability * if battery_low { 0.75 } else { 0.25 };
        value += expected_battery_delta * if battery_low { 0.85 } else { 0.35 };
        value += dock_likelihood * if dock_action { 0.35 } else { 0.12 };
        if dock_action && (battery_low || memory_charge > 0.5) {
            value += 0.25;
        }
        if (explore_action || turn_action) && !battery_low {
            value += 0.12 + prediction_uncertainty * 0.08;
        }
        if approach_action && memory_charge > 0.5 {
            value += 0.12;
        }
        if remembered_best > 0.5 {
            value += 0.12;
        }
        if battery_low && (go_action || explore_action) && charge_probability < 0.25 {
            value -= 0.35;
        }
        value -= bump_risk * (0.45 + forward * 0.25);
        value -= surface_collision_risk * (0.35 + forward * 0.35);
        value -= cliff_risk * 0.9;
        value -= wheel_drop_risk * 0.95;
        value -= stuck_risk * 0.25;
        value -= memory_danger * 0.2;
        value -= has_warning * 0.15;
        value -= safety_veto_likely * 0.55;
        if stop_action && (cliff_risk > 0.5 || wheel_drop_risk > 0.5 || safety_veto_likely > 0.5) {
            value += 0.18;
        }

        ActionValueOutput {
            value: value.clamp(-1.0, 1.0),
            confidence: (0.45 + danger_confidence * 0.25 + charge_probability * 0.1)
                .clamp(0.0, 1.0),
        }
    }
}

fn surface_anticipated_collision_risk(now: &Now) -> f32 {
    now.extensions
        .get("surface.scene_graph")
        .and_then(|value| value.get("anticipation"))
        .and_then(|value| value.get("max_collision_risk"))
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0) as f32
}

impl NeuralModel<(ActionValueInput, Now), ActionValueOutput> for HardcodedActionValuePredictor {
    fn predict(&self, input: (ActionValueInput, Now)) -> Result<ActionValueOutput> {
        Ok(self.predict_from_now(&input.1, &input.0))
    }
}

#[derive(Clone, Debug, Default)]
pub struct CopyCurrentEyePredictor;

impl CopyCurrentEyePredictor {
    pub fn predict_from_now(&self, now: &Now, _input: &EyeNextInput) -> EyeNextOutput {
        match pete_experience::eye_frame_rgb(now) {
            Some((width, height, rgb)) => EyeNextOutput {
                width,
                height,
                rgb,
                confidence: 0.75,
            },
            None => EyeNextOutput {
                width: EYE_NEXT_WIDTH,
                height: EYE_NEXT_HEIGHT,
                rgb: vec![0; EYE_NEXT_RGB_LEN],
                confidence: 0.0,
            },
        }
    }
}

impl NeuralModel<(EyeNextInput, Now), EyeNextOutput> for CopyCurrentEyePredictor {
    fn predict(&self, input: (EyeNextInput, Now)) -> Result<EyeNextOutput> {
        Ok(self.predict_from_now(&input.1, &input.0))
    }
}

#[derive(Clone, Debug, Default)]
pub struct CopyCurrentEarPredictor;

impl CopyCurrentEarPredictor {
    pub fn predict_from_now(&self, now: &Now, input: &EarNextInput) -> EarNextOutput {
        let Some(features) = pete_experience::ear_frame_features(now) else {
            return EarNextOutput {
                sample_rate_hz: 0,
                channels: 0,
                pcm: Vec::new(),
                features: vec![0.0; input.ear_features.len()],
                confidence: 0.0,
            };
        };
        EarNextOutput {
            sample_rate_hz: 0,
            channels: 0,
            pcm: Vec::new(),
            features,
            confidence: 0.75,
        }
    }
}

impl NeuralModel<(EarNextInput, Now), EarNextOutput> for CopyCurrentEarPredictor {
    fn predict(&self, input: (EarNextInput, Now)) -> Result<EarNextOutput> {
        Ok(self.predict_from_now(&input.1, &input.0))
    }
}
