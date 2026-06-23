use anyhow::Result;
use netherwick_actions::{ActionPrimitive, ApproachTarget, ExploreStyle, InspectTarget, TurnDir};
use netherwick_body::BodySense;
use netherwick_experience::ExperienceLatent;
use netherwick_now::{
    DriveSense, LlmSense, MemorySense, PredictionSense, SafetySense, SurpriseSense,
};
use serde::{Deserialize, Serialize};

pub trait Conductor {
    fn choose(&mut self, input: ConductorInput) -> Result<ActionPrimitive>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConductorInput {
    pub latent: ExperienceLatent,
    pub drives: DriveSense,
    pub memory: MemorySense,
    pub predictions: PredictionSense,
    pub surprise: SurpriseSense,
    pub llm: LlmSense,
    pub safety: SafetySense,
    pub body: BodySense,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConductorConfig {
    pub critical_battery: f32,
    pub low_battery: f32,
    pub danger_threshold: f32,
    pub novelty_threshold: f32,
}

impl Default for ConductorConfig {
    fn default() -> Self {
        Self {
            critical_battery: 0.10,
            low_battery: 0.20,
            danger_threshold: 0.70,
            novelty_threshold: 0.50,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SimpleConductor {
    pub config: ConductorConfig,
}

impl Conductor for SimpleConductor {
    fn choose(&mut self, input: ConductorInput) -> Result<ActionPrimitive> {
        if input.body.flags.wheel_drop {
            return Ok(ActionPrimitive::Stop);
        }
        if input.body.battery_level <= self.config.critical_battery {
            return Ok(ActionPrimitive::Dock);
        }
        if input.memory.place_danger >= self.config.danger_threshold
            || input.drives.danger_avoidance >= self.config.danger_threshold
        {
            return Ok(ActionPrimitive::Turn {
                direction: TurnDir::Left,
                intensity: 0.5,
                duration_ms: 1_000,
            });
        }
        if input.body.battery_level <= self.config.low_battery
            && input.memory.place_charge_value > 0.5
        {
            return Ok(ActionPrimitive::Approach {
                target: ApproachTarget::Charger,
            });
        }
        if input.drives.curiosity >= self.config.novelty_threshold {
            return Ok(ActionPrimitive::Inspect {
                target: InspectTarget::Novelty,
            });
        }
        Ok(ActionPrimitive::Explore {
            style: ExploreStyle::RandomWalk,
            duration_ms: 1_000,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn docks_on_critical_battery() {
        let mut conductor = SimpleConductor::default();
        let mut body = BodySense::default();
        body.battery_level = 0.05;
        let input = ConductorInput {
            latent: ExperienceLatent::default(),
            drives: DriveSense::default(),
            memory: MemorySense::default(),
            predictions: PredictionSense::default(),
            surprise: SurpriseSense::default(),
            llm: LlmSense::default(),
            safety: SafetySense::default(),
            body,
        };

        assert_eq!(conductor.choose(input).unwrap(), ActionPrimitive::Dock);
    }
}
