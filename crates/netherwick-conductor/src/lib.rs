use anyhow::Result;
use netherwick_actions::{
    ActionPrimitive, ApproachTarget, ExploreStyle, InspectTarget, ReignCommand, ReignMode, TurnDir,
};
use netherwick_body::BodySense;
use netherwick_experience::ExperienceLatent;
use netherwick_now::{
    DriveSense, LlmSense, MemorySense, PredictionSense, ReignSense, SafetySense, SurpriseSense,
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
    pub reign: ReignSense,
    pub body: BodySense,
    pub proposals: Vec<ActionPrimitive>,
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
        if let Some(action) = direct_reign_action(&input) {
            return Ok(action);
        }
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
        if let Some(action) = assisted_reign_action(&input) {
            return Ok(action);
        }
        if let Some(action) = input.proposals.last() {
            return Ok(action.clone());
        }
        Ok(ActionPrimitive::Explore {
            style: ExploreStyle::RandomWalk,
            duration_ms: 1_000,
        })
    }
}

fn direct_reign_action(input: &ConductorInput) -> Option<ActionPrimitive> {
    let reign_input = input.reign.latest.as_ref()?;
    let action = reign_input.command.to_action()?;
    if matches!(reign_input.command, ReignCommand::Stop) || reign_input.mode == ReignMode::Direct {
        return Some(action);
    }

    None
}

fn assisted_reign_action(input: &ConductorInput) -> Option<ActionPrimitive> {
    let reign_input = input.reign.latest.as_ref()?;
    let action = reign_input.command.to_action()?;
    if reign_input.mode == ReignMode::Assist
        && input.proposals.iter().any(|proposal| proposal == &action)
    {
        Some(action)
    } else {
        None
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
            reign: ReignSense::default(),
            body,
            proposals: Vec::new(),
        };

        assert_eq!(conductor.choose(input).unwrap(), ActionPrimitive::Dock);
    }

    #[test]
    fn direct_reign_overrides_default_curiosity_drive() {
        let mut conductor = SimpleConductor::default();
        let command = ReignCommand::Turn {
            direction: TurnDir::Right,
            intensity: 0.4,
            duration_ms: 500,
        };
        let mut reign = ReignSense::default();
        reign.active = true;
        reign.mode = Some(ReignMode::Direct);
        reign.latest = Some(netherwick_actions::ReignInput {
            id: Default::default(),
            issued_at_ms: 100,
            expires_at_ms: 1_000,
            source: netherwick_actions::ReignSource::WebRemote,
            mode: ReignMode::Direct,
            command: command.clone(),
            priority: 1.0,
            note: None,
        });
        let mut drives = DriveSense::default();
        drives.curiosity = 1.0;
        let input = ConductorInput {
            latent: ExperienceLatent::default(),
            drives,
            memory: MemorySense::default(),
            predictions: PredictionSense::default(),
            surprise: SurpriseSense::default(),
            llm: LlmSense::default(),
            safety: SafetySense::default(),
            reign,
            body: BodySense::default(),
            proposals: Vec::new(),
        };

        assert_eq!(
            conductor.choose(input).unwrap(),
            command.to_action().unwrap()
        );
    }
}
