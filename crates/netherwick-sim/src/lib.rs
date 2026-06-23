use anyhow::Result;
use async_trait::async_trait;
use netherwick_body::{BodySense, MotorCommand, RobotBody};
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub struct SimBody {
    body: BodySense,
    _rng: StdRng,
}

impl SimBody {
    pub fn new(seed: u64) -> Self {
        Self {
            body: BodySense::default(),
            _rng: StdRng::seed_from_u64(seed),
        }
    }
}

#[async_trait]
impl RobotBody for SimBody {
    async fn read_body(&mut self) -> Result<BodySense> {
        Ok(self.body.clone())
    }

    async fn apply_motor(&mut self, cmd: MotorCommand) -> Result<()> {
        self.body.velocity.forward_m_s = cmd.forward;
        self.body.velocity.turn_rad_s = cmd.turn;
        self.body.odometry.x_m += cmd.forward * 0.1;
        self.body.odometry.heading_rad += cmd.turn * 0.1;
        self.body.battery_level = (self.body.battery_level - cmd.forward.abs() * 0.01).max(0.0);
        self.body.last_update_ms = self.body.last_update_ms.saturating_add(100);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ArenaConfig {
    pub width_m: f32,
    pub height_m: f32,
}
