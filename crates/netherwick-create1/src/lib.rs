use anyhow::Result;
use async_trait::async_trait;
use netherwick_body::{BodySense, MotorCommand, RobotBody};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Create1Config {
    pub port: Option<String>,
    pub baud_rate: u32,
}

#[derive(Debug, Default)]
pub struct Create1Body {
    body: BodySense,
}

#[async_trait]
impl RobotBody for Create1Body {
    async fn read_body(&mut self) -> Result<BodySense> {
        Ok(self.body.clone())
    }

    async fn apply_motor(&mut self, cmd: MotorCommand) -> Result<()> {
        self.body.velocity.forward_m_s = cmd.forward;
        self.body.velocity.turn_rad_s = cmd.turn;
        Ok(())
    }
}
