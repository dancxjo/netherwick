use anyhow::Result;
use async_trait::async_trait;
use netherwick_now::{
    EarSense, ExtensionSense, EyeSense, FaceSense, GpsSense, ImuSense, RangeSense, VoiceSense,
};

#[async_trait]
pub trait SenseProducer {
    async fn poll(&mut self) -> Result<SensePacket>;
}

#[derive(Clone, Debug, PartialEq)]
pub enum SensePacket {
    Eye(EyeSense),
    Ear(EarSense),
    Range(RangeSense),
    Imu(ImuSense),
    Gps(GpsSense),
    Face(FaceSense),
    Voice(VoiceSense),
    Extension(ExtensionSense),
}
