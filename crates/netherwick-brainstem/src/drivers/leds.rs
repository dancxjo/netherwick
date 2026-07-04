use crate::hardware::BrainstemHardware;

pub struct Leds;

impl Leds {
    pub const fn new() -> Self {
        Self
    }

    pub fn boot_indicator<H>(&mut self, hardware: &mut H)
    where
        H: BrainstemHardware,
    {
        for _ in 0..3 {
            hardware.set_indicators(true);
            hardware.delay_ms(120);
            hardware.set_indicators(false);
            hardware.delay_ms(120);
        }
        hardware.set_indicators(true);
        hardware.delay_ms(250);
    }

    #[allow(dead_code)]
    pub fn idle_once<H>(&mut self, hardware: &mut H)
    where
        H: BrainstemHardware,
    {
        hardware.set_primary_indicator(true);
        hardware.delay_ms(crate::body::IDLE_BLINK_MS);
        hardware.set_primary_indicator(false);
        hardware.delay_ms(crate::body::IDLE_BLINK_MS);
    }

    #[allow(dead_code)]
    pub fn error_once<H>(&mut self, hardware: &mut H)
    where
        H: BrainstemHardware,
    {
        for _ in 0..3 {
            hardware.set_indicators(true);
            hardware.delay_ms(crate::body::ERROR_BLINK_MS);
            hardware.set_indicators(false);
            hardware.delay_ms(crate::body::ERROR_BLINK_MS);
        }
        hardware.delay_ms(crate::body::ERROR_PAUSE_MS);
    }
}
