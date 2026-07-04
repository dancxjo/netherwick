use heapless::Deque;

use crate::body;
use crate::events::BrainstemEvent;
use crate::hardware::BrainstemHardware;

pub struct CreatePower;

impl CreatePower {
    pub const fn new() -> Self {
        Self
    }

    pub fn wake<H, const N: usize>(
        &mut self,
        hardware: &mut H,
        events: &mut Deque<BrainstemEvent, N>,
    ) where
        H: BrainstemHardware,
    {
        let _ = events.push_back(BrainstemEvent::CreatePowerOnRequested);
        self.toggle(hardware, events);
        hardware.delay_ms(body::CREATE_WAKE_WAIT_MS);
    }

    pub fn sleep<H, const N: usize>(
        &mut self,
        hardware: &mut H,
        events: &mut Deque<BrainstemEvent, N>,
    ) where
        H: BrainstemHardware,
    {
        let _ = events.push_back(BrainstemEvent::CreatePowerOffRequested);
        self.toggle(hardware, events);
    }

    fn toggle<H, const N: usize>(&mut self, hardware: &mut H, events: &mut Deque<BrainstemEvent, N>)
    where
        H: BrainstemHardware,
    {
        hardware.set_power_toggle(true);
        hardware.delay_ms(body::POWER_TOGGLE_PULSE_MS);
        hardware.set_power_toggle(false);
        let _ = events.push_back(BrainstemEvent::CreatePowerToggled);
    }

    pub fn pulse_brc<H, const N: usize>(
        &mut self,
        hardware: &mut H,
        events: &mut Deque<BrainstemEvent, N>,
    ) where
        H: BrainstemHardware,
    {
        if !body::CREATE_BRC_ENABLED {
            return;
        }

        let _ = events.push_back(BrainstemEvent::CreateBrcPulseRequested);
        hardware.set_brc(false);
        hardware.delay_ms(body::BRC_LOW_PULSE_MS);
        hardware.set_brc(true);
        hardware.delay_ms(body::POST_BRC_SETTLE_MS);
        let _ = events.push_back(BrainstemEvent::CreateBrcPulsed);
    }
}
