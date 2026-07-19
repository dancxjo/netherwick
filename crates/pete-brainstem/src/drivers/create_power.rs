#![allow(dead_code)]

use heapless::Deque;

use crate::body;
use crate::events::BrainstemEvent;
use crate::hardware::BrainstemHardware;
use crate::status;

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
        push_event(events, BrainstemEvent::CreatePowerOnRequested);
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
        push_event(events, BrainstemEvent::CreatePowerOffRequested);
        self.toggle(hardware, events);
    }

    fn toggle<H, const N: usize>(&mut self, hardware: &mut H, events: &mut Deque<BrainstemEvent, N>)
    where
        H: BrainstemHardware,
    {
        hardware.begin_power_toggle_pulse();
        hardware.delay_ms(body::POWER_TOGGLE_PULSE_MS);
        hardware.end_power_toggle_pulse();
        push_event(events, BrainstemEvent::CreatePowerToggled);
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

        push_event(events, BrainstemEvent::CreateBrcPulseRequested);
        hardware.set_brc(false);
        hardware.delay_ms(body::BRC_LOW_PULSE_MS);
        hardware.set_brc(true);
        hardware.delay_ms(body::POST_BRC_SETTLE_MS);
        push_event(events, BrainstemEvent::CreateBrcPulsed);
    }
}

fn push_event<const N: usize>(events: &mut Deque<BrainstemEvent, N>, event: BrainstemEvent) {
    status::signal_event(&event);
    let _ = events.push_back(event);
}
