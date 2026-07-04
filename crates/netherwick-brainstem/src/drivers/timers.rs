use heapless::Deque;

use crate::events::BrainstemEvent;
use crate::hardware::BrainstemHardware;

pub struct Timers {
    last_tick_ms: u32,
}

impl Timers {
    pub const fn new() -> Self {
        Self { last_tick_ms: 0 }
    }

    pub fn poll<H, const N: usize>(
        &mut self,
        hardware: &mut H,
        events: &mut Deque<BrainstemEvent, N>,
    ) where
        H: BrainstemHardware,
    {
        let now_ms = hardware.now_us() / 1_000;
        if now_ms != self.last_tick_ms {
            self.last_tick_ms = now_ms;
            let _ = events.push_back(BrainstemEvent::TickMs(now_ms));
        }
    }
}
