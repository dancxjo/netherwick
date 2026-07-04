use heapless::Deque;

use crate::commands::{BrainstemCommand, DEMO_SCRIPT};
use crate::drivers::{
    create_power::CreatePower, create_uart::CreateUart, leds::Leds, timers::Timers,
};
use crate::events::{BrainstemError, BrainstemEvent};
use crate::hardware::BrainstemHardware;

const EVENT_QUEUE_CAPACITY: usize = 16;
const COMMAND_QUEUE_CAPACITY: usize = 16;

pub struct Runtime<H>
where
    H: BrainstemHardware,
{
    hardware: H,
    events: Deque<BrainstemEvent, EVENT_QUEUE_CAPACITY>,
    commands: Deque<BrainstemCommand, COMMAND_QUEUE_CAPACITY>,
    timers: Timers,
    power: CreatePower,
    create_uart: CreateUart,
    leds: Leds,
}

impl<H> Runtime<H>
where
    H: BrainstemHardware,
{
    pub fn new(hardware: H) -> Self {
        let mut events = Deque::new();
        let _ = events.push_back(BrainstemEvent::Boot);
        Self {
            hardware,
            events,
            commands: Deque::new(),
            timers: Timers::new(),
            power: CreatePower::new(),
            create_uart: CreateUart::new(),
            leds: Leds::new(),
        }
    }

    pub fn run_demo(mut self) -> ! {
        self.leds.boot_indicator(&mut self.hardware);
        for command in DEMO_SCRIPT {
            let _ = self.commands.push_back(*command);
        }

        loop {
            self.poll();
            if self.consume_next_command().is_err() {
                self.enter_error(BrainstemError::CreateNoResponse);
            }
            if self.commands.is_empty() {
                self.idle();
            }
        }
    }

    fn poll(&mut self) {
        self.timers.poll(&mut self.hardware, &mut self.events);
        self.create_uart.poll(&mut self.hardware, &mut self.events);
    }

    fn consume_next_command(&mut self) -> Result<(), BrainstemError> {
        let Some(command) = self.commands.pop_front() else {
            return Ok(());
        };

        match command {
            BrainstemCommand::WakeCreate => {
                self.power.wake(&mut self.hardware, &mut self.events);
                self.create_uart
                    .wait_for_response(&mut self.hardware, &mut self.events)
            }
            BrainstemCommand::SleepCreate => {
                self.power.sleep(&mut self.hardware, &mut self.events);
                Ok(())
            }
            BrainstemCommand::PulseBrc => {
                self.power.pulse_brc(&mut self.hardware, &mut self.events);
                Ok(())
            }
            BrainstemCommand::StartOi => self
                .create_uart
                .start_oi(&mut self.hardware, &mut self.events),
            BrainstemCommand::SetOiMode(mode) => {
                self.create_uart
                    .set_mode(&mut self.hardware, &mut self.events, mode)
            }
            BrainstemCommand::Drive {
                left_mm_s,
                right_mm_s,
                duration_ms,
            } => self.create_uart.drive_direct(
                &mut self.hardware,
                &mut self.events,
                left_mm_s,
                right_mm_s,
                duration_ms,
            ),
            BrainstemCommand::StopDrive => {
                self.create_uart.stop(&mut self.hardware, &mut self.events)
            }
        }
    }

    fn idle(&mut self) -> ! {
        self.hardware.set_indicators(false);
        loop {
            self.poll();
            if let Some(command) = self.commands.pop_front() {
                let _ = self.commands.push_front(command);
                if self.consume_next_command().is_err() {
                    self.enter_error(BrainstemError::CreateNoResponse);
                }
            }
            self.leds.idle_once(&mut self.hardware);
        }
    }

    fn enter_error(&mut self, error: BrainstemError) -> ! {
        let _ = self.events.push_back(BrainstemEvent::Error(error));
        let _ = self.create_uart.stop(&mut self.hardware, &mut self.events);
        loop {
            self.leds.error_once(&mut self.hardware);
        }
    }
}
