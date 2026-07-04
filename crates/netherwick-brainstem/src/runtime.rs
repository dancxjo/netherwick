use heapless::Deque;

use crate::commands::{BrainstemCommand, DEMO_SCRIPT};
use crate::drivers::{
    create_power::CreatePower, create_uart::CreateUart, leds::Leds, timers::Timers,
};
use crate::events::{BrainstemError, BrainstemEvent};
use crate::hardware::BrainstemHardware;
use crate::status::{self, DemoState, RuntimeState};

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
        status::set_runtime_state(RuntimeState::Booting);
        status::set_demo_state(DemoState::NotStarted);
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
        status::set_runtime_state(RuntimeState::RunningDemo);
        self.leds.boot_indicator(&mut self.hardware);
        for command in DEMO_SCRIPT {
            let _ = self.commands.push_back(*command);
        }

        loop {
            self.poll();
            if let Err(error) = self.consume_next_command() {
                self.enter_error(error);
            }
            if self.commands.is_empty() {
                let _ = self.create_uart.stop(&mut self.hardware, &mut self.events);
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
            status::set_command(None);
            return Ok(());
        };
        status::set_command(Some(command));

        match command {
            BrainstemCommand::WakeCreate => {
                status::set_demo_state(DemoState::WaitingForCreate);
                self.power.wake(&mut self.hardware, &mut self.events);
                let result = self
                    .create_uart
                    .wait_for_response(&mut self.hardware, &mut self.events);
                if result.is_ok() {
                    status::set_create_power_on(true);
                }
                result
            }
            BrainstemCommand::SleepCreate => {
                status::set_demo_state(DemoState::PowerCycling);
                self.create_uart
                    .stop(&mut self.hardware, &mut self.events)?;
                self.power.sleep(&mut self.hardware, &mut self.events);
                status::set_create_power_on(false);
                Ok(())
            }
            BrainstemCommand::PulseBrc => {
                self.power.pulse_brc(&mut self.hardware, &mut self.events);
                Ok(())
            }
            BrainstemCommand::StartOi => {
                let result = self
                    .create_uart
                    .start_oi(&mut self.hardware, &mut self.events);
                if result.is_ok() {
                    status::set_demo_state(DemoState::OiStarted);
                }
                result
            }
            BrainstemCommand::SetOiMode(mode) => {
                let result = self
                    .create_uart
                    .set_mode(&mut self.hardware, &mut self.events, mode);
                if result.is_ok() {
                    status::set_oi_mode(mode);
                }
                result
            }
            BrainstemCommand::Drive {
                left_mm_s,
                right_mm_s,
                duration_ms,
            } => {
                status::set_demo_state(DemoState::Moving);
                self.create_uart.drive_direct(
                    &mut self.hardware,
                    &mut self.events,
                    left_mm_s,
                    right_mm_s,
                    duration_ms,
                )
            }
            BrainstemCommand::StopDrive => {
                self.create_uart.stop(&mut self.hardware, &mut self.events)
            }
        }
    }

    fn idle(&mut self) -> ! {
        status::set_runtime_state(RuntimeState::Idle);
        status::set_demo_state(DemoState::Idle);
        status::set_command(None);
        self.hardware.set_indicators(false);
        loop {
            self.poll();
            if let Some(command) = self.commands.pop_front() {
                let _ = self.commands.push_front(command);
                if let Err(error) = self.consume_next_command() {
                    self.enter_error(error);
                }
            }
            self.leds.idle_once(&mut self.hardware);
        }
    }

    fn enter_error(&mut self, error: BrainstemError) -> ! {
        status::set_error(error);
        let _ = self.events.push_back(BrainstemEvent::Error(error));
        let _ = self.create_uart.stop(&mut self.hardware, &mut self.events);
        loop {
            self.leds.error_once(&mut self.hardware);
        }
    }
}
