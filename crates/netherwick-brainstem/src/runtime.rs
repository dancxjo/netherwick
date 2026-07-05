use heapless::Deque;

use crate::body;
use crate::commands::{BrainstemCommand, DEMO_SCRIPT};
use crate::drivers::{create_uart::CreateUart, leds::Leds, timers::Timers};
use crate::events::{BrainstemError, BrainstemEvent};
use crate::hardware::BrainstemHardware;
use crate::status::{self, DemoState, RuntimeActionCode, RuntimeState};

const EVENT_QUEUE_CAPACITY: usize = 16;
const COMMAND_QUEUE_CAPACITY: usize = 16;
const RUNTIME_TICK_MS: u32 = 10;
const SENSOR_PROBE_PERIOD_MS: u32 = 100;
const RESPONSIVE_BYTES_REQUIRED: u8 = 1;

#[derive(Clone, Copy, Eq, PartialEq)]
enum RuntimeMode {
    Running,
    Idle,
    Error,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ActiveAction {
    None,
    PowerPulse {
        release_at_ms: u32,
        wake_wait_until_ms: Option<u32>,
        power_on: bool,
    },
    BrcLow {
        release_at_ms: u32,
    },
    BrcSettle {
        until_ms: u32,
    },
    WakeSettle {
        until_ms: u32,
    },
    WaitForCreate {
        deadline_ms: u32,
        next_probe_ms: u32,
        bytes_seen: u8,
    },
    Settle {
        until_ms: u32,
    },
    Driving {
        stop_at_ms: u32,
    },
}

pub struct Runtime<H>
where
    H: BrainstemHardware,
{
    hardware: H,
    events: Deque<BrainstemEvent, EVENT_QUEUE_CAPACITY>,
    commands: Deque<BrainstemCommand, COMMAND_QUEUE_CAPACITY>,
    timers: Timers,
    create_uart: CreateUart,
    leds: Leds,
    mode: RuntimeMode,
    active: ActiveAction,
    stop_sent: bool,
    error_blink_next_ms: u32,
    error_blink_on: bool,
    error_blink_count: u8,
    idle_blink_next_ms: u32,
    idle_blink_on: bool,
    create_responsive: bool,
}

impl<H> Runtime<H>
where
    H: BrainstemHardware,
{
    pub fn new(hardware: H) -> Self {
        let mut events = Deque::new();
        status::signal_event(&BrainstemEvent::Boot);
        let _ = events.push_back(BrainstemEvent::Boot);
        status::set_runtime_state(RuntimeState::Booting);
        status::set_demo_state(DemoState::NotStarted);
        Self {
            hardware,
            events,
            commands: Deque::new(),
            timers: Timers::new(),
            create_uart: CreateUart::new(),
            leds: Leds::new(),
            mode: RuntimeMode::Running,
            active: ActiveAction::None,
            stop_sent: false,
            error_blink_next_ms: 0,
            error_blink_on: false,
            error_blink_count: 0,
            idle_blink_next_ms: 0,
            idle_blink_on: false,
            create_responsive: false,
        }
    }

    pub fn run_demo(mut self) -> ! {
        self.start_demo();
        loop {
            self.tick();
            self.hardware.delay_ms(RUNTIME_TICK_MS);
        }
    }

    pub fn start_demo(&mut self) {
        status::set_runtime_state(RuntimeState::RunningDemo);
        self.leds.boot_indicator(&mut self.hardware);
        for command in DEMO_SCRIPT {
            let _ = self.commands.push_back(*command);
        }
    }

    #[allow(dead_code)]
    pub fn enqueue_command(&mut self, command: BrainstemCommand) -> Result<(), BrainstemCommand> {
        self.commands.push_back(command)
    }

    pub fn tick(&mut self) {
        status::set_runtime_action(self.active_action_code());
        self.poll();
        self.feed_watchdog_placeholder();

        match self.mode {
            RuntimeMode::Running => {
                if let Err(error) = self.advance_active_action() {
                    self.enter_error(error);
                    return;
                }

                if self.active == ActiveAction::None {
                    if let Err(error) = self.start_next_command() {
                        self.enter_error(error);
                    } else if self.commands.is_empty() && self.active == ActiveAction::None {
                        self.enter_idle();
                    }
                }
            }
            RuntimeMode::Idle => self.idle_tick(),
            RuntimeMode::Error => self.error_tick(),
        }
    }

    fn poll(&mut self) {
        self.timers.poll(&mut self.hardware, &mut self.events);
        self.create_uart.poll(&mut self.hardware, &mut self.events);
    }

    fn start_next_command(&mut self) -> Result<(), BrainstemError> {
        let Some(command) = self.commands.pop_front() else {
            status::set_command(None);
            return Ok(());
        };
        status::set_command(Some(command));

        let now_ms = self.now_ms();
        match command {
            BrainstemCommand::WakeCreate => {
                status::set_demo_state(DemoState::WaitingForCreate);
                self.push_event(BrainstemEvent::CreatePowerOnRequested);
                self.hardware.set_power_toggle(true);
                self.active = ActiveAction::PowerPulse {
                    release_at_ms: now_ms.wrapping_add(body::POWER_TOGGLE_PULSE_MS),
                    wake_wait_until_ms: Some(now_ms.wrapping_add(body::CREATE_WAKE_WAIT_MS)),
                    power_on: true,
                };
            }
            BrainstemCommand::SleepCreate => {
                status::set_demo_state(DemoState::PowerCycling);
                self.stop_drive()?;
                self.push_event(BrainstemEvent::CreatePowerOffRequested);
                self.hardware.set_power_toggle(true);
                self.active = ActiveAction::PowerPulse {
                    release_at_ms: now_ms.wrapping_add(body::POWER_TOGGLE_PULSE_MS),
                    wake_wait_until_ms: None,
                    power_on: false,
                };
            }
            BrainstemCommand::PulseBrc => {
                if body::CREATE_BRC_ENABLED {
                    self.push_event(BrainstemEvent::CreateBrcPulseRequested);
                    self.hardware.set_brc(false);
                    self.active = ActiveAction::BrcLow {
                        release_at_ms: now_ms.wrapping_add(body::BRC_LOW_PULSE_MS),
                    };
                }
            }
            BrainstemCommand::StartOi => {
                self.create_uart
                    .start_oi(&mut self.hardware, &mut self.events)?;
                status::set_demo_state(DemoState::OiStarted);
                self.active = ActiveAction::Settle {
                    until_ms: now_ms.wrapping_add(body::POST_MODE_SETTLE_MS),
                };
            }
            BrainstemCommand::SetOiMode(mode) => {
                self.create_uart
                    .set_mode(&mut self.hardware, &mut self.events, mode)?;
                status::set_oi_mode(mode);
                self.active = ActiveAction::Settle {
                    until_ms: now_ms.wrapping_add(body::POST_MODE_SETTLE_MS),
                };
            }
            BrainstemCommand::Drive {
                left_mm_s,
                right_mm_s,
                duration_ms,
            } => {
                if !self.create_responsive {
                    self.stop_drive()?;
                    return Err(BrainstemError::CreateNoResponse);
                }
                status::set_demo_state(DemoState::Moving);
                self.stop_sent = false;
                self.create_uart.drive_direct_start(
                    &mut self.hardware,
                    &mut self.events,
                    left_mm_s,
                    right_mm_s,
                    duration_ms,
                )?;
                self.active = ActiveAction::Driving {
                    stop_at_ms: now_ms.wrapping_add(duration_ms),
                };
            }
            BrainstemCommand::StopDrive => {
                self.stop_drive()?;
            }
        }

        Ok(())
    }

    fn advance_active_action(&mut self) -> Result<(), BrainstemError> {
        let now_ms = self.now_ms();
        match self.active {
            ActiveAction::None => Ok(()),
            ActiveAction::PowerPulse {
                release_at_ms,
                wake_wait_until_ms,
                power_on,
            } => {
                if time_reached(now_ms, release_at_ms) {
                    self.hardware.set_power_toggle(false);
                    self.push_event(BrainstemEvent::CreatePowerToggled);
                    status::set_create_power_on(power_on);
                    self.active = match wake_wait_until_ms {
                        Some(until_ms) => ActiveAction::WakeSettle { until_ms },
                        None => ActiveAction::None,
                    };
                }
                Ok(())
            }
            ActiveAction::BrcLow { release_at_ms } => {
                if time_reached(now_ms, release_at_ms) {
                    self.hardware.set_brc(true);
                    self.push_event(BrainstemEvent::CreateBrcPulsed);
                    self.active = ActiveAction::BrcSettle {
                        until_ms: now_ms.wrapping_add(body::POST_BRC_SETTLE_MS),
                    };
                }
                Ok(())
            }
            ActiveAction::BrcSettle { until_ms } | ActiveAction::Settle { until_ms } => {
                if time_reached(now_ms, until_ms) {
                    self.active = ActiveAction::None;
                }
                Ok(())
            }
            ActiveAction::WakeSettle { until_ms } => {
                if time_reached(now_ms, until_ms) {
                    self.active = ActiveAction::WaitForCreate {
                        deadline_ms: now_ms.wrapping_add(body::CREATE_RESPONSIVE_TIMEOUT_MS),
                        next_probe_ms: now_ms,
                        bytes_seen: 0,
                    };
                }
                Ok(())
            }
            ActiveAction::WaitForCreate {
                deadline_ms,
                next_probe_ms,
                mut bytes_seen,
            } => {
                while let Some(event) = self.events.pop_front() {
                    match event {
                        BrainstemEvent::CreatePacketReceived { .. } => {
                            bytes_seen = bytes_seen.saturating_add(1);
                        }
                        BrainstemEvent::Error(error) => return Err(error),
                        _ => {}
                    }
                }

                if bytes_seen >= RESPONSIVE_BYTES_REQUIRED {
                    self.create_responsive = true;
                    status::set_create_power_on(true);
                    self.active = ActiveAction::None;
                    return Ok(());
                }

                if time_reached(now_ms, deadline_ms) {
                    self.stop_drive()?;
                    self.create_responsive = false;
                    status::mark_uart_rx_error();
                    return Err(BrainstemError::CreateNoResponse);
                }

                if time_reached(now_ms, next_probe_ms) {
                    self.create_uart.request_sensor_packet(
                        &mut self.hardware,
                        body::CREATE_SENSOR_PROBE_PACKET,
                    )?;
                    self.active = ActiveAction::WaitForCreate {
                        deadline_ms,
                        next_probe_ms: now_ms.wrapping_add(SENSOR_PROBE_PERIOD_MS),
                        bytes_seen,
                    };
                } else {
                    self.active = ActiveAction::WaitForCreate {
                        deadline_ms,
                        next_probe_ms,
                        bytes_seen,
                    };
                }
                Ok(())
            }
            ActiveAction::Driving { stop_at_ms } => {
                if time_reached(now_ms, stop_at_ms) {
                    self.stop_drive()?;
                    self.active = ActiveAction::None;
                }
                Ok(())
            }
        }
    }

    fn stop_drive(&mut self) -> Result<(), BrainstemError> {
        self.create_uart
            .stop(&mut self.hardware, &mut self.events)?;
        self.stop_sent = true;
        Ok(())
    }

    fn enter_idle(&mut self) {
        let _ = self.stop_drive();
        self.mode = RuntimeMode::Idle;
        self.active = ActiveAction::None;
        status::set_runtime_state(RuntimeState::Idle);
        status::set_demo_state(DemoState::Idle);
        status::set_command(None);
        self.hardware.set_indicators(false);
        self.idle_blink_next_ms = self.now_ms();
    }

    fn idle_tick(&mut self) {
        let now_ms = self.now_ms();
        if time_reached(now_ms, self.idle_blink_next_ms) {
            self.idle_blink_on = !self.idle_blink_on;
            self.hardware.set_indicators(self.idle_blink_on);
            self.idle_blink_next_ms = now_ms.wrapping_add(body::IDLE_BLINK_MS);
        }
    }

    fn enter_error(&mut self, error: BrainstemError) {
        status::set_error(error);
        self.push_event(BrainstemEvent::Error(error));
        let _ = self.stop_drive();
        self.mode = RuntimeMode::Error;
        self.active = ActiveAction::None;
        self.error_blink_next_ms = self.now_ms();
        self.error_blink_count = 0;
        self.error_blink_on = false;
    }

    fn error_tick(&mut self) {
        let now_ms = self.now_ms();
        if !time_reached(now_ms, self.error_blink_next_ms) {
            return;
        }

        if self.error_blink_count >= 6 {
            self.hardware.set_indicators(false);
            self.error_blink_on = false;
            self.error_blink_count = 0;
            self.error_blink_next_ms = now_ms.wrapping_add(body::ERROR_PAUSE_MS);
            return;
        }

        self.error_blink_on = !self.error_blink_on;
        self.hardware.set_indicators(self.error_blink_on);
        self.error_blink_count = self.error_blink_count.saturating_add(1);
        self.error_blink_next_ms = now_ms.wrapping_add(body::ERROR_BLINK_MS);
    }

    fn now_ms(&mut self) -> u32 {
        self.hardware.now_us() / 1_000
    }

    fn active_action_code(&self) -> RuntimeActionCode {
        match self.active {
            ActiveAction::None => RuntimeActionCode::None,
            ActiveAction::PowerPulse { .. } => RuntimeActionCode::PowerPulse,
            ActiveAction::BrcLow { .. } => RuntimeActionCode::BrcLow,
            ActiveAction::BrcSettle { .. } => RuntimeActionCode::BrcSettle,
            ActiveAction::WakeSettle { .. } => RuntimeActionCode::WakeSettle,
            ActiveAction::WaitForCreate { .. } => RuntimeActionCode::WaitForCreate,
            ActiveAction::Settle { .. } => RuntimeActionCode::Settle,
            ActiveAction::Driving { .. } => RuntimeActionCode::Driving,
        }
    }

    fn push_event(&mut self, event: BrainstemEvent) {
        status::signal_event(&event);
        let _ = self.events.push_back(event);
    }

    fn feed_watchdog_placeholder(&mut self) {
        // Hardware watchdog feeding belongs here once a backend exposes it.
        // It must remain owned by this safety/runtime lane, not Wi-Fi.
    }
}

fn time_reached(now_ms: u32, deadline_ms: u32) -> bool {
    now_ms.wrapping_sub(deadline_ms) < u32::MAX / 2
}
