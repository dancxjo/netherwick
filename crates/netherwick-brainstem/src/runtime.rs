use heapless::Deque;

use crate::body;
use crate::commands::{BrainstemCommand, RuntimeCommand, ARM_SCRIPT, DEMO_SCRIPT, DISARM_SCRIPT};
use crate::drivers::{create_uart::CreateUart, leds::Leds, timers::Timers};
use crate::events::{BrainstemError, BrainstemEvent};
use crate::hardware::BrainstemHardware;
use crate::status::{self, DemoState, RuntimeActionCode, RuntimeState};

const EVENT_QUEUE_CAPACITY: usize = 16;
const COMMAND_QUEUE_CAPACITY: usize = 16;
const RUNTIME_TICK_MS: u32 = 10;
const SENSOR_PROBE_PERIOD_MS: u32 = 100;
const WAKE_PROBE_RESPONSE_BYTES_REQUIRED: u8 = 2;
const CREATE_AXLE_TRACK_MM: i32 = 258;

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
        power_toggled: bool,
    },
    WaitForCreate {
        deadline_ms: u32,
        next_probe_ms: u32,
        response_bytes: u8,
        oi_started: bool,
        power_toggled: bool,
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
    commands: Deque<RuntimeCommand, COMMAND_QUEUE_CAPACITY>,
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
    estop_latched: bool,
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
            estop_latched: false,
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
    pub fn enqueue_command(&mut self, command: RuntimeCommand) -> Result<(), RuntimeCommand> {
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
        self.poll_control_command();
    }

    fn poll_control_command(&mut self) {
        let Some(command) = status::take_control_command() else {
            return;
        };

        match command {
            BrainstemCommand::Stop | BrainstemCommand::EStop => {
                self.commands.clear();
                self.active = ActiveAction::None;
                let command = match command {
                    BrainstemCommand::Stop => RuntimeCommand::Stop,
                    BrainstemCommand::EStop => RuntimeCommand::EStop,
                    _ => unreachable!(),
                };
                let _ = self.commands.push_front(command);
                self.mode = RuntimeMode::Running;
            }
            BrainstemCommand::Arm => {
                for command in ARM_SCRIPT {
                    let _ = self.commands.push_back(*command);
                }
                if self.mode == RuntimeMode::Idle || self.mode == RuntimeMode::Error {
                    self.mode = RuntimeMode::Running;
                    status::set_runtime_state(RuntimeState::RunningDemo);
                }
            }
            BrainstemCommand::Disarm => {
                self.commands.clear();
                self.active = ActiveAction::None;
                for command in DISARM_SCRIPT.iter().rev() {
                    let _ = self.commands.push_front(*command);
                }
                self.mode = RuntimeMode::Running;
            }
            BrainstemCommand::CmdVel { .. } => {
                if let Some(command) = runtime_command_from_forebrain(command) {
                    self.enqueue_latest_velocity(command);
                }
                if self.mode == RuntimeMode::Idle || self.mode == RuntimeMode::Error {
                    self.mode = RuntimeMode::Running;
                    status::set_runtime_state(RuntimeState::RunningDemo);
                }
            }
            _ => {
                if let Some(command) = runtime_command_from_forebrain(command) {
                    let _ = self.commands.push_back(command);
                }
                if self.mode == RuntimeMode::Idle || self.mode == RuntimeMode::Error {
                    self.mode = RuntimeMode::Running;
                    status::set_runtime_state(RuntimeState::RunningDemo);
                }
            }
        }
    }

    fn enqueue_latest_velocity(&mut self, command: RuntimeCommand) {
        let pending = self.commands.len();
        for _ in 0..pending {
            let Some(existing) = self.commands.pop_front() else {
                break;
            };
            if !matches!(existing, RuntimeCommand::CmdVel { .. }) {
                let _ = self.commands.push_back(existing);
            }
        }

        if matches!(self.active, ActiveAction::Driving { .. }) {
            self.active = ActiveAction::None;
            let _ = self.commands.push_front(command);
        } else {
            let _ = self.commands.push_back(command);
        }
    }

    fn start_next_command(&mut self) -> Result<(), BrainstemError> {
        let Some(command) = self.commands.pop_front() else {
            status::set_command(None);
            return Ok(());
        };
        status::set_command(Some(command));

        let now_ms = self.now_ms();
        match command {
            RuntimeCommand::Stop | RuntimeCommand::StopDrive => {
                self.stop_drive()?;
                self.active = ActiveAction::None;
            }
            RuntimeCommand::EStop => {
                self.stop_drive()?;
                self.estop_latched = true;
                self.active = ActiveAction::None;
            }
            RuntimeCommand::ClearEStop => {
                self.estop_latched = false;
            }
            RuntimeCommand::WakeCreate => {
                self.create_responsive = false;
                status::set_create_power_unknown();
                status::set_oi_mode_unknown();
                status::set_demo_state(DemoState::WaitingForCreate);
                self.active = ActiveAction::WaitForCreate {
                    deadline_ms: now_ms.wrapping_add(body::CREATE_RESPONSIVE_TIMEOUT_MS),
                    next_probe_ms: now_ms,
                    response_bytes: 0,
                    oi_started: false,
                    power_toggled: false,
                };
            }
            RuntimeCommand::SleepCreate => {
                self.create_responsive = false;
                status::set_oi_mode_unknown();
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
            RuntimeCommand::PulseBrc => {
                if body::CREATE_BRC_ENABLED {
                    self.push_event(BrainstemEvent::CreateBrcPulseRequested);
                    self.hardware.set_brc(false);
                    self.active = ActiveAction::BrcLow {
                        release_at_ms: now_ms.wrapping_add(body::BRC_LOW_PULSE_MS),
                    };
                }
            }
            RuntimeCommand::StartOi => {
                self.ensure_create_responsive()?;
                self.create_uart
                    .start_oi(&mut self.hardware, &mut self.events)?;
                status::set_demo_state(DemoState::OiStarted);
                self.active = ActiveAction::Settle {
                    until_ms: now_ms.wrapping_add(body::POST_START_SETTLE_MS),
                };
            }
            RuntimeCommand::SetMode(mode) => {
                self.ensure_create_responsive()?;
                self.create_uart
                    .set_mode(&mut self.hardware, &mut self.events, mode)?;
                status::set_oi_mode(mode);
                self.active = ActiveAction::Settle {
                    until_ms: now_ms.wrapping_add(body::POST_MODE_SETTLE_MS),
                };
            }
            RuntimeCommand::Drive {
                left_mm_s,
                right_mm_s,
                duration_ms,
            } => self.start_drive_direct(left_mm_s, right_mm_s, Some(duration_ms), now_ms)?,
            RuntimeCommand::DriveDirect {
                left_mm_s,
                right_mm_s,
                duration_ms,
            } => self.start_drive_direct(left_mm_s, right_mm_s, duration_ms, now_ms)?,
            RuntimeCommand::CmdVel {
                linear_mm_s,
                angular_mrad_s,
                duration_ms,
            } => {
                let half_delta = angular_mrad_s as i32 * CREATE_AXLE_TRACK_MM / 2_000;
                let left = clamp_i16(linear_mm_s as i32 - half_delta);
                let right = clamp_i16(linear_mm_s as i32 + half_delta);
                self.start_drive_direct(left, right, duration_ms, now_ms)?;
            }
            RuntimeCommand::DriveArc {
                velocity_mm_s,
                radius_mm,
                duration_ms,
            } => self.start_drive_arc(velocity_mm_s, radius_mm, duration_ms, now_ms)?,
            RuntimeCommand::SongPlay { id } => {
                self.ensure_create_responsive()?;
                self.create_uart
                    .play_song(&mut self.hardware, &mut self.events, id)?;
            }
            RuntimeCommand::Dock => {
                self.ensure_create_responsive()?;
                self.create_uart
                    .seek_dock(&mut self.hardware, &mut self.events)?;
            }
            RuntimeCommand::SetLights { pattern } => {
                self.create_uart
                    .set_lights(&mut self.hardware, &mut self.events, pattern)?;
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
                    if wake_wait_until_ms.is_none() {
                        status::set_create_power_on(power_on);
                    }
                    self.active = match wake_wait_until_ms {
                        Some(until_ms) => ActiveAction::WakeSettle {
                            until_ms,
                            power_toggled: true,
                        },
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
            ActiveAction::WakeSettle {
                until_ms,
                power_toggled,
            } => {
                if time_reached(now_ms, until_ms) {
                    self.active = ActiveAction::WaitForCreate {
                        deadline_ms: now_ms.wrapping_add(body::CREATE_RESPONSIVE_TIMEOUT_MS),
                        next_probe_ms: now_ms,
                        response_bytes: 0,
                        oi_started: false,
                        power_toggled,
                    };
                }
                Ok(())
            }
            ActiveAction::WaitForCreate {
                deadline_ms,
                next_probe_ms,
                mut response_bytes,
                oi_started,
                power_toggled,
            } => {
                while let Some(event) = self.events.pop_front() {
                    match event {
                        BrainstemEvent::CreatePacketReceived { bytes, .. } => {
                            response_bytes = response_bytes.saturating_add(bytes.len() as u8);
                        }
                        BrainstemEvent::Error(error) => return Err(error),
                        _ => {}
                    }
                }
                status::set_wake_probe_progress(
                    response_bytes as u32,
                    WAKE_PROBE_RESPONSE_BYTES_REQUIRED as u32,
                );

                if response_bytes >= WAKE_PROBE_RESPONSE_BYTES_REQUIRED {
                    self.create_responsive = true;
                    status::set_create_power_on(true);
                    self.active = ActiveAction::None;
                    return Ok(());
                }

                if time_reached(now_ms, deadline_ms) {
                    if !power_toggled {
                        self.push_event(BrainstemEvent::CreatePowerOnRequested);
                        self.hardware.set_power_toggle(true);
                        self.active = ActiveAction::PowerPulse {
                            release_at_ms: now_ms.wrapping_add(body::POWER_TOGGLE_PULSE_MS),
                            wake_wait_until_ms: Some(
                                now_ms.wrapping_add(body::CREATE_WAKE_WAIT_MS),
                            ),
                            power_on: true,
                        };
                        return Ok(());
                    }
                    self.stop_drive()?;
                    self.create_responsive = false;
                    status::set_create_power_unknown();
                    status::set_oi_mode_unknown();
                    status::mark_uart_rx_error();
                    return Err(BrainstemError::CreateNoResponse);
                }

                if time_reached(now_ms, next_probe_ms) {
                    if !oi_started {
                        self.create_uart.flush_rx(&mut self.hardware);
                        self.create_uart
                            .start_oi(&mut self.hardware, &mut self.events)?;
                        self.active = ActiveAction::WaitForCreate {
                            deadline_ms,
                            next_probe_ms: now_ms.wrapping_add(body::POST_START_SETTLE_MS),
                            response_bytes: 0,
                            oi_started: true,
                            power_toggled,
                        };
                        return Ok(());
                    }
                    self.create_uart.flush_rx(&mut self.hardware);
                    response_bytes = 0;
                    status::set_wake_probe_progress(
                        response_bytes as u32,
                        WAKE_PROBE_RESPONSE_BYTES_REQUIRED as u32,
                    );
                    self.create_uart.request_sensor_packet(
                        &mut self.hardware,
                        body::CREATE_SENSOR_PROBE_PACKET,
                    )?;
                    self.active = ActiveAction::WaitForCreate {
                        deadline_ms,
                        next_probe_ms: now_ms.wrapping_add(SENSOR_PROBE_PERIOD_MS),
                        response_bytes,
                        oi_started,
                        power_toggled,
                    };
                } else {
                    self.active = ActiveAction::WaitForCreate {
                        deadline_ms,
                        next_probe_ms,
                        response_bytes,
                        oi_started,
                        power_toggled,
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

    fn start_drive_direct(
        &mut self,
        left_mm_s: i16,
        right_mm_s: i16,
        duration_ms: Option<u32>,
        now_ms: u32,
    ) -> Result<(), BrainstemError> {
        let Some(duration_ms) = duration_ms else {
            self.stop_drive()?;
            return Err(BrainstemError::Timeout);
        };
        self.ensure_motion_allowed()?;

        status::set_demo_state(DemoState::Moving);
        self.stop_sent = false;
        self.create_uart.drive_direct(
            &mut self.hardware,
            &mut self.events,
            left_mm_s,
            right_mm_s,
            duration_ms,
        )?;
        self.active = ActiveAction::Driving {
            stop_at_ms: now_ms.wrapping_add(duration_ms),
        };
        Ok(())
    }

    fn start_drive_arc(
        &mut self,
        velocity_mm_s: i16,
        radius_mm: i16,
        duration_ms: Option<u32>,
        now_ms: u32,
    ) -> Result<(), BrainstemError> {
        let Some(duration_ms) = duration_ms else {
            self.stop_drive()?;
            return Err(BrainstemError::Timeout);
        };
        self.ensure_motion_allowed()?;

        status::set_demo_state(DemoState::Moving);
        self.stop_sent = false;
        self.create_uart.drive_arc(
            &mut self.hardware,
            &mut self.events,
            velocity_mm_s,
            radius_mm,
        )?;
        self.active = ActiveAction::Driving {
            stop_at_ms: now_ms.wrapping_add(duration_ms),
        };
        Ok(())
    }

    fn ensure_create_responsive(&mut self) -> Result<(), BrainstemError> {
        if !self.create_responsive {
            return Err(BrainstemError::CreateNoResponse);
        }
        Ok(())
    }

    fn ensure_motion_allowed(&mut self) -> Result<(), BrainstemError> {
        if self.estop_latched {
            self.stop_drive()?;
            return Err(BrainstemError::CreateNoResponse);
        }
        self.ensure_create_responsive()?;
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

fn runtime_command_from_forebrain(command: BrainstemCommand) -> Option<RuntimeCommand> {
    match command {
        BrainstemCommand::Ping
        | BrainstemCommand::Status
        | BrainstemCommand::Bootsel
        | BrainstemCommand::Arm
        | BrainstemCommand::Disarm => None,
        BrainstemCommand::Stop => Some(RuntimeCommand::Stop),
        BrainstemCommand::EStop => Some(RuntimeCommand::EStop),
        BrainstemCommand::ClearEStop => Some(RuntimeCommand::ClearEStop),
        BrainstemCommand::SetMode(mode) => Some(RuntimeCommand::SetMode(mode)),
        BrainstemCommand::CmdVel {
            linear_mm_s,
            angular_mrad_s,
            ttl_ms,
            ..
        } => Some(RuntimeCommand::CmdVel {
            linear_mm_s,
            angular_mrad_s,
            duration_ms: Some(ttl_ms),
        }),
        BrainstemCommand::SongPlay { id } => Some(RuntimeCommand::SongPlay { id }),
        BrainstemCommand::Dock => Some(RuntimeCommand::Dock),
        BrainstemCommand::SetLights { pattern } => Some(RuntimeCommand::SetLights { pattern }),
    }
}

fn clamp_i16(value: i32) -> i16 {
    value.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}
