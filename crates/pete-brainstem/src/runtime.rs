use heapless::Deque;

use crate::body;
use crate::commands::{
    BrainstemCommand, EscapeDirection, FeedbackKind, PowerStateRequest, RuntimeCommand,
    SafetyAction, SafetyPolicy, SongTone, ACQUIRE_CREATE_SCRIPT, DISARM_SCRIPT, MAX_SONG_TONES,
    RESTART_CREATE_SCRIPT,
};
use crate::drivers::{create_uart::CreateUart, leds::Leds, timers::Timers};
use crate::events::{BrainstemError, BrainstemEvent};
use crate::hardware::BrainstemHardware;
use crate::network_registry;
use crate::status::{self, BodyState, RuntimeActionCode, RuntimeState};

const EVENT_QUEUE_CAPACITY: usize = 16;
const COMMAND_QUEUE_CAPACITY: usize = 16;
const RUNTIME_TICK_MS: u32 = 10;
const SENSOR_PROBE_PERIOD_MS: u32 = 100;
const FULL_MODE_REFRESH_PERIOD_MS: u32 = 1_000;
const SUPERVISION_LIGHT_PERIOD_MS: u32 = 180;
const LOW_BATTERY_PERCENT: u32 = 20;
const WAKE_PROBE_RESPONSE_BYTES_REQUIRED: u8 = 1;
const CREATE_AXLE_TRACK_MM: i32 = 258;
const BEARING_SLOWDOWN_MRAD: i32 = 1_000;
const MIN_TRACK_SPEED_MM_S: i32 = 35;
const BUMP_ESCAPE_BACKOFF_MS: u32 = 450;
const BUMP_ESCAPE_TURN_MS: u32 = 650;
const FEEDBACK_SLOT_BASE: u8 = 10;
const FEEDBACK_KIND_COUNT: usize = 6;

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

#[derive(Clone, Copy)]
struct SensorStream {
    packet_id: u8,
    period_ms: u32,
    next_request_ms: u32,
}

#[derive(Clone, Copy)]
struct QueuedCommand {
    command_id: u32,
    command: RuntimeCommand,
}

pub struct Runtime<H>
where
    H: BrainstemHardware,
{
    hardware: H,
    events: Deque<BrainstemEvent, EVENT_QUEUE_CAPACITY>,
    commands: Deque<QueuedCommand, COMMAND_QUEUE_CAPACITY>,
    timers: Timers,
    create_uart: CreateUart,
    leds: Leds,
    mode: RuntimeMode,
    active: ActiveAction,
    active_command_id: Option<u32>,
    stop_sent: bool,
    heartbeat_stop_at_ms: Option<u32>,
    sensor_stream: Option<SensorStream>,
    next_imu_poll_ms: u32,
    next_full_mode_refresh_ms: u32,
    next_supervision_light_ms: u32,
    supervision_light_phase: u8,
    safety_policy: SafetyPolicy,
    safety_latched: bool,
    chirps: [[SongTone; MAX_SONG_TONES]; FEEDBACK_KIND_COUNT],
    chirp_counts: [u8; FEEDBACK_KIND_COUNT],
    error_blink_next_ms: u32,
    error_blink_on: bool,
    error_blink_count: u8,
    idle_blink_next_ms: u32,
    idle_blink_on: bool,
    create_responsive: bool,
    estop_latched: bool,
    last_bump: bool,
    last_cliff: bool,
    last_wheel_drop: bool,
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
        status::set_body_state(BodyState::NotStarted);
        Self {
            hardware,
            events,
            commands: Deque::new(),
            timers: Timers::new(),
            create_uart: CreateUart::new(),
            leds: Leds::new(),
            mode: RuntimeMode::Running,
            active: ActiveAction::None,
            active_command_id: None,
            stop_sent: false,
            heartbeat_stop_at_ms: None,
            sensor_stream: None,
            next_imu_poll_ms: 0,
            next_full_mode_refresh_ms: 0,
            next_supervision_light_ms: 0,
            supervision_light_phase: 0,
            safety_policy: SafetyPolicy::default(),
            safety_latched: false,
            chirps: [[SongTone::default(); MAX_SONG_TONES]; FEEDBACK_KIND_COUNT],
            chirp_counts: [0; FEEDBACK_KIND_COUNT],
            error_blink_next_ms: 0,
            error_blink_on: false,
            error_blink_count: 0,
            idle_blink_next_ms: 0,
            idle_blink_on: false,
            create_responsive: false,
            estop_latched: false,
            last_bump: false,
            last_cliff: false,
            last_wheel_drop: false,
        }
    }

    pub fn run(mut self) -> ! {
        self.start();
        loop {
            self.tick();
            self.hardware.delay_ms(RUNTIME_TICK_MS);
        }
    }

    fn start(&mut self) {
        self.leds.boot_indicator(&mut self.hardware);
        self.queue_create_acquisition(0);
        status::set_runtime_state(RuntimeState::Running);
    }

    #[allow(dead_code)]
    pub fn enqueue_command(&mut self, command: RuntimeCommand) -> Result<(), RuntimeCommand> {
        self.commands
            .push_back(QueuedCommand {
                command_id: 0,
                command,
            })
            .map_err(|queued| queued.command)
    }

    pub fn tick(&mut self) {
        status::set_runtime_action(self.active_action_code());
        self.poll();
        self.hardware.feed_watchdog();
        self.poll_imu();
        if let Err(error) = self.poll_sensor_stream() {
            self.enter_error(error);
            return;
        }
        if let Err(error) = self.enforce_safety_policy() {
            self.enter_error(error);
            return;
        }
        if let Err(error) = self.enforce_heartbeat_stop() {
            self.enter_error(error);
            return;
        }
        if let Err(error) = self.maintain_full_mode() {
            self.enter_error(error);
            return;
        }
        if let Err(error) = self.animate_supervision_lights() {
            self.enter_error(error);
            return;
        }
        if status::take_expired_authority(self.now_ms()) {
            self.interrupt_active_command();
            self.commands.clear();
            self.active = ActiveAction::None;
            self.heartbeat_stop_at_ms = None;
            let _ = self.stop_drive();
        }

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
        self.poll_session_replace();
        self.poll_authority_transition();
        self.timers.poll(&mut self.hardware, &mut self.events);
        self.create_uart.poll(&mut self.hardware, &mut self.events);
        let snapshot = status::snapshot(self.now_ms());
        if matches!(snapshot.oi_mode, 1..=3) {
            self.create_responsive = true;
            status::set_create_power_on(true);
        }
        self.poll_control_command();
    }

    fn poll_authority_transition(&mut self) {
        let Some(generation) = status::pending_authority_transition() else {
            return;
        };
        self.interrupt_active_command();
        self.commands.clear();
        self.active = ActiveAction::None;
        self.heartbeat_stop_at_ms = None;
        let _ = self.stop_drive();
        status::set_command(None);
        status::set_runtime_state(RuntimeState::Idle);
        status::set_body_state(BodyState::Idle);
        status::acknowledge_authority_transition(generation);
        let _ = self.commands.push_back(QueuedCommand {
            command_id: 0,
            command: RuntimeCommand::PlayFeedback {
                kind: FeedbackKind::Ok,
            },
        });
    }

    fn queue_create_acquisition(&mut self, command_id: u32) {
        for command in ACQUIRE_CREATE_SCRIPT {
            let _ = self.commands.push_back(QueuedCommand {
                command_id,
                command: *command,
            });
        }
        self.mode = RuntimeMode::Running;
    }

    fn maintain_full_mode(&mut self) -> Result<(), BrainstemError> {
        let now_ms = self.now_ms();
        let snapshot = status::snapshot(now_ms);
        if !time_reached(now_ms, self.next_full_mode_refresh_ms)
            || low_battery_and_charging(&snapshot)
        {
            return Ok(());
        }

        // RX health is evidence, not permission to transmit. If the Create has
        // rebooted, gone passive, or our wake probe was wrong, START + FULL is
        // the idempotent assertion that lets the brainstem regain control.
        if !self.create_responsive || snapshot.oi_mode != 3 {
            self.create_uart
                .start_oi(&mut self.hardware, &mut self.events)?;
        }
        self.create_uart.set_mode(
            &mut self.hardware,
            &mut self.events,
            crate::commands::CreateOiMode::Full,
        )?;
        if snapshot.oi_mode == 0 {
            self.create_uart.start_mode_stream(&mut self.hardware)?;
        }
        self.next_full_mode_refresh_ms = now_ms.wrapping_add(FULL_MODE_REFRESH_PERIOD_MS);
        Ok(())
    }

    fn animate_supervision_lights(&mut self) -> Result<(), BrainstemError> {
        let now_ms = self.now_ms();
        if !self.create_responsive
            || status::snapshot(now_ms).oi_mode != 3
            || !time_reached(now_ms, self.next_supervision_light_ms)
        {
            return Ok(());
        }

        let (led_bits, color, intensity, period_ms) = if self.mode == RuntimeMode::Error {
            let on = self.supervision_light_phase & 1 == 0;
            (
                if on { 0x03 } else { 0 },
                255,
                if on { 255 } else { 0 },
                300,
            )
        } else if self.estop_latched || self.safety_latched {
            (0x03, 255, 255, 500)
        } else {
            // Three-light bounce: power -> binary 1 -> binary 2 -> binary 1.
            match self.supervision_light_phase & 3 {
                0 => (0, 0, 220, SUPERVISION_LIGHT_PERIOD_MS),
                1 | 3 => (0x01, 0, 0, SUPERVISION_LIGHT_PERIOD_MS),
                _ => (0x02, 0, 0, SUPERVISION_LIGHT_PERIOD_MS),
            }
        };
        self.create_uart
            .set_supervision_lights(&mut self.hardware, led_bits, color, intensity)?;
        self.supervision_light_phase = self.supervision_light_phase.wrapping_add(1);
        self.next_supervision_light_ms = now_ms.wrapping_add(period_ms);
        Ok(())
    }

    fn poll_session_replace(&mut self) {
        let Some(generation) = status::pending_session_replace() else {
            return;
        };
        self.interrupt_active_command();
        self.commands.clear();
        self.active = ActiveAction::None;
        self.heartbeat_stop_at_ms = None;
        self.sensor_stream = None;
        network_registry::clear_motherbrain_registration();
        let _ = self.stop_drive();
        status::set_command(None);
        status::set_runtime_state(RuntimeState::Idle);
        status::set_body_state(BodyState::Idle);
        status::set_session_safety_snapshot(self.estop_latched, self.safety_latched);
        // The session module supplies the pending hash before requesting the
        // barrier. Until it is wired, generation itself is a fail-closed token.
        status::acknowledge_session_replace(generation, status::pending_session_hash());
    }

    fn poll_control_command(&mut self) {
        let Some(command) = status::take_control_command() else {
            return;
        };
        let command_id = status::last_dispatched_command_id();

        match command {
            BrainstemCommand::Stop | BrainstemCommand::EStop => {
                self.interrupt_active_command();
                self.commands.clear();
                self.active = ActiveAction::None;
                self.heartbeat_stop_at_ms = None;
                let command = match command {
                    BrainstemCommand::Stop => RuntimeCommand::Stop,
                    BrainstemCommand::EStop => RuntimeCommand::EStop,
                    _ => unreachable!(),
                };
                let _ = self.commands.push_front(QueuedCommand {
                    command_id,
                    command,
                });
                self.mode = RuntimeMode::Running;
            }
            BrainstemCommand::Arm => {
                self.queue_create_acquisition(command_id);
                if self.mode == RuntimeMode::Idle || self.mode == RuntimeMode::Error {
                    self.mode = RuntimeMode::Running;
                    status::set_runtime_state(RuntimeState::Running);
                }
            }
            BrainstemCommand::Disarm => {
                self.interrupt_active_command();
                self.commands.clear();
                self.active = ActiveAction::None;
                self.heartbeat_stop_at_ms = None;
                for command in DISARM_SCRIPT.iter().rev() {
                    let _ = self.commands.push_front(QueuedCommand {
                        command_id,
                        command: *command,
                    });
                }
                self.mode = RuntimeMode::Running;
            }
            BrainstemCommand::RestartCreate => {
                self.interrupt_active_command();
                self.commands.clear();
                self.active = ActiveAction::None;
                self.heartbeat_stop_at_ms = None;
                for command in RESTART_CREATE_SCRIPT.iter().rev() {
                    let _ = self.commands.push_front(QueuedCommand {
                        command_id,
                        command: *command,
                    });
                }
                self.mode = RuntimeMode::Running;
                status::set_runtime_state(RuntimeState::Running);
            }
            BrainstemCommand::CmdVel { .. } => {
                if let Some(command) = runtime_command_from_forebrain(command) {
                    self.enqueue_latest_velocity(command_id, command);
                }
                if self.mode == RuntimeMode::Idle || self.mode == RuntimeMode::Error {
                    self.mode = RuntimeMode::Running;
                    status::set_runtime_state(RuntimeState::Running);
                }
            }
            _ => {
                if let Some(command) = runtime_command_from_forebrain(command) {
                    let _ = self.commands.push_back(QueuedCommand {
                        command_id,
                        command,
                    });
                }
                if self.mode == RuntimeMode::Idle || self.mode == RuntimeMode::Error {
                    self.mode = RuntimeMode::Running;
                    status::set_runtime_state(RuntimeState::Running);
                }
            }
        }
    }

    fn enqueue_latest_velocity(&mut self, command_id: u32, command: RuntimeCommand) {
        let pending = self.commands.len();
        for _ in 0..pending {
            let Some(existing) = self.commands.pop_front() else {
                break;
            };
            if !matches!(existing.command, RuntimeCommand::CmdVel { .. }) {
                let _ = self.commands.push_back(existing);
            }
        }

        if matches!(self.active, ActiveAction::Driving { .. }) {
            self.interrupt_active_command();
            self.active = ActiveAction::None;
            let _ = self.commands.push_front(QueuedCommand {
                command_id,
                command,
            });
        } else {
            let _ = self.commands.push_back(QueuedCommand {
                command_id,
                command,
            });
        }
    }

    fn start_next_command(&mut self) -> Result<(), BrainstemError> {
        let Some(queued) = self.commands.pop_front() else {
            status::set_command(None);
            return Ok(());
        };
        let command = queued.command;
        let command_code = status::set_command(Some(command));
        self.active_command_id = Some(queued.command_id);
        status::mark_command_started(queued.command_id, command_code);

        let now_ms = self.now_ms();
        match command {
            RuntimeCommand::Stop | RuntimeCommand::StopDrive => {
                self.stop_drive()?;
                self.active = ActiveAction::None;
            }
            RuntimeCommand::EStop => {
                self.stop_drive()?;
                self.estop_latched = true;
                status::mark_estop_latched();
                status::mark_safety_tripped(status::SafetyEventKind::EStop);
                self.active = ActiveAction::None;
            }
            RuntimeCommand::ClearEStop => {
                self.estop_latched = false;
                status::mark_estop_cleared();
                status::mark_safety_cleared(status::SafetyEventKind::EStop);
            }
            RuntimeCommand::WakeCreate => {
                self.create_responsive = false;
                status::set_create_power_unknown();
                status::set_oi_mode_unknown();
                status::set_body_state(BodyState::WaitingForCreate);
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
                status::set_body_state(BodyState::PowerCycling);
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
                self.create_uart
                    .start_oi(&mut self.hardware, &mut self.events)?;
                status::set_body_state(BodyState::OiStarted);
                self.active = ActiveAction::Settle {
                    until_ms: now_ms.wrapping_add(body::POST_START_SETTLE_MS),
                };
            }
            RuntimeCommand::SetCreateBaud(baud) => {
                self.create_uart.flush_rx(&mut self.hardware);
                self.hardware
                    .set_create_uart_baud(baud)
                    .map_err(|_| BrainstemError::UartFraming)?;
                self.create_responsive = false;
                status::set_oi_mode_unknown();
                self.next_full_mode_refresh_ms = now_ms;
            }
            RuntimeCommand::SetMode(mode) => {
                self.create_uart
                    .set_mode(&mut self.hardware, &mut self.events, mode)?;
                if mode == crate::commands::CreateOiMode::Full {
                    self.next_full_mode_refresh_ms =
                        now_ms.wrapping_add(FULL_MODE_REFRESH_PERIOD_MS);
                }
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
                self.start_cmd_vel(linear_mm_s, angular_mrad_s, duration_ms, now_ms)?;
            }
            RuntimeCommand::FaceBearing {
                bearing_mrad,
                max_angular_mrad_s,
                tolerance_mrad,
                duration_ms,
            } => self.start_face_bearing(
                bearing_mrad,
                max_angular_mrad_s,
                tolerance_mrad,
                duration_ms,
                now_ms,
            )?,
            RuntimeCommand::TrackBearing {
                bearing_mrad,
                range_mm,
                max_linear_mm_s,
                max_angular_mrad_s,
                stop_range_mm,
                duration_ms,
            } => self.start_track_bearing(
                bearing_mrad,
                range_mm,
                max_linear_mm_s,
                max_angular_mrad_s,
                stop_range_mm,
                duration_ms,
                now_ms,
            )?,
            RuntimeCommand::TurnBy {
                angle_mrad,
                angular_mrad_s,
                timeout_ms,
            } => self.start_turn_by(angle_mrad, angular_mrad_s, timeout_ms, now_ms)?,
            RuntimeCommand::DriveFor {
                distance_mm,
                velocity_mm_s,
                timeout_ms,
            } => self.start_drive_for(distance_mm, velocity_mm_s, timeout_ms, now_ms)?,
            RuntimeCommand::BumpEscape {
                direction,
                backoff_mm_s,
                turn_angular_mrad_s,
            } => self.queue_bump_escape(
                self.active_command_id.unwrap_or(0),
                direction,
                backoff_mm_s,
                turn_angular_mrad_s,
            )?,
            RuntimeCommand::HoldHeading {
                heading_error_mrad,
                velocity_mm_s,
                max_angular_mrad_s,
                duration_ms,
            } => self.start_hold_heading(
                heading_error_mrad,
                velocity_mm_s,
                max_angular_mrad_s,
                duration_ms,
                now_ms,
            )?,
            RuntimeCommand::TurnToHeading {
                heading_error_mrad,
                angular_mrad_s,
                tolerance_mrad,
                timeout_ms,
            } => self.start_face_bearing(
                heading_error_mrad,
                angular_mrad_s,
                tolerance_mrad,
                timeout_ms,
                now_ms,
            )?,
            RuntimeCommand::ArcFor {
                velocity_mm_s,
                radius_mm,
                duration_ms,
            } => self.start_drive_arc(velocity_mm_s, radius_mm, Some(duration_ms), now_ms)?,
            RuntimeCommand::CreepUntil {
                velocity_mm_s,
                angular_mrad_s,
                timeout_ms,
            } => self.start_cmd_vel(velocity_mm_s, angular_mrad_s, Some(timeout_ms), now_ms)?,
            RuntimeCommand::ScanArc {
                angle_mrad,
                angular_mrad_s,
                timeout_ms,
            } => self.start_turn_by(angle_mrad, angular_mrad_s, timeout_ms, now_ms)?,
            RuntimeCommand::DockAlign {
                bearing_mrad,
                range_mm,
                max_linear_mm_s,
                max_angular_mrad_s,
                stop_range_mm,
                duration_ms,
            } => self.start_track_bearing(
                bearing_mrad,
                range_mm,
                max_linear_mm_s,
                max_angular_mrad_s,
                stop_range_mm,
                duration_ms,
                now_ms,
            )?,
            RuntimeCommand::WallFollow {
                distance_error_mm,
                velocity_mm_s,
                max_angular_mrad_s,
                duration_ms,
            } => self.start_wall_follow(
                distance_error_mm,
                velocity_mm_s,
                max_angular_mrad_s,
                duration_ms,
                now_ms,
            )?,
            RuntimeCommand::WiggleAlign {
                amplitude_mrad,
                angular_mrad_s,
                cycles,
            } => self.queue_wiggle_align(amplitude_mrad, angular_mrad_s, cycles)?,
            RuntimeCommand::Unstick {
                direction,
                backoff_mm_s,
                turn_angular_mrad_s,
            } => self.queue_bump_escape(
                self.active_command_id.unwrap_or(0),
                direction,
                backoff_mm_s,
                turn_angular_mrad_s,
            )?,
            RuntimeCommand::CliffGuard { clear } => {
                if !clear {
                    self.stop_drive()?;
                }
            }
            RuntimeCommand::HeartbeatStop { timeout_ms } => {
                self.heartbeat_stop_at_ms = Some(now_ms.wrapping_add(timeout_ms));
            }
            RuntimeCommand::DriveArc {
                velocity_mm_s,
                radius_mm,
                duration_ms,
            } => self.start_drive_arc(velocity_mm_s, radius_mm, duration_ms, now_ms)?,
            RuntimeCommand::RequestSensors { packet_id } => {
                self.create_uart
                    .request_sensor_packet(&mut self.hardware, packet_id)?;
            }
            RuntimeCommand::StreamSensors {
                enabled,
                packet_id,
                period_ms,
            } => {
                if enabled {
                    self.sensor_stream = Some(SensorStream {
                        packet_id,
                        period_ms: period_ms.max(RUNTIME_TICK_MS),
                        next_request_ms: now_ms,
                    });
                } else {
                    self.sensor_stream = None;
                }
            }
            RuntimeCommand::SetSafetyPolicy { policy } => {
                self.safety_policy = policy;
                self.safety_latched = false;
            }
            RuntimeCommand::ClearMotionQueue => {
                self.clear_motion_queue()?;
            }
            RuntimeCommand::DefineChirp {
                kind,
                tones,
                tone_count,
            } => {
                let index = feedback_index(kind);
                self.chirps[index] = tones;
                self.chirp_counts[index] = tone_count.min(MAX_SONG_TONES as u8);
                self.ensure_create_responsive()?;
                self.create_uart.define_song(
                    &mut self.hardware,
                    &mut self.events,
                    feedback_slot(kind),
                    &self.chirps[index],
                    self.chirp_counts[index],
                )?;
            }
            RuntimeCommand::PlayFeedback { kind } => {
                if !self.create_responsive {
                    self.active = ActiveAction::None;
                    self.complete_active_command();
                    return Ok(());
                }
                self.ensure_create_responsive()?;
                let (tones, tone_count) = self.feedback_tones(kind);
                self.create_uart.define_song(
                    &mut self.hardware,
                    &mut self.events,
                    feedback_slot(kind),
                    &tones,
                    tone_count,
                )?;
                self.create_uart.play_song(
                    &mut self.hardware,
                    &mut self.events,
                    feedback_slot(kind),
                )?;
            }
            RuntimeCommand::CalibrateTurn {
                angular_mrad_s,
                duration_ms,
            } => self.start_cmd_vel(0, angular_mrad_s, Some(duration_ms), now_ms)?,
            RuntimeCommand::ResetOdometry => {
                status::mark_odometry_reset();
            }
            RuntimeCommand::ZeroImuOrientation => {
                let _ = status::zero_imu_orientation_from_gravity();
            }
            RuntimeCommand::ClearImuOrientation => {
                status::clear_imu_orientation_calibration();
            }
            RuntimeCommand::RestartMpu => match self.hardware.restart_imu() {
                Ok(()) => status::mark_imu_health(crate::drivers::imu::ImuHealth::Unknown),
                Err(health) => status::mark_imu_health(health),
            },
            RuntimeCommand::SongPlay { id } => {
                self.ensure_create_responsive()?;
                self.create_uart
                    .play_song(&mut self.hardware, &mut self.events, id)?;
            }
            RuntimeCommand::SongDefine {
                id,
                tones,
                tone_count,
            } => {
                self.ensure_create_responsive()?;
                self.create_uart.define_song(
                    &mut self.hardware,
                    &mut self.events,
                    id,
                    &tones,
                    tone_count,
                )?;
            }
            RuntimeCommand::Dock => {
                self.ensure_create_responsive()?;
                self.create_uart
                    .seek_dock(&mut self.hardware, &mut self.events)?;
            }
            RuntimeCommand::SetLights {
                led_bits,
                color,
                intensity,
            } => {
                self.create_uart.set_lights(
                    &mut self.hardware,
                    &mut self.events,
                    led_bits,
                    color,
                    intensity,
                )?;
            }
        }

        if self.active == ActiveAction::None {
            self.complete_active_command();
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
                    if self.active == ActiveAction::None {
                        self.complete_active_command();
                    }
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
                    self.complete_active_command();
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
                    self.complete_active_command();
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
                    self.create_responsive = false;
                    status::set_create_power_unknown();
                    status::set_oi_mode_unknown();
                    status::mark_uart_rx_error();
                    // Do not strand supervision in Error because RX failed.
                    // The queued START/FULL commands and periodic assertion
                    // still have value when the receive side is broken.
                    self.active = ActiveAction::None;
                    self.complete_active_command();
                    return Ok(());
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
                    self.complete_active_command();
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

    fn start_cmd_vel(
        &mut self,
        linear_mm_s: i16,
        angular_mrad_s: i16,
        duration_ms: Option<u32>,
        now_ms: u32,
    ) -> Result<(), BrainstemError> {
        let half_delta = angular_mrad_s as i32 * CREATE_AXLE_TRACK_MM / 2_000;
        let left = clamp_i16(linear_mm_s as i32 - half_delta);
        let right = clamp_i16(linear_mm_s as i32 + half_delta);
        self.start_drive_direct(left, right, duration_ms, now_ms)
    }

    fn start_face_bearing(
        &mut self,
        bearing_mrad: i16,
        max_angular_mrad_s: i16,
        tolerance_mrad: i16,
        ttl_ms: u32,
        now_ms: u32,
    ) -> Result<(), BrainstemError> {
        let error = bearing_mrad as i32;
        let tolerance = abs_i32(tolerance_mrad as i32);
        if abs_i32(error) <= tolerance {
            self.stop_drive()?;
            self.active = ActiveAction::None;
            return Ok(());
        }

        let max_angular = abs_i32(max_angular_mrad_s as i32);
        if max_angular == 0 || ttl_ms == 0 {
            self.stop_drive()?;
            return Err(BrainstemError::Timeout);
        }

        let angular = clamp_i16(error.clamp(-max_angular, max_angular));
        let turn_ms = ((abs_i32(error) - tolerance) as u32)
            .saturating_mul(1_000)
            .checked_div(abs_i32(angular as i32) as u32)
            .unwrap_or(ttl_ms)
            .max(1)
            .min(ttl_ms);
        self.start_cmd_vel(0, angular, Some(turn_ms), now_ms)
    }

    fn start_track_bearing(
        &mut self,
        bearing_mrad: i16,
        range_mm: u16,
        max_linear_mm_s: i16,
        max_angular_mrad_s: i16,
        stop_range_mm: u16,
        ttl_ms: u32,
        now_ms: u32,
    ) -> Result<(), BrainstemError> {
        if ttl_ms == 0 || (stop_range_mm > 0 && range_mm <= stop_range_mm) {
            self.stop_drive()?;
            self.active = ActiveAction::None;
            return Ok(());
        }

        let error = bearing_mrad as i32;
        let max_angular = abs_i32(max_angular_mrad_s as i32);
        let angular = clamp_i16(error.clamp(-max_angular, max_angular));
        let slowdown = abs_i32(error).min(BEARING_SLOWDOWN_MRAD);
        let scale = BEARING_SLOWDOWN_MRAD - slowdown;
        let max_linear = abs_i32(max_linear_mm_s as i32);
        let mut linear = max_linear * scale / BEARING_SLOWDOWN_MRAD;
        if linear > 0 {
            linear = linear.max(MIN_TRACK_SPEED_MM_S).min(max_linear);
        }
        if max_linear_mm_s < 0 {
            linear = -linear;
        }

        self.start_cmd_vel(clamp_i16(linear), angular, Some(ttl_ms), now_ms)
    }

    fn start_turn_by(
        &mut self,
        angle_mrad: i16,
        angular_mrad_s: i16,
        timeout_ms: u32,
        now_ms: u32,
    ) -> Result<(), BrainstemError> {
        let angle = angle_mrad as i32;
        let angular_abs = abs_i32(angular_mrad_s as i32);
        if angle == 0 || angular_abs == 0 || timeout_ms == 0 {
            self.stop_drive()?;
            self.active = ActiveAction::None;
            return Ok(());
        }

        let angular = if angle > 0 { angular_abs } else { -angular_abs };
        let duration_ms = (abs_i32(angle) as u32)
            .saturating_mul(1_000)
            .checked_div(angular_abs as u32)
            .unwrap_or(timeout_ms)
            .max(1)
            .min(timeout_ms);
        self.start_cmd_vel(0, clamp_i16(angular), Some(duration_ms), now_ms)
    }

    fn start_drive_for(
        &mut self,
        distance_mm: i16,
        velocity_mm_s: i16,
        timeout_ms: u32,
        now_ms: u32,
    ) -> Result<(), BrainstemError> {
        let distance = distance_mm as i32;
        let velocity_abs = abs_i32(velocity_mm_s as i32);
        if distance == 0 || velocity_abs == 0 || timeout_ms == 0 {
            self.stop_drive()?;
            self.active = ActiveAction::None;
            return Ok(());
        }

        let velocity = if distance > 0 {
            velocity_abs
        } else {
            -velocity_abs
        };
        let duration_ms = (abs_i32(distance) as u32)
            .saturating_mul(1_000)
            .checked_div(velocity_abs as u32)
            .unwrap_or(timeout_ms)
            .max(1)
            .min(timeout_ms);
        self.start_cmd_vel(clamp_i16(velocity), 0, Some(duration_ms), now_ms)
    }

    fn queue_bump_escape(
        &mut self,
        command_id: u32,
        direction: EscapeDirection,
        backoff_mm_s: i16,
        turn_angular_mrad_s: i16,
    ) -> Result<(), BrainstemError> {
        self.ensure_motion_allowed()?;
        let backoff = -abs_i32(backoff_mm_s as i32);
        let turn_abs = abs_i32(turn_angular_mrad_s as i32);
        let turn = match direction {
            EscapeDirection::Left => turn_abs,
            EscapeDirection::Right => -turn_abs,
            EscapeDirection::Either => turn_abs,
        };
        let _ = self.commands.push_front(QueuedCommand {
            command_id,
            command: RuntimeCommand::CmdVel {
                linear_mm_s: 0,
                angular_mrad_s: clamp_i16(turn),
                duration_ms: Some(BUMP_ESCAPE_TURN_MS),
            },
        });
        let _ = self.commands.push_front(QueuedCommand {
            command_id,
            command: RuntimeCommand::CmdVel {
                linear_mm_s: clamp_i16(backoff),
                angular_mrad_s: 0,
                duration_ms: Some(BUMP_ESCAPE_BACKOFF_MS),
            },
        });
        self.active = ActiveAction::None;
        Ok(())
    }

    fn start_hold_heading(
        &mut self,
        heading_error_mrad: i16,
        velocity_mm_s: i16,
        max_angular_mrad_s: i16,
        ttl_ms: u32,
        now_ms: u32,
    ) -> Result<(), BrainstemError> {
        let max_angular = abs_i32(max_angular_mrad_s as i32);
        let angular = (heading_error_mrad as i32).clamp(-max_angular, max_angular);
        self.start_cmd_vel(velocity_mm_s, clamp_i16(angular), Some(ttl_ms), now_ms)
    }

    fn start_wall_follow(
        &mut self,
        distance_error_mm: i16,
        velocity_mm_s: i16,
        max_angular_mrad_s: i16,
        ttl_ms: u32,
        now_ms: u32,
    ) -> Result<(), BrainstemError> {
        let max_angular = abs_i32(max_angular_mrad_s as i32);
        let angular = (distance_error_mm as i32 * 8).clamp(-max_angular, max_angular);
        self.start_cmd_vel(velocity_mm_s, clamp_i16(angular), Some(ttl_ms), now_ms)
    }

    fn queue_wiggle_align(
        &mut self,
        amplitude_mrad: i16,
        angular_mrad_s: i16,
        cycles: u8,
    ) -> Result<(), BrainstemError> {
        self.ensure_motion_allowed()?;
        let cycles = cycles.min(6);
        for cycle in (0..cycles).rev() {
            let sign = if cycle % 2 == 0 { 1 } else { -1 };
            let _ = self.commands.push_front(QueuedCommand {
                command_id: self.active_command_id.unwrap_or(0),
                command: RuntimeCommand::TurnBy {
                    angle_mrad: clamp_i16(amplitude_mrad as i32 * sign),
                    angular_mrad_s,
                    timeout_ms: 800,
                },
            });
        }
        self.active = ActiveAction::None;
        Ok(())
    }

    fn clear_motion_queue(&mut self) -> Result<(), BrainstemError> {
        let pending = self.commands.len();
        for _ in 0..pending {
            let Some(command) = self.commands.pop_front() else {
                break;
            };
            if !is_motion_command(command.command) {
                let _ = self.commands.push_back(command);
            }
        }
        if matches!(self.active, ActiveAction::Driving { .. }) {
            self.interrupt_active_command();
            self.stop_drive()?;
            self.active = ActiveAction::None;
        }
        Ok(())
    }

    fn poll_sensor_stream(&mut self) -> Result<(), BrainstemError> {
        let Some(mut stream) = self.sensor_stream else {
            return Ok(());
        };
        let now_ms = self.now_ms();
        if self.create_responsive && time_reached(now_ms, stream.next_request_ms) {
            self.create_uart
                .request_sensor_packet(&mut self.hardware, stream.packet_id)?;
            stream.next_request_ms = now_ms.wrapping_add(stream.period_ms);
        }
        self.sensor_stream = Some(stream);
        Ok(())
    }

    fn poll_imu(&mut self) {
        if !body::IMU_ENABLED {
            status::mark_imu_health(crate::drivers::imu::ImuHealth::Absent);
            return;
        }

        let now_ms = self.now_ms();
        if !time_reached(now_ms, self.next_imu_poll_ms) {
            return;
        }
        self.next_imu_poll_ms = now_ms.wrapping_add(body::IMU_POLL_PERIOD_MS.max(1));

        match self.hardware.poll_imu_sample(now_ms) {
            Ok(Some(sample)) => status::mark_imu_sample(sample),
            Ok(None) => {}
            Err(health) => status::mark_imu_health(health),
        }
    }

    fn enforce_safety_policy(&mut self) -> Result<(), BrainstemError> {
        let snapshot = status::snapshot(self.now_ms());
        let flags = snapshot.create_sensor_flags;
        let bump = flags & ((1 << 0) | (1 << 1)) != 0;
        let wheel_drop = flags & (1 << 2) != 0;
        let cliff = flags & ((1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)) != 0;
        let imu_ok = body::IMU_ENABLED && snapshot.imu_health == status::ImuHealthCode::Ok as u8;
        let tilt = imu_ok && snapshot.imu_tilt_magnitude_mrad as i16 >= body::IMU_TILT_STOP_MRAD;
        let impact = imu_ok && snapshot.imu_impact_score_mm_s2 >= body::IMU_IMPACT_STOP_MM_S2;

        if !bump && !cliff && !wheel_drop && !tilt && !impact {
            self.update_safety_edges(bump, cliff, wheel_drop);
            if !self.safety_policy.wheel_drop_latch {
                if self.safety_latched {
                    status::mark_safety_cleared(status::SafetyEventKind::WheelDrop);
                    status::mark_wheel_drop_cleared();
                }
                self.safety_latched = false;
            }
            return Ok(());
        }
        self.update_safety_edges(bump, cliff, wheel_drop);
        if self.safety_latched {
            return Ok(());
        }

        if wheel_drop && self.safety_policy.wheel_drop_latch {
            status::mark_safety_tripped(status::SafetyEventKind::WheelDrop);
            status::mark_wheel_drop_latched();
            self.safety_latched = true;
            self.interrupt_active_command();
            self.commands.clear();
            self.stop_drive()?;
            self.active = ActiveAction::None;
            let _ = self.play_feedback_now(FeedbackKind::Danger);
            return Ok(());
        }

        let action = if tilt {
            status::mark_safety_tripped(status::SafetyEventKind::Tilt);
            SafetyAction::Stop
        } else if impact {
            status::mark_safety_tripped(status::SafetyEventKind::Impact);
            SafetyAction::Stop
        } else if cliff {
            status::mark_safety_tripped(status::SafetyEventKind::Cliff);
            self.safety_policy.cliff
        } else if bump {
            status::mark_safety_tripped(status::SafetyEventKind::Bump);
            self.safety_policy.bump
        } else {
            SafetyAction::Stop
        };
        self.apply_safety_action(action)?;
        let _ = self.play_feedback_now(FeedbackKind::Danger);
        Ok(())
    }

    fn apply_safety_action(&mut self, action: SafetyAction) -> Result<(), BrainstemError> {
        match action {
            SafetyAction::None => Ok(()),
            SafetyAction::Stop => {
                self.safety_latched = true;
                self.interrupt_active_command();
                self.commands.clear();
                self.stop_drive()?;
                self.active = ActiveAction::None;
                Ok(())
            }
            SafetyAction::Backoff => {
                self.safety_latched = true;
                let command_id = self.active_command_id.unwrap_or(0);
                self.interrupt_active_command();
                self.commands.clear();
                self.active = ActiveAction::None;
                let _ = self.commands.push_front(QueuedCommand {
                    command_id,
                    command: RuntimeCommand::CmdVel {
                        linear_mm_s: -80,
                        angular_mrad_s: 0,
                        duration_ms: Some(BUMP_ESCAPE_BACKOFF_MS),
                    },
                });
                Ok(())
            }
            SafetyAction::BumpEscape => {
                self.safety_latched = true;
                let command_id = self.active_command_id.unwrap_or(0);
                self.interrupt_active_command();
                self.commands.clear();
                self.active = ActiveAction::None;
                self.queue_bump_escape(command_id, EscapeDirection::Either, 80, 900)
            }
        }
    }

    fn feedback_tones(&self, kind: FeedbackKind) -> ([SongTone; MAX_SONG_TONES], u8) {
        let index = feedback_index(kind);
        if self.chirp_counts[index] > 0 {
            return (self.chirps[index], self.chirp_counts[index]);
        }
        default_feedback_tones(kind)
    }

    fn play_feedback_now(&mut self, kind: FeedbackKind) -> Result<(), BrainstemError> {
        self.ensure_create_responsive()?;
        let (tones, tone_count) = self.feedback_tones(kind);
        self.create_uart.define_song(
            &mut self.hardware,
            &mut self.events,
            feedback_slot(kind),
            &tones,
            tone_count,
        )?;
        self.create_uart
            .play_song(&mut self.hardware, &mut self.events, feedback_slot(kind))
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

        status::set_body_state(BodyState::Moving);
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

        status::set_body_state(BodyState::Moving);
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

    fn enforce_heartbeat_stop(&mut self) -> Result<(), BrainstemError> {
        let Some(deadline_ms) = self.heartbeat_stop_at_ms else {
            return Ok(());
        };
        if time_reached(self.now_ms(), deadline_ms) {
            self.interrupt_active_command();
            self.commands.clear();
            self.active = ActiveAction::None;
            self.heartbeat_stop_at_ms = None;
            status::revoke_authority();
            status::mark_heartbeat_expired();
            status::mark_safety_tripped(status::SafetyEventKind::Heartbeat);
            self.stop_drive()?;
        }
        Ok(())
    }

    fn enter_idle(&mut self) {
        let _ = self.stop_drive();
        self.complete_active_command();
        self.mode = RuntimeMode::Idle;
        self.active = ActiveAction::None;
        status::set_runtime_state(RuntimeState::Idle);
        status::set_body_state(BodyState::Idle);
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
        self.fail_active_command(error);
        let _ = self.stop_drive();
        self.mode = RuntimeMode::Error;
        self.active = ActiveAction::None;
        self.error_blink_next_ms = self.now_ms();
        self.error_blink_count = 0;
        self.error_blink_on = false;
        if self.create_responsive {
            let _ = self.play_feedback_now(FeedbackKind::Error);
        }
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

    fn complete_active_command(&mut self) {
        if let Some(command_id) = self.active_command_id.take() {
            status::mark_command_completed(command_id);
        }
    }

    fn interrupt_active_command(&mut self) {
        if let Some(command_id) = self.active_command_id.take() {
            status::mark_command_interrupted(command_id);
        }
    }

    fn fail_active_command(&mut self, error: BrainstemError) {
        let Some(command_id) = self.active_command_id.take() else {
            return;
        };
        match error {
            BrainstemError::CreateNoResponse | BrainstemError::Timeout => {
                status::mark_command_timed_out(command_id);
            }
            BrainstemError::UartFraming | BrainstemError::InvalidPacket => {
                status::mark_command_interrupted(command_id);
            }
        }
    }

    fn update_safety_edges(&mut self, bump: bool, cliff: bool, wheel_drop: bool) {
        if bump != self.last_bump {
            status::mark_bump_changed(bump);
            self.last_bump = bump;
        }
        if cliff != self.last_cliff {
            status::mark_cliff_changed(cliff);
            self.last_cliff = cliff;
        }
        if wheel_drop != self.last_wheel_drop {
            if !wheel_drop {
                status::mark_wheel_drop_cleared();
            }
            self.last_wheel_drop = wheel_drop;
        }
    }
}

fn time_reached(now_ms: u32, deadline_ms: u32) -> bool {
    now_ms.wrapping_sub(deadline_ms) < u32::MAX / 2
}

fn low_battery_and_charging(snapshot: &status::BrainstemStatus) -> bool {
    let actively_charging = matches!(snapshot.create_sensor_charging_state, 1..=3);
    actively_charging
        && snapshot.create_sensor_capacity_mah > 0
        && u32::from(snapshot.create_sensor_charge_mah) * 100
            <= u32::from(snapshot.create_sensor_capacity_mah) * LOW_BATTERY_PERCENT
}

fn runtime_command_from_forebrain(command: BrainstemCommand) -> Option<RuntimeCommand> {
    match command {
        BrainstemCommand::Ping
        | BrainstemCommand::Status
        | BrainstemCommand::Bootsel
        | BrainstemCommand::Arm
        | BrainstemCommand::Disarm
        | BrainstemCommand::RestartCreate => None,
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
        BrainstemCommand::DriveDirect {
            left_mm_s,
            right_mm_s,
            ttl_ms,
            ..
        } => Some(RuntimeCommand::DriveDirect {
            left_mm_s,
            right_mm_s,
            duration_ms: Some(ttl_ms),
        }),
        BrainstemCommand::DriveArc {
            velocity_mm_s,
            radius_mm,
            ttl_ms,
            ..
        } => Some(RuntimeCommand::DriveArc {
            velocity_mm_s,
            radius_mm,
            duration_ms: Some(ttl_ms),
        }),
        BrainstemCommand::FaceBearing {
            bearing_mrad,
            max_angular_mrad_s,
            tolerance_mrad,
            ttl_ms,
            ..
        } => Some(RuntimeCommand::FaceBearing {
            bearing_mrad,
            max_angular_mrad_s,
            tolerance_mrad,
            duration_ms: ttl_ms,
        }),
        BrainstemCommand::TrackBearing {
            bearing_mrad,
            range_mm,
            max_linear_mm_s,
            max_angular_mrad_s,
            stop_range_mm,
            ttl_ms,
            ..
        } => Some(RuntimeCommand::TrackBearing {
            bearing_mrad,
            range_mm,
            max_linear_mm_s,
            max_angular_mrad_s,
            stop_range_mm,
            duration_ms: ttl_ms,
        }),
        BrainstemCommand::TurnBy {
            angle_mrad,
            angular_mrad_s,
            timeout_ms,
            ..
        } => Some(RuntimeCommand::TurnBy {
            angle_mrad,
            angular_mrad_s,
            timeout_ms,
        }),
        BrainstemCommand::DriveFor {
            distance_mm,
            velocity_mm_s,
            timeout_ms,
            ..
        } => Some(RuntimeCommand::DriveFor {
            distance_mm,
            velocity_mm_s,
            timeout_ms,
        }),
        BrainstemCommand::BumpEscape {
            direction,
            backoff_mm_s,
            turn_angular_mrad_s,
            ..
        } => Some(RuntimeCommand::BumpEscape {
            direction,
            backoff_mm_s,
            turn_angular_mrad_s,
        }),
        BrainstemCommand::HoldHeading {
            heading_error_mrad,
            velocity_mm_s,
            max_angular_mrad_s,
            ttl_ms,
            ..
        } => Some(RuntimeCommand::HoldHeading {
            heading_error_mrad,
            velocity_mm_s,
            max_angular_mrad_s,
            duration_ms: ttl_ms,
        }),
        BrainstemCommand::TurnToHeading {
            heading_error_mrad,
            angular_mrad_s,
            tolerance_mrad,
            timeout_ms,
            ..
        } => Some(RuntimeCommand::TurnToHeading {
            heading_error_mrad,
            angular_mrad_s,
            tolerance_mrad,
            timeout_ms,
        }),
        BrainstemCommand::ArcFor {
            velocity_mm_s,
            radius_mm,
            duration_ms,
            ..
        } => Some(RuntimeCommand::ArcFor {
            velocity_mm_s,
            radius_mm,
            duration_ms,
        }),
        BrainstemCommand::CreepUntil {
            velocity_mm_s,
            angular_mrad_s,
            timeout_ms,
            ..
        } => Some(RuntimeCommand::CreepUntil {
            velocity_mm_s,
            angular_mrad_s,
            timeout_ms,
        }),
        BrainstemCommand::ScanArc {
            angle_mrad,
            angular_mrad_s,
            timeout_ms,
            ..
        } => Some(RuntimeCommand::ScanArc {
            angle_mrad,
            angular_mrad_s,
            timeout_ms,
        }),
        BrainstemCommand::DockAlign {
            bearing_mrad,
            range_mm,
            max_linear_mm_s,
            max_angular_mrad_s,
            stop_range_mm,
            ttl_ms,
            ..
        } => Some(RuntimeCommand::DockAlign {
            bearing_mrad,
            range_mm,
            max_linear_mm_s,
            max_angular_mrad_s,
            stop_range_mm,
            duration_ms: ttl_ms,
        }),
        BrainstemCommand::WallFollow {
            distance_error_mm,
            velocity_mm_s,
            max_angular_mrad_s,
            ttl_ms,
            ..
        } => Some(RuntimeCommand::WallFollow {
            distance_error_mm,
            velocity_mm_s,
            max_angular_mrad_s,
            duration_ms: ttl_ms,
        }),
        BrainstemCommand::WiggleAlign {
            amplitude_mrad,
            angular_mrad_s,
            cycles,
            ..
        } => Some(RuntimeCommand::WiggleAlign {
            amplitude_mrad,
            angular_mrad_s,
            cycles,
        }),
        BrainstemCommand::Unstick {
            direction,
            backoff_mm_s,
            turn_angular_mrad_s,
            ..
        } => Some(RuntimeCommand::Unstick {
            direction,
            backoff_mm_s,
            turn_angular_mrad_s,
        }),
        BrainstemCommand::CliffGuard { clear, .. } => Some(RuntimeCommand::CliffGuard { clear }),
        BrainstemCommand::HeartbeatStop { timeout_ms, .. } => {
            Some(RuntimeCommand::HeartbeatStop { timeout_ms })
        }
        BrainstemCommand::RequestSensors { packet_id, .. } => {
            Some(RuntimeCommand::RequestSensors { packet_id })
        }
        BrainstemCommand::StreamSensors {
            enabled,
            packet_id,
            period_ms,
            ..
        } => Some(RuntimeCommand::StreamSensors {
            enabled,
            packet_id,
            period_ms,
        }),
        BrainstemCommand::SetSafetyPolicy { policy, .. } => {
            Some(RuntimeCommand::SetSafetyPolicy { policy })
        }
        BrainstemCommand::ClearMotionQueue { .. } => Some(RuntimeCommand::ClearMotionQueue),
        BrainstemCommand::DefineChirp {
            kind,
            tones,
            tone_count,
            ..
        } => Some(RuntimeCommand::DefineChirp {
            kind,
            tones,
            tone_count,
        }),
        BrainstemCommand::PlayFeedback { kind, .. } => Some(RuntimeCommand::PlayFeedback { kind }),
        BrainstemCommand::PowerState { request, .. } => match request {
            PowerStateRequest::Wake => Some(RuntimeCommand::WakeCreate),
            PowerStateRequest::Sleep => Some(RuntimeCommand::SleepCreate),
            PowerStateRequest::PulseBrc => Some(RuntimeCommand::PulseBrc),
            PowerStateRequest::StartOi => Some(RuntimeCommand::StartOi),
            PowerStateRequest::DebugBaud19200 => Some(RuntimeCommand::SetCreateBaud(19_200)),
            PowerStateRequest::DebugBaud57600 => Some(RuntimeCommand::SetCreateBaud(57_600)),
            PowerStateRequest::DebugBaud115200 => Some(RuntimeCommand::SetCreateBaud(115_200)),
        },
        BrainstemCommand::CalibrateTurn {
            angular_mrad_s,
            duration_ms,
            ..
        } => Some(RuntimeCommand::CalibrateTurn {
            angular_mrad_s,
            duration_ms,
        }),
        BrainstemCommand::ResetOdometry { .. } => Some(RuntimeCommand::ResetOdometry),
        BrainstemCommand::ZeroImuOrientation { .. } => Some(RuntimeCommand::ZeroImuOrientation),
        BrainstemCommand::ClearImuOrientation { .. } => Some(RuntimeCommand::ClearImuOrientation),
        BrainstemCommand::RestartMpu => Some(RuntimeCommand::RestartMpu),
        BrainstemCommand::SongPlay { id } => Some(RuntimeCommand::SongPlay { id }),
        BrainstemCommand::SongDefine {
            id,
            tones,
            tone_count,
            ..
        } => Some(RuntimeCommand::SongDefine {
            id,
            tones,
            tone_count,
        }),
        BrainstemCommand::Dock => Some(RuntimeCommand::Dock),
        BrainstemCommand::SetLights {
            led_bits,
            color,
            intensity,
        } => Some(RuntimeCommand::SetLights {
            led_bits,
            color,
            intensity,
        }),
        BrainstemCommand::GetCapabilities => None,
        BrainstemCommand::GetEvents { .. } => None,
    }
}

fn clamp_i16(value: i32) -> i16 {
    value.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

fn abs_i32(value: i32) -> i32 {
    value.saturating_abs()
}

fn is_motion_command(command: RuntimeCommand) -> bool {
    matches!(
        command,
        RuntimeCommand::DriveDirect { .. }
            | RuntimeCommand::CmdVel { .. }
            | RuntimeCommand::DriveArc { .. }
            | RuntimeCommand::Drive { .. }
            | RuntimeCommand::StopDrive
            | RuntimeCommand::FaceBearing { .. }
            | RuntimeCommand::TrackBearing { .. }
            | RuntimeCommand::TurnBy { .. }
            | RuntimeCommand::DriveFor { .. }
            | RuntimeCommand::BumpEscape { .. }
            | RuntimeCommand::HoldHeading { .. }
            | RuntimeCommand::TurnToHeading { .. }
            | RuntimeCommand::ArcFor { .. }
            | RuntimeCommand::CreepUntil { .. }
            | RuntimeCommand::ScanArc { .. }
            | RuntimeCommand::DockAlign { .. }
            | RuntimeCommand::WallFollow { .. }
            | RuntimeCommand::WiggleAlign { .. }
            | RuntimeCommand::Unstick { .. }
            | RuntimeCommand::CliffGuard { .. }
            | RuntimeCommand::CalibrateTurn { .. }
    )
}

fn feedback_index(kind: FeedbackKind) -> usize {
    match kind {
        FeedbackKind::Ok => 0,
        FeedbackKind::Error => 1,
        FeedbackKind::Armed => 2,
        FeedbackKind::LostTarget => 3,
        FeedbackKind::DockSeen => 4,
        FeedbackKind::Danger => 5,
    }
}

fn feedback_slot(kind: FeedbackKind) -> u8 {
    FEEDBACK_SLOT_BASE + feedback_index(kind) as u8
}

fn default_feedback_tones(kind: FeedbackKind) -> ([SongTone; MAX_SONG_TONES], u8) {
    let mut tones = [SongTone::default(); MAX_SONG_TONES];
    let notes: &[(u8, u8)] = match kind {
        FeedbackKind::Ok => &[(76, 6), (84, 10)],
        FeedbackKind::Error => &[(45, 12), (40, 16)],
        // Solresol "fasolsi": prepare / make ready.
        FeedbackKind::Armed => &[(65, 8), (67, 8), (71, 12)],
        FeedbackKind::LostTarget => &[(55, 8), (52, 8), (48, 12)],
        FeedbackKind::DockSeen => &[(67, 8), (71, 8), (74, 12)],
        FeedbackKind::Danger => &[(40, 6), (40, 6), (40, 12)],
    };
    for (i, (note, duration_64ths)) in notes.iter().enumerate() {
        tones[i] = SongTone {
            note: *note,
            duration_64ths: *duration_64ths,
        };
    }
    (tones, notes.len() as u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drivers::imu::ImuSample;
    use crate::hardware::SerialRead;

    struct FakeHardware {
        now_us: u32,
        writes: heapless::Vec<u8, 32>,
        imu_sample: Option<ImuSample>,
        imu_health: Option<crate::drivers::imu::ImuHealth>,
    }

    impl FakeHardware {
        fn new(now_ms: u32) -> Self {
            Self {
                now_us: now_ms * 1_000,
                writes: heapless::Vec::new(),
                imu_sample: None,
                imu_health: None,
            }
        }

        fn with_imu_sample(now_ms: u32, imu_sample: ImuSample) -> Self {
            Self {
                now_us: now_ms * 1_000,
                writes: heapless::Vec::new(),
                imu_sample: Some(imu_sample),
                imu_health: None,
            }
        }
    }

    impl BrainstemHardware for FakeHardware {
        fn delay_ms(&mut self, ms: u32) {
            self.now_us = self.now_us.wrapping_add(ms * 1_000);
        }

        fn now_us(&mut self) -> u32 {
            self.now_us
        }

        fn feed_watchdog(&mut self) {}

        fn set_power_toggle(&mut self, _high: bool) {}

        fn set_brc(&mut self, _high: bool) {}

        fn set_indicators(&mut self, _on: bool) {}

        fn set_primary_indicator(&mut self, _on: bool) {}

        fn write_byte(&mut self, byte: u8) -> Result<(), ()> {
            let _ = self.writes.push(byte);
            Ok(())
        }

        fn flush_uart(&mut self) -> Result<(), ()> {
            Ok(())
        }

        fn read_byte(&mut self) -> SerialRead {
            SerialRead::WouldBlock
        }

        fn poll_imu_sample(
            &mut self,
            _now_ms: u32,
        ) -> Result<Option<ImuSample>, crate::drivers::imu::ImuHealth> {
            if let Some(health) = self.imu_health {
                return Err(health);
            }
            Ok(self.imu_sample.take())
        }
    }

    #[test]
    fn startup_acquires_create_in_full_mode() {
        let _guard = status::status_test_guard();
        let mut runtime = Runtime::new(FakeHardware::new(0));

        runtime.start();

        assert_eq!(runtime.commands.len(), ACQUIRE_CREATE_SCRIPT.len());
        assert!(runtime.commands.iter().any(|queued| matches!(
            queued.command,
            RuntimeCommand::SetMode(crate::commands::CreateOiMode::Full)
        )));
        assert!(matches!(runtime.mode, RuntimeMode::Running));
        assert_eq!(
            status::snapshot(0).current_runtime_state,
            RuntimeState::Running as u8
        );
    }

    #[test]
    fn responsive_create_is_refreshed_in_full_mode() {
        let _guard = status::status_test_guard();
        let mut runtime = Runtime::new(FakeHardware::new(1_000));
        runtime.create_responsive = true;
        status::set_oi_mode(crate::commands::CreateOiMode::Full);

        assert!(runtime.maintain_full_mode().is_ok());

        assert!(runtime.hardware.writes.contains(&132));
        assert_eq!(status::snapshot(1_000).oi_mode, 3);
    }

    #[test]
    fn unresponsive_create_still_gets_start_and_full() {
        let _guard = status::status_test_guard();
        let mut runtime = Runtime::new(FakeHardware::new(1_000));

        assert!(runtime.maintain_full_mode().is_ok());

        assert_eq!(runtime.hardware.writes.as_slice(), &[128, 132]);
        assert_eq!(status::snapshot(1_000).oi_mode, 3);
    }

    #[test]
    fn low_battery_while_charging_pauses_full_mode_assertion() {
        let _guard = status::status_test_guard();
        status::mark_create_sensor_packet(
            0,
            crate::events::CreateSensorPacket {
                charging_state: 2,
                charge_mah: 200,
                capacity_mah: 2_000,
                ..crate::events::CreateSensorPacket::default()
            },
        );
        let mut runtime = Runtime::new(FakeHardware::new(1_000));

        assert!(runtime.maintain_full_mode().is_ok());

        assert!(runtime.hardware.writes.is_empty());
        status::mark_create_sensor_packet(
            0,
            crate::events::CreateSensorPacket {
                charging_state: 0,
                charge_mah: 2_000,
                capacity_mah: 2_000,
                ..crate::events::CreateSensorPacket::default()
            },
        );
    }

    #[test]
    fn armed_feedback_says_fasolsi() {
        let (tones, count) = default_feedback_tones(FeedbackKind::Armed);

        assert_eq!(count, 3);
        assert_eq!(tones[0].note, 65); // fa
        assert_eq!(tones[1].note, 67); // sol
        assert_eq!(tones[2].note, 71); // si
    }

    #[test]
    fn runtime_poll_imu_sample_updates_status() {
        let _guard = status::status_test_guard();
        status::clear_imu_orientation_calibration();
        let previous_samples = status::snapshot(1_000).imu_sample_count;
        let mut runtime = Runtime::new(FakeHardware::with_imu_sample(
            1_000,
            ImuSample {
                timestamp_ms: 1_000,
                gyro_z_mrad_s: 500,
                ..ImuSample::stationary(1_000)
            },
        ));

        runtime.tick();

        let snapshot = status::snapshot(1_000);
        assert_eq!(snapshot.imu_health, status::ImuHealthCode::Ok as u8);
        assert_eq!(snapshot.imu_sample_count, previous_samples.wrapping_add(1));
        assert_eq!(snapshot.imu_yaw_rate_mrad_s, 500);
    }

    #[test]
    fn missing_imu_does_not_stop_motion() {
        let _guard = status::status_test_guard();
        let mut hardware = FakeHardware::new(1_000);
        hardware.imu_health = Some(crate::drivers::imu::ImuHealth::Absent);
        let mut runtime = Runtime::new(hardware);
        runtime.create_responsive = true;
        runtime.active = ActiveAction::Driving { stop_at_ms: 5_000 };
        runtime.active_command_id = Some(77);

        runtime.tick();

        let snapshot = status::snapshot(1_000);
        assert_eq!(snapshot.imu_health, status::ImuHealthCode::Absent as u8);
        assert!(matches!(runtime.active, ActiveAction::Driving { .. }));
        assert!(!runtime.safety_latched);
    }

    #[test]
    fn safety_tick_stops_active_motion_on_imu_tilt() {
        let _guard = status::status_test_guard();
        status::clear_imu_orientation_calibration();
        let mut runtime = Runtime::new(FakeHardware::new(1_000));
        runtime.create_responsive = true;
        runtime.active = ActiveAction::Driving { stop_at_ms: 5_000 };
        runtime.active_command_id = Some(77);
        status::mark_imu_sample(ImuSample {
            timestamp_ms: 1_000,
            accel_x_mm_s2: 9_807,
            accel_y_mm_s2: 0,
            accel_z_mm_s2: 1_000,
            ..ImuSample::stationary(1_000)
        });

        runtime.tick();

        assert!(matches!(runtime.active, ActiveAction::None));
        assert!(runtime.safety_latched);
        assert!(runtime
            .events
            .iter()
            .any(|event| matches!(event, BrainstemEvent::DriveStopped)));
        assert!(!runtime.hardware.writes.is_empty());
    }
}
