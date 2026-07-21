use heapless::Deque;
use pete_cockpit_protocol::{CONTACT_WITHDRAWAL_DURATION_MS, CONTACT_WITHDRAWAL_SPEED_MM_S};

use crate::audio::{
    cue_tones, tone_duration_ms, AudioAnnunciator, AuditoryCue, CueRequestResult,
    AUTOMATIC_CUE_SLOT,
};
use crate::body;
use crate::commands::{
    BrainstemCommand, FeedbackKind, PowerStateRequest, RuntimeCommand, SafetyLatchKind, SongTone,
    ACQUIRE_CREATE_SCRIPT, DISARM_SCRIPT, MAX_SONG_TONES, RESTART_CREATE_SCRIPT,
};
use crate::drivers::{
    create_uart::{CreateUart, CREATE_BUTTON_LED_MASK, CREATE_LED_ADVANCE, CREATE_LED_PLAY},
    leds::Leds,
    timers::Timers,
};
use crate::events::{BrainstemError, BrainstemEvent};
use crate::hardware::BrainstemHardware;
use crate::network_registry;
use crate::status::{self, BodyState, RuntimeActionCode, RuntimeState};

const EVENT_QUEUE_CAPACITY: usize = 16;
const COMMAND_QUEUE_CAPACITY: usize = 16;
pub(crate) const RUNTIME_TICK_MS: u32 = 10;
const SENSOR_PROBE_PERIOD_MS: u32 = 100;
const FULL_MODE_REFRESH_PERIOD_MS: u32 = 1_000;
const HEALTHY_LIGHT_STEP_MS: u32 = 100;
const LOW_BATTERY_PERCENT: u32 = 20;
const LOW_BATTERY_AUDIO_CLEAR_PERCENT: u32 = 25;
const WAKE_PROBE_RESPONSE_BYTES_REQUIRED: u8 = 1;
const CREATE_AXLE_TRACK_MM: i32 = 258;
const FEEDBACK_SLOT_BASE: u8 = 10;
const FEEDBACK_KIND_COUNT: usize = 6;
const MOTHERBRAIN_RESET_PULSE_MS: u32 = 100;
const MOTHERBRAIN_RESET_COOLDOWN_MS: u32 = 30_000;
const MOTHERBRAIN_RESET_HISTORY_CAPACITY: usize = 16;
const MOTHERBRAIN_RESET_SERVICE_SCOPE: u8 = 4;
const CONNECTED_POWER_LED_COLOR: u8 = 128;
const CONNECTED_POWER_LED_INTENSITY: u8 = 255;
const ORIENTATION_PROBE_IMU_MAX_AGE_MS: u32 = body::IMU_POLL_PERIOD_MS * 5;
const ORIENTATION_PROBE_MIN_ACCEL_MM_S2: u16 = 7_000;
const ORIENTATION_PROBE_MAX_ACCEL_MM_S2: u16 = 13_000;
const CONTACT_REPEAT_WINDOW_MS: u32 = 2_000;
const DOCK_DEPARTURE_SPEED_MM_S: i16 = -200;
const DOCK_DEPARTURE_DURATION_MS: u32 = 1_500;
const CREATE_CHARGING_SOURCES_PACKET: u8 = 34;
const CREATE_CHARGING_SOURCES_POLL_PERIOD_MS: u32 = 250;
const CREATE_COMPLETE_SENSOR_PACKET: u8 = 0;
const CREATE_COMPLETE_SENSOR_POLL_PERIOD_MS: u32 = 750;
const CREATE_LINK_FRESHNESS_TIMEOUT_MS: u32 = 1_000;
// Accelerometer-derived tilt is contaminated by short acceleration and dock-ramp
// impacts. Keep impact detection immediate, but require the gravity-vector tilt
// threshold to remain crossed before turning it into a latched motion veto.
const IMU_TILT_LATCH_HOLD_MS: u32 = 100;
const CAREFUL_MODE_MIN_TTL_MS: u32 = 250;
const CAREFUL_MODE_MAX_TTL_MS: u32 = 15_000;

#[derive(Clone, Copy, Eq, PartialEq)]
enum RuntimeMode {
    Running,
    Idle,
    Error,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum SafetyResponse {
    Stop,
    ContactWithdrawal,
}

fn healthy_supervision_lights(phase: u8) -> (u8, u8, u8, u32) {
    let led_bits = if (phase / 8) & 1 == 0 {
        CREATE_LED_PLAY
    } else {
        CREATE_LED_ADVANCE
    };
    (
        led_bits,
        CONNECTED_POWER_LED_COLOR,
        CONNECTED_POWER_LED_INTENSITY,
        HEALTHY_LIGHT_STEP_MS,
    )
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ActiveAction {
    None,
    PowerPulse {
        release_at_ms: u32,
        wake_wait_until_ms: Option<u32>,
        power_on: bool,
    },
    WakeSettle {
        until_ms: u32,
    },
    WaitForCreate {
        deadline_ms: u32,
        next_probe_ms: u32,
        response_bytes: u8,
        oi_started: bool,
        allow_power_toggle_on_timeout: bool,
    },
    Settle {
        until_ms: u32,
    },
    Driving {
        stop_at_ms: u32,
    },
    DockDeparture {
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
    safety_recovery: bool,
}

impl QueuedCommand {
    fn new(command_id: u32, command: RuntimeCommand) -> Self {
        Self {
            command_id,
            command,
            safety_recovery: false,
        }
    }

    fn safety_recovery(command_id: u32, command: RuntimeCommand) -> Self {
        Self {
            command_id,
            command,
            safety_recovery: true,
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct MotherbrainResetIdentity {
    session_hash: u32,
    lease_hash: u32,
    command_id: u32,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum MotherbrainResetOutcome {
    Refused(status::MotherbrainResetRefusal),
    Asserted,
    Completed,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct MotherbrainResetRecord {
    identity: MotherbrainResetIdentity,
    outcome: MotherbrainResetOutcome,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct ActiveMotherbrainReset {
    identity: MotherbrainResetIdentity,
    release_at_ms: u32,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct ActiveContactWithdrawal {
    started_at_ms: u32,
    baseline_odometry_mm: i32,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct ActiveVelocity {
    linear_mm_s: i16,
    angular_mrad_s: i16,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct ActiveEscape {
    kind: SafetyLatchKind,
    hazard_generation: u32,
    linear_mm_s: i16,
    angular_mrad_s: i16,
    ttl_ms: u32,
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
    active_velocity: Option<ActiveVelocity>,
    active_escape: Option<ActiveEscape>,
    stop_sent: bool,
    heartbeat_stop_at_ms: Option<u32>,
    careful_mode_until_ms: Option<u32>,
    sensor_stream: Option<SensorStream>,
    next_charging_sources_poll_ms: u32,
    next_complete_sensor_poll_ms: u32,
    next_imu_poll_ms: u32,
    next_full_mode_refresh_ms: u32,
    next_supervision_light_ms: u32,
    supervision_light_phase: u8,
    safety_latched: bool,
    safety_latch_kind: Option<status::SafetyEventKind>,
    dock_departure_pending: bool,
    charging_interlock_latched: bool,
    chirps: [[SongTone; MAX_SONG_TONES]; FEEDBACK_KIND_COUNT],
    chirp_counts: [u8; FEEDBACK_KIND_COUNT],
    audio: AudioAnnunciator,
    song_durations_ms: [u32; 16],
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
    safety_observation_initialized: bool,
    tilt_observed_since_ms: Option<u32>,
    active_motherbrain_reset: Option<ActiveMotherbrainReset>,
    motherbrain_reset_cooldown_until_ms: u32,
    motherbrain_reset_hardware_enabled: bool,
    motherbrain_reset_history: [Option<MotherbrainResetRecord>; MOTHERBRAIN_RESET_HISTORY_CAPACITY],
    motherbrain_reset_history_next: usize,
    safety_recovery_motion: bool,
    active_contact_withdrawal: Option<ActiveContactWithdrawal>,
    last_contact_withdrawal_at_ms: Option<u32>,
    repeated_contact_count: u8,
    last_observed_uart_rx_packets: u32,
    last_create_packet_at_ms: Option<u32>,
    low_battery_active: bool,
    charging_active: bool,
    imu_recovery_since_ms: Option<u32>,
    motion_inconsistency_cooldown_until_ms: u32,
    docking_active: bool,
    last_dock_ir: u8,
    restart_create_pending: bool,
    create_full_ready: bool,
    ever_create_full_ready: bool,
    imu_fault_active: bool,
    last_motion_inconsistent: bool,
}

impl<H> Runtime<H>
where
    H: BrainstemHardware,
{
    pub fn new(hardware: H) -> Self {
        let mut events = Deque::new();
        let observed_uart_rx_packets = status::snapshot(0).uart_rx_packets;
        status::signal_event(&BrainstemEvent::Boot);
        let _ = events.push_back(BrainstemEvent::Boot);
        status::set_runtime_state(RuntimeState::Booting);
        status::set_body_state(BodyState::NotStarted);
        status::set_careful_mode_until(None);
        status::set_audio_silent(false);
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
            active_velocity: None,
            active_escape: None,
            stop_sent: false,
            heartbeat_stop_at_ms: None,
            careful_mode_until_ms: None,
            sensor_stream: None,
            next_charging_sources_poll_ms: 0,
            next_complete_sensor_poll_ms: 0,
            next_imu_poll_ms: 0,
            next_full_mode_refresh_ms: 0,
            next_supervision_light_ms: 0,
            supervision_light_phase: 0,
            safety_latched: false,
            safety_latch_kind: None,
            dock_departure_pending: false,
            charging_interlock_latched: false,
            chirps: [[SongTone::default(); MAX_SONG_TONES]; FEEDBACK_KIND_COUNT],
            chirp_counts: [0; FEEDBACK_KIND_COUNT],
            audio: AudioAnnunciator::new(),
            song_durations_ms: [0; 16],
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
            safety_observation_initialized: false,
            tilt_observed_since_ms: None,
            active_motherbrain_reset: None,
            motherbrain_reset_cooldown_until_ms: 0,
            motherbrain_reset_hardware_enabled: body::MOTHERBRAIN_RESET_ENABLED,
            motherbrain_reset_history: [None; MOTHERBRAIN_RESET_HISTORY_CAPACITY],
            motherbrain_reset_history_next: 0,
            safety_recovery_motion: false,
            active_contact_withdrawal: None,
            last_contact_withdrawal_at_ms: None,
            repeated_contact_count: 0,
            last_observed_uart_rx_packets: observed_uart_rx_packets,
            last_create_packet_at_ms: None,
            low_battery_active: false,
            charging_active: false,
            imu_recovery_since_ms: None,
            motion_inconsistency_cooldown_until_ms: 0,
            docking_active: false,
            last_dock_ir: 0,
            restart_create_pending: false,
            create_full_ready: false,
            ever_create_full_ready: false,
            imu_fault_active: false,
            last_motion_inconsistent: false,
        }
    }

    pub fn run(mut self) -> ! {
        self.start();
        loop {
            self.tick();
            self.hardware.delay_ms(RUNTIME_TICK_MS);
        }
    }

    pub(crate) fn start(&mut self) {
        self.leds.boot_indicator(&mut self.hardware);
        self.queue_create_acquisition(0);
        status::set_runtime_state(RuntimeState::Running);
    }

    pub(crate) fn shutdown(&mut self) {
        self.interrupt_active_command();
        self.commands.clear();
        self.active = ActiveAction::None;
        self.heartbeat_stop_at_ms = None;
        self.cancel_careful_mode();
        let _ = self.stop_drive();
        status::set_command(None);
        status::set_runtime_state(RuntimeState::Idle);
        status::set_body_state(BodyState::Idle);
    }

    #[allow(dead_code)]
    pub fn enqueue_command(&mut self, command: RuntimeCommand) -> Result<(), RuntimeCommand> {
        self.commands
            .push_back(QueuedCommand::new(0, command))
            .map_err(|queued| queued.command)
    }

    pub fn tick(&mut self) {
        status::set_runtime_action(self.active_action_code());
        self.poll();
        self.poll_motherbrain_reset();
        self.hardware.feed_watchdog();
        self.poll_charging_indicator();
        self.poll_imu();
        self.observe_audio_transitions();
        if let Err(error) = self.poll_sensor_stream() {
            self.enter_error(error);
            return;
        }
        if let Err(error) = self.enforce_careful_mode_timeout() {
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
        self.publish_safety_snapshot();
        if let Err(error) = self.maintain_full_mode() {
            self.enter_error(error);
            return;
        }
        if let Err(error) = self.animate_supervision_lights() {
            self.enter_error(error);
            return;
        }
        if status::take_expired_authority(self.now_ms()) {
            self.request_audio(AuditoryCue::AuthorityLost);
            self.heartbeat_stop_at_ms = None;
            self.cancel_careful_mode();
            if self.active_contact_withdrawal.is_none() {
                self.interrupt_active_command();
                self.commands.clear();
                self.active = ActiveAction::None;
                let _ = self.stop_drive();
            }
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
        self.poll_audio();
    }

    fn poll(&mut self) {
        self.poll_session_replace();
        self.poll_authority_transition();
        self.timers.poll(&mut self.hardware, &mut self.events);
        self.create_uart.poll(&mut self.hardware, &mut self.events);
        let now_ms = self.now_ms();
        let snapshot = status::snapshot(now_ms);
        let was_create_responsive = self.create_responsive;
        if snapshot.uart_rx_packets != self.last_observed_uart_rx_packets {
            self.last_observed_uart_rx_packets = snapshot.uart_rx_packets;
            self.last_create_packet_at_ms = Some(now_ms);
        }
        let create_link_fresh = self.last_create_packet_at_ms.is_some_and(|last_packet_ms| {
            now_ms.wrapping_sub(last_packet_ms) <= CREATE_LINK_FRESHNESS_TIMEOUT_MS
        });
        if matches!(snapshot.oi_mode, 1..=3) && create_link_fresh {
            self.create_responsive = true;
            status::set_create_power_on(true);
        } else if self.last_create_packet_at_ms.is_some() && !create_link_fresh {
            // OI mode and power state are observations, not durable promises.
            // Once the Create stops answering, invalidate the cached mode so
            // active output is stopped by Full-mode supervision and later
            // motion cannot start against a dead link.
            self.create_responsive = false;
            self.cancel_careful_mode();
            status::set_oi_mode_unknown();
        }
        if was_create_responsive && !self.create_responsive {
            self.create_full_ready = false;
            self.request_audio(AuditoryCue::CreateError);
        }
        self.poll_control_command();
    }

    fn poll_authority_transition(&mut self) {
        let Some(generation) = status::pending_authority_transition() else {
            return;
        };
        let now_ms = self.now_ms();
        if status::pending_authority_continues_owner(now_ms) {
            status::acknowledge_authority_transition(generation);
            return;
        }
        let replacing = status::has_active_authority(now_ms);
        self.heartbeat_stop_at_ms = None;
        self.cancel_careful_mode();
        if self.active_contact_withdrawal.is_none() {
            self.interrupt_active_command();
            self.commands.clear();
            self.active = ActiveAction::None;
            let _ = self.stop_drive();
            status::set_command(None);
            status::set_runtime_state(RuntimeState::Idle);
            status::set_body_state(BodyState::Idle);
        }
        status::acknowledge_authority_transition(generation);
        self.request_audio(if replacing {
            AuditoryCue::AuthorityReplaced
        } else {
            AuditoryCue::AuthorityAcquired
        });
    }

    fn queue_create_acquisition(&mut self, command_id: u32) {
        for command in ACQUIRE_CREATE_SCRIPT {
            let _ = self
                .commands
                .push_back(QueuedCommand::new(command_id, *command));
        }
        self.mode = RuntimeMode::Running;
    }

    fn maintain_full_mode(&mut self) -> Result<(), BrainstemError> {
        let now_ms = self.now_ms();
        let snapshot = status::snapshot(now_ms);
        if !time_reached(now_ms, self.next_full_mode_refresh_ms) {
            return Ok(());
        }
        let motor_output_active = matches!(
            self.active,
            ActiveAction::Driving { .. } | ActiveAction::DockDeparture { .. }
        );
        if motor_output_active {
            if snapshot.oi_mode != 3 {
                self.stop_drive()?;
                self.interrupt_active_command();
                self.active = ActiveAction::None;
                return Err(BrainstemError::CreateNoResponse);
            }
            // Re-sending OI Full zeros wheel output on Create 1 even when the
            // reported mode is already Full. Never overlay that supervision
            // write on a bounded motor program; fresh mode loss still takes
            // the fail-closed branch above.
            self.next_full_mode_refresh_ms = now_ms.wrapping_add(FULL_MODE_REFRESH_PERIOD_MS);
            return Ok(());
        }
        if low_battery_and_charging(&snapshot) && snapshot.oi_mode == 3 {
            return Ok(());
        }

        // RX health is evidence, not permission to transmit. If the Create has
        // rebooted, gone passive, or our wake probe was wrong, START + FULL is
        // the idempotent assertion that lets the brainstem regain control.
        if !self.create_responsive || snapshot.oi_mode != 3 {
            self.create_uart
                .start_oi(&mut self.hardware, &mut self.events)?;
        }
        if snapshot.oi_mode == 3 {
            self.create_uart.refresh_full_mode(&mut self.hardware)?;
        } else {
            self.create_uart.set_mode(
                &mut self.hardware,
                &mut self.events,
                crate::commands::CreateOiMode::Full,
            )?;
        }
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
                if on { CREATE_BUTTON_LED_MASK } else { 0 },
                255,
                if on { 255 } else { 0 },
                300,
            )
        } else if self.estop_latched || self.safety_latched || self.charging_interlock_latched {
            (CREATE_BUTTON_LED_MASK, 255, 255, 500)
        } else {
            // Keep POWER stable while PLAY and ADVANCE alternate more quickly.
            healthy_supervision_lights(self.supervision_light_phase)
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
        self.heartbeat_stop_at_ms = None;
        self.cancel_careful_mode();
        self.sensor_stream = None;
        network_registry::clear_motherbrain_registration();
        if self.active_contact_withdrawal.is_none() {
            self.interrupt_active_command();
            self.commands.clear();
            self.active = ActiveAction::None;
            let _ = self.stop_drive();
            status::set_command(None);
            status::set_runtime_state(RuntimeState::Idle);
            status::set_body_state(BodyState::Idle);
        }
        self.publish_safety_snapshot();
        // The session module supplies the pending hash before requesting the
        // barrier. Until it is wired, generation itself is a fail-closed token.
        status::acknowledge_session_replace(generation, status::pending_session_hash());
    }

    fn publish_safety_snapshot(&self) {
        status::set_session_safety_snapshot(
            self.estop_latched,
            self.safety_latched,
            self.charging_interlock_latched,
            self.safety_latch_kind,
        );
    }

    fn poll_control_command(&mut self) {
        let Some(command) = status::take_control_command() else {
            return;
        };
        let command_id = status::last_dispatched_command_id();
        let (service_session_hash, service_lease_hash) = status::last_dispatched_service_identity();

        if self.active_contact_withdrawal.is_some() && !command_preempts_contact_withdrawal(command)
        {
            // The possessor may lose or replace authority while this runs.
            // Ordinary commands cannot supersede a local reflex.
            status::mark_command_interrupted(command_id);
            return;
        }

        if self.active_contact_withdrawal.is_some() && command_preempts_contact_withdrawal(command)
        {
            let stopped = self.stop_drive().is_ok();
            self.finish_contact_withdrawal(
                status::ContactWithdrawalOutcome::SafetyPreempted,
                matches!(command, BrainstemCommand::EStop)
                    .then_some(status::SafetyEventKind::EStop),
                stopped,
            );
        }

        match command {
            BrainstemCommand::Stop | BrainstemCommand::EStop => {
                self.docking_active = false;
                self.interrupt_active_command();
                self.commands.clear();
                self.active = ActiveAction::None;
                self.heartbeat_stop_at_ms = None;
                if matches!(command, BrainstemCommand::EStop) {
                    self.cancel_careful_mode();
                }
                let command = match command {
                    BrainstemCommand::Stop => RuntimeCommand::Stop,
                    BrainstemCommand::EStop => RuntimeCommand::EStop,
                    _ => unreachable!(),
                };
                let _ = self
                    .commands
                    .push_front(QueuedCommand::new(command_id, command));
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
                self.cancel_careful_mode();
                for command in DISARM_SCRIPT.iter().rev() {
                    let _ = self
                        .commands
                        .push_front(QueuedCommand::new(command_id, *command));
                }
                self.mode = RuntimeMode::Running;
            }
            BrainstemCommand::RestartCreate => {
                self.restart_create_pending = true;
                self.interrupt_active_command();
                self.commands.clear();
                self.active = ActiveAction::None;
                self.heartbeat_stop_at_ms = None;
                self.cancel_careful_mode();
                for command in RESTART_CREATE_SCRIPT.iter().rev() {
                    let _ = self
                        .commands
                        .push_front(QueuedCommand::new(command_id, *command));
                }
                self.mode = RuntimeMode::Running;
                status::set_runtime_state(RuntimeState::Running);
            }
            BrainstemCommand::ResetMotherbrain => {
                self.request_motherbrain_reset(MotherbrainResetIdentity {
                    session_hash: service_session_hash,
                    lease_hash: service_lease_hash,
                    command_id,
                });
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
            BrainstemCommand::EscapeMotion { .. } => {
                if let Some(command) = runtime_command_from_forebrain(command) {
                    let pending = self.commands.len();
                    for _ in 0..pending {
                        let Some(existing) = self.commands.pop_front() else {
                            break;
                        };
                        if matches!(existing.command, RuntimeCommand::EscapeMotion { .. }) {
                            status::mark_command_interrupted(existing.command_id);
                        } else {
                            let _ = self.commands.push_back(existing);
                        }
                    }
                    let _ = self
                        .commands
                        .push_back(QueuedCommand::safety_recovery(command_id, command));
                }
                if self.mode == RuntimeMode::Idle || self.mode == RuntimeMode::Error {
                    self.mode = RuntimeMode::Running;
                    status::set_runtime_state(RuntimeState::Running);
                }
            }
            _ => {
                if let Some(command) = runtime_command_from_forebrain(command) {
                    let _ = self
                        .commands
                        .push_back(QueuedCommand::new(command_id, command));
                }
                if self.mode == RuntimeMode::Idle || self.mode == RuntimeMode::Error {
                    self.mode = RuntimeMode::Running;
                    status::set_runtime_state(RuntimeState::Running);
                }
            }
        }
    }

    fn request_motherbrain_reset(&mut self, identity: MotherbrainResetIdentity) {
        let now_ms = self.now_ms();
        status::mark_motherbrain_reset_requested(
            identity.command_id,
            identity.session_hash,
            identity.lease_hash,
        );

        if let Some(record) = self
            .motherbrain_reset_history
            .iter()
            .flatten()
            .find(|record| record.identity == identity)
            .copied()
        {
            Self::replay_motherbrain_reset_outcome(record);
            return;
        }

        let refusal = if identity.command_id == 0 {
            Some(status::MotherbrainResetRefusal::InvalidCommandId)
        } else if !self.motherbrain_reset_hardware_enabled {
            Some(status::MotherbrainResetRefusal::HardwareDisabled)
        } else if !status::active_service_authority_matches(
            identity.session_hash,
            identity.lease_hash,
            now_ms,
            MOTHERBRAIN_RESET_SERVICE_SCOPE,
        ) {
            Some(status::MotherbrainResetRefusal::InvalidServiceAuthority)
        } else if self.active_motherbrain_reset.is_some()
            || !time_reached(now_ms, self.motherbrain_reset_cooldown_until_ms)
        {
            Some(status::MotherbrainResetRefusal::Cooldown)
        } else {
            let snapshot = status::snapshot(now_ms);
            let stopped = snapshot.body_state == BodyState::Idle as u8
                && self.active == ActiveAction::None
                && self.commands.is_empty()
                && self.heartbeat_stop_at_ms.is_none();
            let disarmed = snapshot.oi_mode == 1;
            (!stopped || !disarmed).then_some(status::MotherbrainResetRefusal::UnsafeState)
        };

        if let Some(reason) = refusal {
            let record = MotherbrainResetRecord {
                identity,
                outcome: MotherbrainResetOutcome::Refused(reason),
            };
            self.remember_motherbrain_reset(record);
            Self::replay_motherbrain_reset_outcome(record);
            self.request_audio(AuditoryCue::ServiceFailure);
            return;
        }

        self.hardware.set_motherbrain_reset(true);
        self.active_motherbrain_reset = Some(ActiveMotherbrainReset {
            identity,
            release_at_ms: now_ms.wrapping_add(MOTHERBRAIN_RESET_PULSE_MS),
        });
        self.motherbrain_reset_cooldown_until_ms =
            now_ms.wrapping_add(MOTHERBRAIN_RESET_COOLDOWN_MS);
        let record = MotherbrainResetRecord {
            identity,
            outcome: MotherbrainResetOutcome::Asserted,
        };
        self.remember_motherbrain_reset(record);
        Self::replay_motherbrain_reset_outcome(record);
    }

    fn poll_motherbrain_reset(&mut self) {
        let Some(active) = self.active_motherbrain_reset else {
            return;
        };
        if time_reached(self.now_ms(), active.release_at_ms) {
            self.hardware.set_motherbrain_reset(false);
            self.active_motherbrain_reset = None;
            let record = MotherbrainResetRecord {
                identity: active.identity,
                outcome: MotherbrainResetOutcome::Completed,
            };
            self.remember_motherbrain_reset(record);
            Self::replay_motherbrain_reset_outcome(record);
            self.request_audio(AuditoryCue::ServiceComplete);
        }
    }

    fn remember_motherbrain_reset(&mut self, record: MotherbrainResetRecord) {
        if let Some(existing) = self
            .motherbrain_reset_history
            .iter_mut()
            .flatten()
            .find(|existing| existing.identity == record.identity)
        {
            *existing = record;
            return;
        }
        self.motherbrain_reset_history[self.motherbrain_reset_history_next] = Some(record);
        self.motherbrain_reset_history_next =
            (self.motherbrain_reset_history_next + 1) % MOTHERBRAIN_RESET_HISTORY_CAPACITY;
    }

    fn replay_motherbrain_reset_outcome(record: MotherbrainResetRecord) {
        let identity = record.identity;
        match record.outcome {
            MotherbrainResetOutcome::Refused(reason) => status::mark_motherbrain_reset_refused(
                reason,
                identity.session_hash,
                identity.lease_hash,
            ),
            MotherbrainResetOutcome::Asserted => status::mark_motherbrain_reset_asserted(
                identity.command_id,
                identity.session_hash,
                identity.lease_hash,
            ),
            MotherbrainResetOutcome::Completed => status::mark_motherbrain_reset_completed(
                identity.command_id,
                identity.session_hash,
                identity.lease_hash,
            ),
        }
    }

    fn enqueue_latest_velocity(&mut self, command_id: u32, command: RuntimeCommand) {
        let RuntimeCommand::CmdVel {
            linear_mm_s,
            angular_mrad_s,
            duration_ms: Some(duration_ms),
        } = command
        else {
            let _ = self
                .commands
                .push_back(QueuedCommand::new(command_id, command));
            return;
        };

        let pending = self.commands.len();
        for _ in 0..pending {
            let Some(existing) = self.commands.pop_front() else {
                break;
            };
            if !matches!(existing.command, RuntimeCommand::CmdVel { .. }) {
                let _ = self.commands.push_back(existing);
            } else if existing.command_id != command_id {
                // A newer velocity command has consumed this queued command
                // before it could start. Keep its accepted lifecycle closed
                // even though no motor write was ever issued for it.
                status::mark_command_interrupted(existing.command_id);
            }
        }

        let velocity = ActiveVelocity {
            linear_mm_s,
            angular_mrad_s,
        };
        if matches!(self.active, ActiveAction::Driving { .. })
            && self.active_velocity == Some(velocity)
        {
            // Possession refreshes cmd_vel every control tick.  Restarting the
            // same drive on every refresh makes the Create brake and re-start
            // continuously. Renew its deadline without touching the motor or
            // transferring lifecycle ownership. The ingress lane records the
            // refresh as a compact CommandRenewed event; a changed velocity
            // still preempts immediately below.
            self.active = ActiveAction::Driving {
                stop_at_ms: self.now_ms().wrapping_add(duration_ms),
            };
        } else if matches!(self.active, ActiveAction::Driving { .. }) {
            self.interrupt_active_command();
            self.active = ActiveAction::None;
            let _ = self.commands.push_front(QueuedCommand::new(
                command_id,
                RuntimeCommand::CmdVel {
                    linear_mm_s,
                    angular_mrad_s,
                    duration_ms: Some(duration_ms),
                },
            ));
        } else {
            let _ = self.commands.push_back(QueuedCommand::new(
                command_id,
                RuntimeCommand::CmdVel {
                    linear_mm_s,
                    angular_mrad_s,
                    duration_ms: Some(duration_ms),
                },
            ));
        }
    }

    fn start_next_command(&mut self) -> Result<(), BrainstemError> {
        let Some(queued) = self.commands.pop_front() else {
            status::set_command(None);
            return Ok(());
        };
        let command = queued.command;
        let now_ms = self.now_ms();
        if self.dock_departure_pending && requires_dock_departure(command) {
            // Full mode terminates Create 1 charging.  Once the charge signal
            // drops, back off the Home Base before starting the caller's
            // body-neutral motion command.  Keep that command queued and do
            // not give the internal departure its lifecycle identity.
            if status::charging_interlock_active(&status::snapshot(now_ms)) {
                let _ = self.commands.push_front(queued);
                return Ok(());
            }
            let _ = self.commands.push_front(queued);
            self.start_dock_departure(now_ms)?;
            return Ok(());
        }
        let command_code = status::set_command(Some(command));
        self.active_command_id = Some(queued.command_id);
        self.safety_recovery_motion = queued.safety_recovery;
        status::mark_command_started(queued.command_id, command_code);
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
                self.request_audio(AuditoryCue::EStop);
                self.active = ActiveAction::None;
            }
            RuntimeCommand::ClearEStop => {
                self.estop_latched = false;
                status::mark_estop_cleared();
                status::mark_safety_cleared(status::SafetyEventKind::EStop);
                self.request_audio(AuditoryCue::SafetyClear);
            }
            RuntimeCommand::WakeCreate => {
                self.create_responsive = false;
                status::set_oi_mode_unknown();
                status::set_body_state(BodyState::WaitingForCreate);
                match status::known_create_power_state(status::snapshot(now_ms).create_power_state)
                {
                    Some(false) => {
                        self.push_event(BrainstemEvent::CreatePowerOnRequested);
                        self.hardware.begin_power_toggle_pulse();
                        self.active = ActiveAction::PowerPulse {
                            release_at_ms: now_ms.wrapping_add(body::POWER_TOGGLE_PULSE_MS),
                            wake_wait_until_ms: Some(
                                now_ms.wrapping_add(body::CREATE_WAKE_WAIT_MS),
                            ),
                            power_on: true,
                        };
                    }
                    known_state => {
                        self.active = ActiveAction::WaitForCreate {
                            deadline_ms: now_ms.wrapping_add(body::CREATE_RESPONSIVE_TIMEOUT_MS),
                            next_probe_ms: now_ms,
                            response_bytes: 0,
                            oi_started: false,
                            // UNKNOWN gets one documented best-effort pulse
                            // after a full probe timeout. Known ON is probe-only:
                            // an RX failure must never toggle a running Create off.
                            allow_power_toggle_on_timeout: known_state.is_none(),
                        };
                    }
                };
            }
            RuntimeCommand::SleepCreate => {
                self.create_responsive = false;
                status::set_oi_mode_unknown();
                self.stop_drive()?;
                match status::known_create_power_state(status::snapshot(now_ms).create_power_state)
                {
                    Some(false) => {
                        // Pin 3 is a toggle, so sleeping an already-OFF Create
                        // succeeds without touching POWER_TOGGLE.
                        status::set_body_state(BodyState::Idle);
                        self.active = ActiveAction::None;
                    }
                    Some(true) => {
                        status::set_body_state(BodyState::PowerCycling);
                        self.push_event(BrainstemEvent::CreatePowerOffRequested);
                        self.hardware.begin_power_toggle_pulse();
                        self.active = ActiveAction::PowerPulse {
                            release_at_ms: now_ms.wrapping_add(body::POWER_TOGGLE_PULSE_MS),
                            wake_wait_until_ms: None,
                            power_on: false,
                        };
                    }
                    None => {
                        // Refuse an ambiguous toggle after stopping output. The
                        // command is reported interrupted, not completed.
                        status::set_body_state(BodyState::Idle);
                        self.refuse_active_command();
                        self.active = ActiveAction::None;
                    }
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
                if linear_mm_s != 0 || angular_mrad_s != 0 {
                    status::mark_velocity_stream_active(
                        queued.command_id,
                        linear_mm_s,
                        angular_mrad_s,
                    );
                }
            }
            RuntimeCommand::ClearSafetyLatch { kind } => {
                self.clear_safety_latch(Some(safety_latch_kind_to_event(kind)));
            }
            RuntimeCommand::CarefulMode { ttl_ms } => {
                self.enter_careful_mode(ttl_ms)?;
            }
            RuntimeCommand::EscapeMotion {
                kind,
                hazard_generation,
                linear_mm_s,
                angular_mrad_s,
                ttl_ms,
            } => {
                status::validate_escape_motion(
                    kind,
                    hazard_generation,
                    linear_mm_s,
                    angular_mrad_s,
                    ttl_ms,
                )
                .map_err(|_| BrainstemError::CreateNoResponse)?;
                self.active_escape = Some(ActiveEscape {
                    kind,
                    hazard_generation,
                    linear_mm_s,
                    angular_mrad_s,
                    ttl_ms,
                });
                self.start_cmd_vel(linear_mm_s, angular_mrad_s, Some(ttl_ms), now_ms)?;
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
                self.song_durations_ms[feedback_slot(kind) as usize] =
                    tone_duration_ms(&tones, self.chirp_counts[index]);
                if self.create_responsive
                    && self
                        .create_uart
                        .define_song(
                            &mut self.hardware,
                            &mut self.events,
                            feedback_slot(kind),
                            &self.chirps[index],
                            self.chirp_counts[index],
                        )
                        .is_err()
                {
                    status::increment_audio_dropped_or_replaced(1);
                }
            }
            RuntimeCommand::PlayFeedback { kind } => {
                if self.audio.silent() {
                    status::increment_audio_suppressed_by_silent();
                    self.active = ActiveAction::None;
                    self.complete_active_command();
                    return Ok(());
                }
                if !self.create_responsive {
                    self.active = ActiveAction::None;
                    self.complete_active_command();
                    return Ok(());
                }
                let (tones, tone_count) = self.feedback_tones(kind);
                if !self.audio.playback_available(now_ms) {
                    status::increment_audio_dropped_or_replaced(1);
                } else if self
                    .create_uart
                    .define_song(
                        &mut self.hardware,
                        &mut self.events,
                        feedback_slot(kind),
                        &tones,
                        tone_count,
                    )
                    .and_then(|()| {
                        self.create_uart.play_song(
                            &mut self.hardware,
                            &mut self.events,
                            feedback_slot(kind),
                        )
                    })
                    .is_err()
                {
                    status::increment_audio_dropped_or_replaced(1);
                } else {
                    self.audio
                        .mark_manual_played(now_ms, tone_duration_ms(&tones, tone_count));
                }
            }
            RuntimeCommand::SetAudioSilent { silent } => {
                self.set_audio_silent(silent);
            }
            RuntimeCommand::CalibrateTurn {
                angular_mrad_s,
                duration_ms,
            } => self.start_cmd_vel(0, angular_mrad_s, Some(duration_ms), now_ms)?,
            RuntimeCommand::OrientationProbe {
                angular_mrad_s,
                duration_ms,
            } => self.start_orientation_probe(angular_mrad_s, duration_ms, now_ms)?,
            RuntimeCommand::ResetOdometry => {
                status::mark_odometry_reset();
            }
            RuntimeCommand::ZeroImuOrientation => {
                if status::zero_imu_orientation_from_gravity()
                    && self.safety_latch_kind == Some(status::SafetyEventKind::Tilt)
                {
                    self.clear_safety_latch(Some(status::SafetyEventKind::Tilt));
                }
            }
            RuntimeCommand::ClearImuOrientation => {
                status::clear_imu_orientation_calibration();
            }
            RuntimeCommand::SongPlay { id } => {
                if self.audio.silent() {
                    status::increment_audio_suppressed_by_silent();
                } else if !self.audio.playback_available(now_ms) {
                    status::increment_audio_dropped_or_replaced(1);
                } else if self.create_responsive
                    && self
                        .create_uart
                        .play_song(&mut self.hardware, &mut self.events, id)
                        .is_err()
                {
                    status::increment_audio_dropped_or_replaced(1);
                } else if self.create_responsive {
                    let duration_ms = self
                        .song_durations_ms
                        .get(id as usize)
                        .copied()
                        .unwrap_or(0)
                        .max(1_000);
                    self.audio.mark_manual_played(now_ms, duration_ms);
                }
            }
            RuntimeCommand::SongDefine {
                id,
                tones,
                tone_count,
            } => {
                if let Some(duration) = self.song_durations_ms.get_mut(id as usize) {
                    *duration = tone_duration_ms(&tones, tone_count);
                }
                if self.create_responsive
                    && self
                        .create_uart
                        .define_song(&mut self.hardware, &mut self.events, id, &tones, tone_count)
                        .is_err()
                {
                    status::increment_audio_dropped_or_replaced(1);
                }
            }
            RuntimeCommand::Dock => {
                self.ensure_create_responsive()?;
                self.create_uart
                    .seek_dock(&mut self.hardware, &mut self.events)?;
                self.docking_active = true;
                self.last_dock_ir = status::snapshot(now_ms).create_sensor_ir_byte;
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
                    self.hardware.end_power_toggle_pulse();
                    self.push_event(BrainstemEvent::CreatePowerToggled);
                    if wake_wait_until_ms.is_none() {
                        status::set_create_power_on(power_on);
                    }
                    self.active = match wake_wait_until_ms {
                        Some(until_ms) => ActiveAction::WakeSettle { until_ms },
                        None => ActiveAction::None,
                    };
                    if self.active == ActiveAction::None {
                        self.complete_active_command();
                    }
                }
                Ok(())
            }
            ActiveAction::Settle { until_ms } => {
                if time_reached(now_ms, until_ms) {
                    self.active = ActiveAction::None;
                    self.complete_active_command();
                }
                Ok(())
            }
            ActiveAction::WakeSettle { until_ms } => {
                if time_reached(now_ms, until_ms) {
                    self.active = ActiveAction::WaitForCreate {
                        deadline_ms: now_ms.wrapping_add(body::CREATE_RESPONSIVE_TIMEOUT_MS),
                        next_probe_ms: now_ms,
                        response_bytes: 0,
                        oi_started: false,
                        allow_power_toggle_on_timeout: false,
                    };
                }
                Ok(())
            }
            ActiveAction::WaitForCreate {
                deadline_ms,
                next_probe_ms,
                mut response_bytes,
                oi_started,
                allow_power_toggle_on_timeout,
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
                    if allow_power_toggle_on_timeout {
                        self.push_event(BrainstemEvent::CreatePowerOnRequested);
                        self.hardware.begin_power_toggle_pulse();
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
                    // Do not silently escalate a failed probe into another
                    // power cycle. An explicit service-scoped restart remains
                    // available to an attended diagnostic operator.
                    self.active = ActiveAction::None;
                    self.fail_active_command(BrainstemError::Timeout);
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
                            allow_power_toggle_on_timeout,
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
                        allow_power_toggle_on_timeout,
                    };
                } else {
                    self.active = ActiveAction::WaitForCreate {
                        deadline_ms,
                        next_probe_ms,
                        response_bytes,
                        oi_started,
                        allow_power_toggle_on_timeout,
                    };
                }
                Ok(())
            }
            ActiveAction::Driving { stop_at_ms } => {
                if time_reached(now_ms, stop_at_ms) {
                    self.stop_drive()?;
                    self.active = ActiveAction::None;
                    self.finish_contact_withdrawal(
                        status::ContactWithdrawalOutcome::Completed,
                        None,
                        true,
                    );
                    self.complete_active_command();
                }
                Ok(())
            }
            ActiveAction::DockDeparture { stop_at_ms } => {
                if time_reached(now_ms, stop_at_ms) {
                    self.stop_drive()?;
                    self.active = ActiveAction::None;
                    status::set_body_state(BodyState::Idle);
                }
                Ok(())
            }
        }
    }

    fn start_dock_departure(&mut self, now_ms: u32) -> Result<(), BrainstemError> {
        self.ensure_create_responsive()?;
        self.dock_departure_pending = false;
        // Dock departure is a fixed, body-local 1.5 second operation. A
        // browser motion heartbeat is shorter (900 ms) and supervises the
        // caller's primitive, not this bounded transition. Clear its deadline
        // before starting so the watchdog cannot cancel the reviewed undock.
        self.heartbeat_stop_at_ms = None;
        self.active_velocity = None;
        status::set_body_state(BodyState::Moving);
        self.stop_sent = false;
        self.create_uart.drive_direct(
            &mut self.hardware,
            &mut self.events,
            DOCK_DEPARTURE_SPEED_MM_S,
            DOCK_DEPARTURE_SPEED_MM_S,
            DOCK_DEPARTURE_DURATION_MS,
        )?;
        self.active = ActiveAction::DockDeparture {
            stop_at_ms: now_ms.wrapping_add(DOCK_DEPARTURE_DURATION_MS),
        };
        Ok(())
    }

    fn stop_drive(&mut self) -> Result<(), BrainstemError> {
        self.create_uart
            .stop(&mut self.hardware, &mut self.events)?;
        self.stop_sent = true;
        self.active_velocity = None;
        self.active_escape = None;
        status::clear_velocity_stream();
        // Once STOP has been sent successfully there is no live motion for the
        // heartbeat watchdog to supervise. Leaving the old deadline armed
        // would later revoke an otherwise valid control lease after a normal
        // TTL-bounded motion pulse had already stopped.
        self.heartbeat_stop_at_ms = None;
        Ok(())
    }

    fn start_cmd_vel(
        &mut self,
        linear_mm_s: i16,
        angular_mrad_s: i16,
        duration_ms: Option<u32>,
        now_ms: u32,
    ) -> Result<(), BrainstemError> {
        if linear_mm_s == 0 && angular_mrad_s == 0 {
            self.stop_drive()?;
            self.active = ActiveAction::None;
            return Ok(());
        }
        let half_delta = angular_mrad_s as i32 * CREATE_AXLE_TRACK_MM / 2_000;
        let left = clamp_i16(linear_mm_s as i32 - half_delta);
        let right = clamp_i16(linear_mm_s as i32 + half_delta);
        self.start_drive_direct(left, right, duration_ms, now_ms)?;
        self.active_velocity = Some(ActiveVelocity {
            linear_mm_s,
            angular_mrad_s,
        });
        Ok(())
    }

    fn start_orientation_probe(
        &mut self,
        angular_mrad_s: i16,
        duration_ms: u32,
        now_ms: u32,
    ) -> Result<(), BrainstemError> {
        let angular_abs = abs_i32(angular_mrad_s as i32);
        if angular_abs == 0 || duration_ms == 0 {
            self.stop_drive()?;
            self.active = ActiveAction::None;
            return Ok(());
        }

        self.ensure_orientation_probe_allowed(now_ms)?;
        if !status::zero_imu_orientation_from_gravity() {
            self.stop_drive()?;
            return Err(BrainstemError::CreateNoResponse);
        }
        if self.safety_latch_kind == Some(status::SafetyEventKind::Tilt) {
            self.clear_safety_latch(Some(status::SafetyEventKind::Tilt));
        }
        self.ensure_orientation_probe_allowed(now_ms)?;

        status::mark_odometry_reset();
        self.start_cmd_vel(0, clamp_i16(angular_abs), Some(duration_ms), now_ms)
    }

    fn ensure_orientation_probe_allowed(&mut self, now_ms: u32) -> Result<(), BrainstemError> {
        if self.estop_latched
            || self.dock_departure_pending
            || self.charging_interlock_latched
            || (self.safety_latched
                && self.safety_latch_kind != Some(status::SafetyEventKind::Tilt))
        {
            self.stop_drive()?;
            return Err(BrainstemError::CreateNoResponse);
        }
        self.ensure_create_responsive()?;

        let snapshot = status::snapshot(now_ms);
        let flags = snapshot.create_sensor_flags;
        let wheel_drop = flags & (1 << 2) != 0;
        let cliff = flags & ((1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)) != 0;
        let imu_ready = body::IMU_ENABLED
            && snapshot.imu_health == status::ImuHealthCode::Ok as u8
            && snapshot.imu_sample_count > 0
            && snapshot.imu_sample_age_ms <= ORIENTATION_PROBE_IMU_MAX_AGE_MS
            && snapshot.imu_accel_magnitude_mm_s2 >= ORIENTATION_PROBE_MIN_ACCEL_MM_S2
            && snapshot.imu_accel_magnitude_mm_s2 <= ORIENTATION_PROBE_MAX_ACCEL_MM_S2;
        let imu_still_tilted = snapshot.imu_tilt_magnitude_mrad as i16 >= body::IMU_TILT_STOP_MRAD;
        let imu_impact = snapshot.imu_impact_score_mm_s2 >= body::IMU_IMPACT_STOP_MM_S2;
        if wheel_drop
            || cliff
            || status::charging_interlock_active(&snapshot)
            || !imu_ready
            || imu_impact
            || (imu_still_tilted && self.safety_latch_kind != Some(status::SafetyEventKind::Tilt))
        {
            self.stop_drive()?;
            return Err(BrainstemError::CreateNoResponse);
        }
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
        if matches!(
            self.active,
            ActiveAction::Driving { .. } | ActiveAction::DockDeparture { .. }
        ) {
            self.interrupt_active_command();
            self.stop_drive()?;
            self.active = ActiveAction::None;
        }
        Ok(())
    }

    fn poll_sensor_stream(&mut self) -> Result<(), BrainstemError> {
        let now_ms = self.now_ms();
        if !self.create_responsive {
            return Ok(());
        }

        if time_reached(now_ms, self.next_charging_sources_poll_ms) {
            self.create_uart
                .request_sensor_packet(&mut self.hardware, CREATE_CHARGING_SOURCES_PACKET)?;
            self.next_charging_sources_poll_ms =
                now_ms.wrapping_add(CREATE_CHARGING_SOURCES_POLL_PERIOD_MS);
            return Ok(());
        }

        if time_reached(now_ms, self.next_complete_sensor_poll_ms) {
            self.create_uart
                .request_sensor_packet(&mut self.hardware, CREATE_COMPLETE_SENSOR_PACKET)?;
            self.next_complete_sensor_poll_ms =
                now_ms.wrapping_add(CREATE_COMPLETE_SENSOR_POLL_PERIOD_MS);
            return Ok(());
        }

        let Some(mut stream) = self.sensor_stream else {
            return Ok(());
        };
        if time_reached(now_ms, stream.next_request_ms) {
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

    fn poll_charging_indicator(&mut self) {
        let active = if body::CREATE_CHARGING_INDICATOR_ENABLED {
            self.hardware.charging_indicator_active()
        } else {
            None
        };
        status::mark_create_charging_indicator(active);
    }

    fn observe_audio_transitions(&mut self) {
        const IMU_RECOVERY_STABLE_MS: u32 = 500;
        const MOTION_INCONSISTENCY_COOLDOWN_MS: u32 = 5_000;

        let now_ms = self.now_ms();
        let snapshot = status::snapshot(now_ms);
        let low_battery_threshold = if self.low_battery_active {
            LOW_BATTERY_AUDIO_CLEAR_PERCENT
        } else {
            LOW_BATTERY_PERCENT
        };
        let low_battery = snapshot.create_sensor_capacity_mah > 0
            && u32::from(snapshot.create_sensor_charge_mah) * 100
                <= u32::from(snapshot.create_sensor_capacity_mah) * low_battery_threshold;
        if low_battery && !self.low_battery_active {
            self.request_audio(AuditoryCue::LowBattery);
        }
        self.low_battery_active = low_battery;

        let charging = status::charging_interlock_active(&snapshot)
            || snapshot.create_sensor_charging_sources & 0b10 != 0;
        if charging && !self.charging_active {
            self.request_audio(AuditoryCue::DockContact);
            self.docking_active = false;
        }
        self.charging_active = charging;

        let imu_fault = matches!(
            snapshot.imu_health,
            x if x == status::ImuHealthCode::Fault as u8
                || x == status::ImuHealthCode::Absent as u8
        );
        if imu_fault && !self.imu_fault_active {
            self.request_audio(AuditoryCue::ImuFault);
            self.imu_fault_active = true;
            self.imu_recovery_since_ms = None;
        } else if self.imu_fault_active && snapshot.imu_health == status::ImuHealthCode::Ok as u8 {
            let recovery_since = *self.imu_recovery_since_ms.get_or_insert(now_ms);
            if now_ms.wrapping_sub(recovery_since) >= IMU_RECOVERY_STABLE_MS {
                self.request_audio(AuditoryCue::Recovery);
                self.imu_fault_active = false;
                self.imu_recovery_since_ms = None;
            }
        } else if self.imu_fault_active {
            self.imu_recovery_since_ms = None;
        }
        let inconsistent =
            snapshot.imu_motion_consistency == status::MotionConsistencyCode::Inconsistent as u8;
        if inconsistent
            && !self.last_motion_inconsistent
            && time_reached(now_ms, self.motion_inconsistency_cooldown_until_ms)
        {
            self.request_audio(AuditoryCue::MotionInconsistency);
            self.motion_inconsistency_cooldown_until_ms =
                now_ms.wrapping_add(MOTION_INCONSISTENCY_COOLDOWN_MS);
        }
        self.last_motion_inconsistent = inconsistent;

        if self.docking_active && snapshot.create_sensor_ir_byte != 0 && self.last_dock_ir == 0 {
            self.request_audio(AuditoryCue::DockSeen);
        }
        self.last_dock_ir = snapshot.create_sensor_ir_byte;

        let full_ready = self.create_responsive && snapshot.oi_mode == 3;
        if full_ready && !self.create_full_ready {
            let cue = if self.restart_create_pending {
                self.restart_create_pending = false;
                AuditoryCue::ServiceComplete
            } else if self.ever_create_full_ready {
                AuditoryCue::Recovery
            } else {
                AuditoryCue::Armed
            };
            self.request_audio(cue);
            self.ever_create_full_ready = true;
        }
        self.create_full_ready = full_ready;
    }

    fn enforce_safety_policy(&mut self) -> Result<(), BrainstemError> {
        let now_ms = self.now_ms();
        let snapshot = status::snapshot(now_ms);
        let flags = snapshot.create_sensor_flags;
        let bump = flags & ((1 << 0) | (1 << 1)) != 0;
        let wheel_drop = flags & (1 << 2) != 0;
        let cliff = flags & ((1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)) != 0;
        let imu_ok = body::IMU_ENABLED && snapshot.imu_health == status::ImuHealthCode::Ok as u8;
        let tilt_observed =
            imu_ok && snapshot.imu_tilt_magnitude_mrad as i16 >= body::IMU_TILT_STOP_MRAD;
        let tilt = if tilt_observed {
            match self.tilt_observed_since_ms {
                Some(started_at_ms) => {
                    time_reached(now_ms, started_at_ms.wrapping_add(IMU_TILT_LATCH_HOLD_MS))
                }
                None => {
                    self.tilt_observed_since_ms = Some(now_ms);
                    false
                }
            }
        } else {
            self.tilt_observed_since_ms = None;
            false
        };
        let impact = imu_ok && snapshot.imu_impact_score_mm_s2 >= body::IMU_IMPACT_STOP_MM_S2;
        let charging = status::charging_interlock_active(&snapshot);
        let home_base = snapshot.create_sensor_charging_sources & 0b10 != 0;

        // Packet 34 is the source of truth for Home Base contact. If Pete was
        // physically removed before an unstarted departure consumed the first
        // motion request, do not carry that reverse program into a later,
        // already off-dock command.
        if !home_base && self.dock_departure_pending {
            self.dock_departure_pending = false;
        }

        // The first complete observation establishes the edge baseline. A
        // bumper held through boot is evidence, not permission to move.
        if !self.safety_observation_initialized {
            self.last_bump = bump;
            self.last_cliff = cliff;
            self.last_wheel_drop = wheel_drop;
            self.safety_observation_initialized = true;
        }
        let fresh_bump_edge = bump && !self.last_bump;
        if let Some(active_escape) = self.active_escape {
            let dominating_hazard = if wheel_drop {
                Some(status::SafetyEventKind::WheelDrop)
            } else if tilt {
                Some(status::SafetyEventKind::Tilt)
            } else if impact {
                Some(status::SafetyEventKind::Impact)
            } else if charging || home_base {
                Some(status::SafetyEventKind::Charging)
            } else {
                match active_escape.kind {
                    SafetyLatchKind::Bump if cliff => Some(status::SafetyEventKind::Cliff),
                    SafetyLatchKind::Cliff if bump => Some(status::SafetyEventKind::Bump),
                    _ => None,
                }
            };
            if let Some(kind) = dominating_hazard {
                status::mark_safety_tripped(kind);
                self.apply_safety_response(kind, SafetyResponse::Stop)?;
                self.request_audio(safety_auditory_cue(kind));
                self.update_safety_edges(bump, cliff, wheel_drop);
                return Ok(());
            }
            if status::validate_escape_motion(
                active_escape.kind,
                active_escape.hazard_generation,
                active_escape.linear_mm_s,
                active_escape.angular_mrad_s,
                active_escape.ttl_ms,
            )
            .is_err()
            {
                self.interrupt_active_command();
                self.commands.clear();
                self.active = ActiveAction::None;
                self.stop_drive()?;
                self.update_safety_edges(bump, cliff, wheel_drop);
                return Ok(());
            }
            // The acknowledged latch remains active and visible. Only this
            // exact generation-bound segment may move through it.
            self.update_safety_edges(bump, cliff, wheel_drop);
            return Ok(());
        }

        if self.careful_mode_active(now_ms) {
            // CAREFUL transfers responsibility for these observations to the
            // active possessor. Keep publishing the raw sensors, but do not
            // turn them back into motor gates until the explicit lease ends.
            self.update_safety_edges(bump, cliff, wheel_drop);
            self.clear_sensor_gates_for_careful();
            return Ok(());
        }

        if home_base && !wheel_drop && !tilt && !impact {
            self.clear_dock_contact_latch();
            if !matches!(self.active, ActiveAction::DockDeparture { .. })
                && !self.dock_departure_pending
            {
                self.clear_motion_queue()?;
                self.stop_drive()?;
                self.finish_contact_withdrawal(
                    status::ContactWithdrawalOutcome::SafetyPreempted,
                    Some(status::SafetyEventKind::Charging),
                    true,
                );
                self.dock_departure_pending = true;
            }
            // Packet 34 lets a Home Base contact replace the conservative
            // unknown-source charging interlock with internal dock handling.
            self.charging_interlock_latched = false;
            return Ok(());
        }

        if charging && !wheel_drop && !tilt && !impact {
            if !self.charging_interlock_latched {
                self.clear_motion_queue()?;
                self.stop_drive()?;
                self.finish_contact_withdrawal(
                    status::ContactWithdrawalOutcome::SafetyPreempted,
                    Some(status::SafetyEventKind::Charging),
                    true,
                );
                self.charging_interlock_latched = true;
            }
            return Ok(());
        }

        if !bump && !cliff && !wheel_drop && !tilt && !impact && !charging {
            self.update_safety_edges(bump, cliff, wheel_drop);
            return Ok(());
        }
        self.update_safety_edges(bump, cliff, wheel_drop);

        if wheel_drop {
            if self.safety_latch_kind != Some(status::SafetyEventKind::WheelDrop) {
                status::mark_safety_tripped(status::SafetyEventKind::WheelDrop);
                status::mark_wheel_drop_latched();
                self.latch_safety(status::SafetyEventKind::WheelDrop);
                self.interrupt_active_command();
                self.commands.clear();
                self.stop_drive()?;
                self.active = ActiveAction::None;
                self.finish_contact_withdrawal(
                    status::ContactWithdrawalOutcome::SafetyPreempted,
                    Some(status::SafetyEventKind::WheelDrop),
                    true,
                );
                self.request_audio(AuditoryCue::WheelDrop);
            }
            return Ok(());
        }
        // A bump latch permits only its own bounded reverse. A stronger local
        // safety observation must still preempt that reflex deterministically.
        if self.safety_latched
            && !(self.active_contact_withdrawal.is_some() && (tilt || impact || cliff))
        {
            return Ok(());
        }

        let (kind, response) = if tilt {
            status::mark_safety_tripped(status::SafetyEventKind::Tilt);
            (status::SafetyEventKind::Tilt, SafetyResponse::Stop)
        } else if impact {
            status::mark_safety_tripped(status::SafetyEventKind::Impact);
            (status::SafetyEventKind::Impact, SafetyResponse::Stop)
        } else if cliff {
            status::mark_safety_tripped(status::SafetyEventKind::Cliff);
            (status::SafetyEventKind::Cliff, SafetyResponse::Stop)
        } else if bump && fresh_bump_edge && self.unsafe_forward_output() {
            status::mark_safety_tripped(status::SafetyEventKind::Bump);
            (
                status::SafetyEventKind::Bump,
                SafetyResponse::ContactWithdrawal,
            )
        } else if bump {
            // A stationary press or boot-restored contact remains observable
            // and latched, but cannot initiate authority-independent motion.
            status::mark_safety_tripped(status::SafetyEventKind::Bump);
            (status::SafetyEventKind::Bump, SafetyResponse::Stop)
        } else if wheel_drop {
            status::mark_safety_tripped(status::SafetyEventKind::WheelDrop);
            (status::SafetyEventKind::WheelDrop, SafetyResponse::Stop)
        } else {
            (status::SafetyEventKind::Bump, SafetyResponse::Stop)
        };
        self.apply_safety_response(kind, response)?;
        self.request_audio(safety_auditory_cue(kind));
        Ok(())
    }

    fn apply_safety_response(
        &mut self,
        kind: status::SafetyEventKind,
        response: SafetyResponse,
    ) -> Result<(), BrainstemError> {
        match response {
            SafetyResponse::Stop => {
                self.latch_safety(kind);
                self.interrupt_active_command();
                self.commands.clear();
                self.stop_drive()?;
                self.active = ActiveAction::None;
                self.finish_contact_withdrawal(
                    status::ContactWithdrawalOutcome::SafetyPreempted,
                    Some(kind),
                    true,
                );
                Ok(())
            }
            SafetyResponse::ContactWithdrawal => {
                self.latch_safety(kind);
                let command_id = self.active_command_id.unwrap_or(0);
                self.interrupt_active_command();
                self.commands.clear();
                self.active = ActiveAction::None;
                self.stop_drive()?;
                if kind == status::SafetyEventKind::Bump {
                    let snapshot = status::snapshot(self.now_ms());
                    self.start_contact_withdrawal(
                        (snapshot.create_sensor_flags & 0b11) as u8,
                        command_id,
                        snapshot.odometry_distance_mm,
                    );
                    self.mode = RuntimeMode::Running;
                    status::set_runtime_state(RuntimeState::Running);
                }
                let _ = self.commands.push_front(QueuedCommand::safety_recovery(
                    0,
                    RuntimeCommand::CmdVel {
                        linear_mm_s: -CONTACT_WITHDRAWAL_SPEED_MM_S,
                        angular_mrad_s: 0,
                        duration_ms: Some(CONTACT_WITHDRAWAL_DURATION_MS),
                    },
                ));
                Ok(())
            }
        }
    }

    fn latch_safety(&mut self, kind: status::SafetyEventKind) {
        self.safety_latched = true;
        self.safety_latch_kind = Some(kind);
    }

    fn clear_dock_contact_latch(&mut self) {
        let Some(kind @ (status::SafetyEventKind::Bump | status::SafetyEventKind::Cliff)) =
            self.safety_latch_kind
        else {
            return;
        };
        // Packet 0 can arrive before the private packet-34 poll identifies
        // Home Base, briefly interpreting dock geometry as a bump/cliff
        // incident. Reconcile only those two contact latches once packet 34
        // proves the source; every stronger latch remains untouched.
        status::mark_safety_cleared(kind);
        self.request_audio(AuditoryCue::SafetyClear);
        self.safety_latched = false;
        self.safety_latch_kind = None;
    }

    fn clear_safety_latch(&mut self, expected: Option<status::SafetyEventKind>) {
        if expected == Some(status::SafetyEventKind::Charging) && self.charging_interlock_latched {
            self.charging_interlock_latched = false;
            return;
        }

        let Some(kind) = self.safety_latch_kind else {
            self.safety_latched = false;
            return;
        };
        if expected.is_some_and(|expected| expected != kind) {
            return;
        }
        let snapshot = status::snapshot(self.now_ms());
        let flags = snapshot.create_sensor_flags;
        let physical_condition_active = match kind {
            status::SafetyEventKind::Bump => flags & 0b11 != 0,
            status::SafetyEventKind::WheelDrop => flags & (1 << 2) != 0,
            status::SafetyEventKind::Cliff => flags & 0b1111_0000 != 0,
            status::SafetyEventKind::Tilt => {
                snapshot.imu_health == status::ImuHealthCode::Ok as u8
                    && snapshot.imu_tilt_magnitude_mrad as i16 >= body::IMU_TILT_STOP_MRAD
            }
            status::SafetyEventKind::Impact => {
                snapshot.imu_health == status::ImuHealthCode::Ok as u8
                    && snapshot.imu_impact_score_mm_s2 >= body::IMU_IMPACT_STOP_MM_S2
            }
            status::SafetyEventKind::Charging => status::charging_interlock_active(&snapshot),
            _ => false,
        };
        if physical_condition_active {
            return;
        }
        status::mark_safety_cleared(kind);
        self.request_audio(AuditoryCue::SafetyClear);
        if kind == status::SafetyEventKind::WheelDrop {
            status::mark_wheel_drop_cleared();
        }
        self.safety_latched = false;
        self.safety_latch_kind = None;
    }

    fn enter_careful_mode(&mut self, requested_ttl_ms: u32) -> Result<(), BrainstemError> {
        self.ensure_create_responsive()?;
        if self.estop_latched {
            return Err(BrainstemError::CreateNoResponse);
        }
        let now_ms = self.now_ms();
        let ttl_ms = requested_ttl_ms.clamp(CAREFUL_MODE_MIN_TTL_MS, CAREFUL_MODE_MAX_TTL_MS);
        self.clear_sensor_gates_for_careful();
        self.careful_mode_until_ms = Some(now_ms.wrapping_add(ttl_ms));
        status::set_careful_mode_until(self.careful_mode_until_ms);
        Ok(())
    }

    fn careful_mode_active(&self, now_ms: u32) -> bool {
        self.careful_mode_until_ms
            .is_some_and(|deadline_ms| !time_reached(now_ms, deadline_ms))
    }

    fn cancel_careful_mode(&mut self) {
        self.careful_mode_until_ms = None;
        status::set_careful_mode_until(None);
    }

    fn clear_sensor_gates_for_careful(&mut self) {
        if let Some(kind) = self.safety_latch_kind.take() {
            status::mark_safety_cleared(kind);
            self.request_audio(AuditoryCue::SafetyClear);
            if kind == status::SafetyEventKind::WheelDrop {
                status::mark_wheel_drop_cleared();
            }
        }
        self.safety_latched = false;
        self.charging_interlock_latched = false;
        self.dock_departure_pending = false;
        self.tilt_observed_since_ms = None;
    }

    fn enforce_careful_mode_timeout(&mut self) -> Result<(), BrainstemError> {
        let Some(deadline_ms) = self.careful_mode_until_ms else {
            return Ok(());
        };
        if !time_reached(self.now_ms(), deadline_ms) {
            return Ok(());
        }

        self.cancel_careful_mode();
        self.interrupt_active_command();
        self.commands.clear();
        self.active = ActiveAction::None;
        self.stop_drive()?;
        self.relatch_current_sensor_gate();
        Ok(())
    }

    fn relatch_current_sensor_gate(&mut self) {
        let snapshot = status::snapshot(self.now_ms());
        let flags = snapshot.create_sensor_flags;
        let imu_ok = body::IMU_ENABLED && snapshot.imu_health == status::ImuHealthCode::Ok as u8;
        let kind = if flags & (1 << 2) != 0 {
            Some(status::SafetyEventKind::WheelDrop)
        } else if flags & 0b1111_0000 != 0 {
            Some(status::SafetyEventKind::Cliff)
        } else if imu_ok && snapshot.imu_tilt_magnitude_mrad as i16 >= body::IMU_TILT_STOP_MRAD {
            Some(status::SafetyEventKind::Tilt)
        } else if imu_ok && snapshot.imu_impact_score_mm_s2 >= body::IMU_IMPACT_STOP_MM_S2 {
            Some(status::SafetyEventKind::Impact)
        } else if flags & 0b11 != 0 {
            Some(status::SafetyEventKind::Bump)
        } else {
            None
        };

        if let Some(kind) = kind {
            status::mark_safety_tripped(kind);
            if kind == status::SafetyEventKind::WheelDrop {
                status::mark_wheel_drop_latched();
            }
            self.latch_safety(kind);
        } else if status::charging_interlock_active(&snapshot) {
            self.charging_interlock_latched = true;
        }
    }

    fn unsafe_forward_output(&self) -> bool {
        self.active_velocity
            .is_some_and(|velocity| velocity.linear_mm_s > 0)
    }

    fn feedback_tones(&self, kind: FeedbackKind) -> ([SongTone; MAX_SONG_TONES], u8) {
        let index = feedback_index(kind);
        if self.chirp_counts[index] > 0 {
            return (self.chirps[index], self.chirp_counts[index]);
        }
        default_feedback_tones(kind)
    }

    fn set_audio_silent(&mut self, silent: bool) {
        let dropped = self.audio.set_silent(silent);
        status::increment_audio_dropped_or_replaced(dropped);
        status::set_audio_silent(silent);
    }

    fn request_audio(&mut self, cue: AuditoryCue) {
        let now_ms = self.now_ms();
        status::mark_audio_cue_requested(cue.code());
        match self.audio.request(cue, now_ms) {
            CueRequestResult::Suppressed => status::increment_audio_suppressed_by_silent(),
            CueRequestResult::Dropped => status::increment_audio_dropped_or_replaced(1),
            CueRequestResult::Ready | CueRequestResult::Queued => {}
        }
    }

    fn poll_audio(&mut self) {
        let now_ms = self.now_ms();
        if !self.create_responsive {
            return;
        }
        let Some(cue) = self.audio.take_ready(now_ms) else {
            return;
        };
        let (tones, tone_count) = cue_tones(cue);
        let played = self
            .create_uart
            .define_song(
                &mut self.hardware,
                &mut self.events,
                AUTOMATIC_CUE_SLOT,
                &tones,
                tone_count,
            )
            .and_then(|()| {
                self.create_uart
                    .play_song(&mut self.hardware, &mut self.events, AUTOMATIC_CUE_SLOT)
            })
            .is_ok();
        if played {
            self.audio.mark_played(cue, now_ms);
            status::mark_audio_cue_played(cue.code(), now_ms);
        } else {
            status::increment_audio_dropped_or_replaced(1);
        }
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
        if left_mm_s == 0 && right_mm_s == 0 {
            self.stop_drive()?;
            self.active = ActiveAction::None;
            return Ok(());
        }
        self.ensure_motion_allowed()?;
        self.active_velocity = None;

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
        if velocity_mm_s == 0 {
            self.stop_drive()?;
            self.active = ActiveAction::None;
            return Ok(());
        }
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
        if self.last_create_packet_at_ms.is_some_and(|last_packet_ms| {
            self.now_ms().wrapping_sub(last_packet_ms) > CREATE_LINK_FRESHNESS_TIMEOUT_MS
        }) {
            self.create_responsive = false;
            status::set_oi_mode_unknown();
        }
        if !self.create_responsive {
            return Err(BrainstemError::CreateNoResponse);
        }
        Ok(())
    }

    fn ensure_motion_allowed(&mut self) -> Result<(), BrainstemError> {
        if self.estop_latched
            || self.dock_departure_pending
            || self.charging_interlock_latched
            || (self.safety_latched && !self.safety_recovery_latch_allows_motion())
        {
            self.stop_drive()?;
            return Err(BrainstemError::CreateNoResponse);
        }
        self.ensure_create_responsive()?;
        Ok(())
    }

    fn safety_recovery_latch_allows_motion(&self) -> bool {
        self.safety_recovery_motion && recoverable_safety_latch(self.safety_latch_kind)
    }

    fn enforce_heartbeat_stop(&mut self) -> Result<(), BrainstemError> {
        let Some(deadline_ms) = self.heartbeat_stop_at_ms else {
            return Ok(());
        };
        if time_reached(self.now_ms(), deadline_ms) {
            self.heartbeat_stop_at_ms = None;
            self.cancel_careful_mode();
            status::revoke_authority();
            status::mark_heartbeat_expired();
            status::mark_safety_tripped(status::SafetyEventKind::Heartbeat);
            self.request_audio(AuditoryCue::HeartbeatLost);
            if self.active_contact_withdrawal.is_none() {
                self.interrupt_active_command();
                self.commands.clear();
                self.active = ActiveAction::None;
                self.stop_drive()?;
            }
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
        let stopped = self.stop_drive().is_ok();
        self.finish_contact_withdrawal(status::ContactWithdrawalOutcome::Failed, None, stopped);
        self.mode = RuntimeMode::Error;
        self.active = ActiveAction::None;
        self.error_blink_next_ms = self.now_ms();
        self.error_blink_count = 0;
        self.error_blink_on = false;
        if self.create_responsive {
            self.request_audio(AuditoryCue::RuntimeError);
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
            ActiveAction::WakeSettle { .. } => RuntimeActionCode::WakeSettle,
            ActiveAction::WaitForCreate { .. } => RuntimeActionCode::WaitForCreate,
            ActiveAction::Settle { .. } => RuntimeActionCode::Settle,
            ActiveAction::Driving { .. } | ActiveAction::DockDeparture { .. } => {
                RuntimeActionCode::Driving
            }
        }
    }

    fn push_event(&mut self, event: BrainstemEvent) {
        status::signal_event(&event);
        let _ = self.events.push_back(event);
    }

    fn complete_active_command(&mut self) {
        self.safety_recovery_motion = false;
        status::clear_velocity_stream();
        if let Some(command_id) = self.active_command_id.take() {
            status::mark_command_completed(command_id);
        }
    }

    fn refuse_active_command(&mut self) {
        self.safety_recovery_motion = false;
        status::clear_velocity_stream();
        if let Some(command_id) = self.active_command_id.take() {
            status::mark_command_interrupted(command_id);
        }
    }

    fn interrupt_active_command(&mut self) {
        if matches!(self.active, ActiveAction::PowerPulse { .. }) {
            self.hardware.end_power_toggle_pulse();
        }
        self.safety_recovery_motion = false;
        self.active_velocity = None;
        self.active_escape = None;
        status::clear_velocity_stream();
        if let Some(command_id) = self.active_command_id.take() {
            status::mark_command_interrupted(command_id);
        }
    }

    fn fail_active_command(&mut self, error: BrainstemError) {
        self.safety_recovery_motion = false;
        status::clear_velocity_stream();
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

    fn start_contact_withdrawal(
        &mut self,
        contact_bits: u8,
        preempted_command_id: u32,
        baseline_odometry_mm: i32,
    ) {
        let now_ms = self.now_ms();
        self.repeated_contact_count = match self.last_contact_withdrawal_at_ms {
            Some(previous) if now_ms.wrapping_sub(previous) <= CONTACT_REPEAT_WINDOW_MS => {
                self.repeated_contact_count.saturating_add(1).max(1)
            }
            _ => 1,
        };
        self.last_contact_withdrawal_at_ms = Some(now_ms);
        self.active_contact_withdrawal = Some(ActiveContactWithdrawal {
            started_at_ms: now_ms,
            baseline_odometry_mm,
        });
        status::mark_contact_withdrawal_started(
            contact_bits,
            self.repeated_contact_count,
            preempted_command_id,
            CONTACT_WITHDRAWAL_SPEED_MM_S.unsigned_abs(),
            CONTACT_WITHDRAWAL_DURATION_MS.min(u32::from(u16::MAX)) as u16,
        );
    }

    fn finish_contact_withdrawal(
        &mut self,
        outcome: status::ContactWithdrawalOutcome,
        dominating_safety: Option<status::SafetyEventKind>,
        final_stopped: bool,
    ) {
        let Some(active) = self.active_contact_withdrawal.take() else {
            return;
        };
        let now_ms = self.now_ms();
        let displacement = status::snapshot(now_ms)
            .odometry_distance_mm
            .wrapping_sub(active.baseline_odometry_mm);
        status::mark_contact_withdrawal_completed(
            outcome,
            dominating_safety,
            final_stopped,
            displacement,
            now_ms.wrapping_sub(active.started_at_ms),
        );
        if matches!(
            outcome,
            status::ContactWithdrawalOutcome::SafetyPreempted
                | status::ContactWithdrawalOutcome::Failed
        ) {
            self.request_audio(AuditoryCue::ServiceFailure);
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
    create_charging_active(snapshot)
        && snapshot.create_sensor_capacity_mah > 0
        && u32::from(snapshot.create_sensor_charge_mah) * 100
            <= u32::from(snapshot.create_sensor_capacity_mah) * LOW_BATTERY_PERCENT
}

fn create_charging_active(snapshot: &status::BrainstemStatus) -> bool {
    snapshot.create_charging_indicator_state == 2
        || matches!(snapshot.create_sensor_charging_state, 1..=3)
}

fn safety_auditory_cue(kind: status::SafetyEventKind) -> AuditoryCue {
    match kind {
        status::SafetyEventKind::EStop => AuditoryCue::EStop,
        status::SafetyEventKind::Bump => AuditoryCue::BumpContact,
        status::SafetyEventKind::Cliff => AuditoryCue::Cliff,
        status::SafetyEventKind::WheelDrop => AuditoryCue::WheelDrop,
        status::SafetyEventKind::Heartbeat => AuditoryCue::HeartbeatLost,
        status::SafetyEventKind::Tilt => AuditoryCue::Tilt,
        status::SafetyEventKind::Impact => AuditoryCue::Impact,
        status::SafetyEventKind::Charging => AuditoryCue::DockContact,
    }
}

fn command_preempts_contact_withdrawal(command: BrainstemCommand) -> bool {
    matches!(
        command,
        BrainstemCommand::Stop | BrainstemCommand::EStop | BrainstemCommand::Disarm
    )
}

fn runtime_command_from_forebrain(command: BrainstemCommand) -> Option<RuntimeCommand> {
    match command {
        BrainstemCommand::Ping
        | BrainstemCommand::Status
        | BrainstemCommand::Bootsel
        | BrainstemCommand::Arm
        | BrainstemCommand::Disarm
        | BrainstemCommand::RestartCreate => None,
        BrainstemCommand::ResetMotherbrain => None,
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
        BrainstemCommand::Unsupported { .. } => None,
        BrainstemCommand::ClearSafetyLatch { kind, .. } => {
            Some(RuntimeCommand::ClearSafetyLatch { kind })
        }
        BrainstemCommand::CarefulMode { ttl_ms, .. } => {
            Some(RuntimeCommand::CarefulMode { ttl_ms })
        }
        BrainstemCommand::EscapeMotion {
            kind,
            hazard_generation,
            linear_mm_s,
            angular_mrad_s,
            ttl_ms,
            ..
        } => Some(RuntimeCommand::EscapeMotion {
            kind,
            hazard_generation,
            linear_mm_s,
            angular_mrad_s,
            ttl_ms,
        }),
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
        BrainstemCommand::SetAudioSilent { silent, .. } => {
            Some(RuntimeCommand::SetAudioSilent { silent })
        }
        BrainstemCommand::PowerState { request, .. } => match request {
            PowerStateRequest::Wake => Some(RuntimeCommand::WakeCreate),
            PowerStateRequest::Sleep => Some(RuntimeCommand::SleepCreate),
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
        BrainstemCommand::OrientationProbe {
            angular_mrad_s,
            duration_ms,
            ..
        } => Some(RuntimeCommand::OrientationProbe {
            angular_mrad_s,
            duration_ms,
        }),
        BrainstemCommand::ResetOdometry { .. } => Some(RuntimeCommand::ResetOdometry),
        BrainstemCommand::ZeroImuOrientation { .. } => Some(RuntimeCommand::ZeroImuOrientation),
        BrainstemCommand::ClearImuOrientation { .. } => Some(RuntimeCommand::ClearImuOrientation),
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
            | RuntimeCommand::EscapeMotion { .. }
            | RuntimeCommand::DriveArc { .. }
            | RuntimeCommand::Drive { .. }
            | RuntimeCommand::StopDrive
            | RuntimeCommand::CalibrateTurn { .. }
            | RuntimeCommand::OrientationProbe { .. }
            | RuntimeCommand::Dock
    )
}

fn requires_dock_departure(command: RuntimeCommand) -> bool {
    match command {
        RuntimeCommand::CmdVel {
            linear_mm_s,
            angular_mrad_s,
            ..
        } => linear_mm_s != 0 || angular_mrad_s != 0,
        RuntimeCommand::EscapeMotion {
            linear_mm_s,
            angular_mrad_s,
            ..
        } => linear_mm_s != 0 || angular_mrad_s != 0,
        RuntimeCommand::Drive {
            left_mm_s,
            right_mm_s,
            ..
        }
        | RuntimeCommand::DriveDirect {
            left_mm_s,
            right_mm_s,
            ..
        } => left_mm_s != 0 || right_mm_s != 0,
        RuntimeCommand::DriveArc { velocity_mm_s, .. } => velocity_mm_s != 0,
        RuntimeCommand::Dock | RuntimeCommand::StopDrive => false,
        _ => is_motion_command(command),
    }
}

fn recoverable_safety_latch(kind: Option<status::SafetyEventKind>) -> bool {
    matches!(
        kind,
        Some(status::SafetyEventKind::Bump | status::SafetyEventKind::Cliff)
    )
}

fn safety_latch_kind_to_event(kind: SafetyLatchKind) -> status::SafetyEventKind {
    match kind {
        SafetyLatchKind::Bump => status::SafetyEventKind::Bump,
        SafetyLatchKind::Cliff => status::SafetyEventKind::Cliff,
        SafetyLatchKind::WheelDrop => status::SafetyEventKind::WheelDrop,
        SafetyLatchKind::Heartbeat => status::SafetyEventKind::Heartbeat,
        SafetyLatchKind::Tilt => status::SafetyEventKind::Tilt,
        SafetyLatchKind::Impact => status::SafetyEventKind::Impact,
        SafetyLatchKind::Charging => status::SafetyEventKind::Charging,
    }
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
        FeedbackKind::Ok => &[(60, 8), (64, 8), (67, 12)],
        FeedbackKind::Error => &[(64, 8), (62, 8), (60, 12)],
        // Solresol "fasolsi": prepare / make ready.
        FeedbackKind::Armed => &[(65, 8), (67, 8), (71, 12)],
        FeedbackKind::LostTarget => &[(55, 8), (52, 8), (48, 12)],
        FeedbackKind::DockSeen => &[(67, 8), (71, 8), (74, 12)],
        FeedbackKind::Danger => &[(48, 6), (48, 6), (48, 12)],
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
#[path = "runtime_tests.rs"]
mod tests;
