#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CockpitRequest {
    Ping,
    Bootsel,
    RestartCreate,
    ResetMotherbrain,
    GetStatus,
    GetCapabilities,
    GetEvents {
        since_seq: u32,
    },
    RegisterNetworkEndpoint(RegisterNetworkEndpoint),
    AcquireControlLease {
        authority: ControlAuthority,
        ttl_ms: u32,
    },
    AcquireServiceLease {
        scope: ServiceScope,
        ttl_ms: u32,
    },
    Arm,
    Disarm,
    Stop,
    EStop,
    ClearEStop,
    ClearSafetyLatch {
        latch: SafetyLatchKind,
    },
    CarefulMode {
        ttl_ms: u32,
    },
    EscapeMotion {
        hazard: SafetyLatchKind,
        hazard_generation: u32,
        linear_mm_s: i16,
        angular_mrad_s: i16,
        ttl_ms: u32,
    },
    CmdVel {
        linear_mm_s: i16,
        angular_mrad_s: i16,
        ttl_ms: u32,
    },
    DriveDirect {
        left_mm_s: i16,
        right_mm_s: i16,
        ttl_ms: u32,
    },
    DriveArc {
        velocity_mm_s: i16,
        radius_mm: i16,
        ttl_ms: u32,
    },
    FaceBearing {
        bearing_mrad: i16,
        max_angular_mrad_s: i16,
        tolerance_mrad: i16,
        ttl_ms: u32,
    },
    TrackBearing {
        bearing_mrad: i16,
        range_mm: u16,
        max_linear_mm_s: i16,
        max_angular_mrad_s: i16,
        stop_range_mm: u16,
        ttl_ms: u32,
    },
    TurnBy {
        angle_mrad: i16,
        angular_mrad_s: i16,
        timeout_ms: u32,
    },
    DriveFor {
        distance_mm: i16,
        velocity_mm_s: i16,
        timeout_ms: u32,
    },
    BumpEscape {
        direction: EscapeDirection,
        backoff_mm_s: i16,
        turn_angular_mrad_s: i16,
    },
    HoldHeading {
        heading_error_mrad: i16,
        velocity_mm_s: i16,
        max_angular_mrad_s: i16,
        ttl_ms: u32,
    },
    TurnToHeading {
        heading_error_mrad: i16,
        angular_mrad_s: i16,
        tolerance_mrad: i16,
        timeout_ms: u32,
    },
    ArcFor {
        velocity_mm_s: i16,
        radius_mm: i16,
        duration_ms: u32,
    },
    CreepUntil {
        velocity_mm_s: i16,
        angular_mrad_s: i16,
        timeout_ms: u32,
    },
    ScanArc {
        angle_mrad: i16,
        angular_mrad_s: i16,
        timeout_ms: u32,
    },
    DockAlign {
        bearing_mrad: i16,
        range_mm: u16,
        max_linear_mm_s: i16,
        max_angular_mrad_s: i16,
        stop_range_mm: u16,
        ttl_ms: u32,
    },
    WallFollow {
        distance_error_mm: i16,
        velocity_mm_s: i16,
        max_angular_mrad_s: i16,
        ttl_ms: u32,
    },
    WiggleAlign {
        amplitude_mrad: i16,
        angular_mrad_s: i16,
        cycles: u8,
    },
    Unstick {
        direction: EscapeDirection,
        backoff_mm_s: i16,
        turn_angular_mrad_s: i16,
    },
    CliffGuard {
        clear: bool,
    },
    HeartbeatStop {
        timeout_ms: u32,
    },
    RequestSensors {
        packet_id: u8,
    },
    StreamSensors {
        enabled: bool,
        packet_id: u8,
        period_ms: u32,
    },
    SetSafetyPolicy {
        policy: SafetyPolicy,
    },
    ClearMotionQueue,
    DefineChirp {
        feedback: FeedbackKind,
        tones: Vec<SongTone>,
    },
    PlayFeedback {
        feedback: FeedbackKind,
    },
    SetAudioSilent {
        silent: bool,
    },
    PowerState {
        request: PowerStateRequest,
    },
    CreatePowerOn,
    CreatePowerOff,
    CalibrateTurn {
        angular_mrad_s: i16,
        duration_ms: u32,
    },
    OrientationProbe {
        angular_mrad_s: i16,
        duration_ms: u32,
    },
    ResetOdometry,
    ZeroImuOrientation,
    ClearImuOrientation,
    SetMode {
        mode: CreateOiMode,
    },
    SongDefine {
        id: u8,
        tones: Vec<SongTone>,
    },
    SongPlay {
        id: u8,
    },
    Dock,
    SetLights {
        pattern: LightPattern,
    },
}

impl CockpitRequest {
    fn required_service_scope(&self) -> Option<ServiceScope> {
        match self {
            Self::Bootsel => Some(ServiceScope::Bootsel),
            Self::RestartCreate => Some(ServiceScope::RestartCreate),
            Self::ResetMotherbrain => Some(ServiceScope::ResetMotherbrain),
            _ => None,
        }
    }

    fn requires_operator_debug(&self) -> bool {
        matches!(self, Self::CarefulMode { .. })
    }

    pub fn authorization_class(&self) -> AuthorizationClass {
        match self {
            Self::Ping | Self::GetStatus | Self::GetCapabilities | Self::GetEvents { .. } => {
                AuthorizationClass::ReadOnly
            }
            Self::Stop | Self::EStop => AuthorizationClass::Emergency,
            Self::RegisterNetworkEndpoint(_)
            | Self::AcquireControlLease { .. }
            | Self::AcquireServiceLease { .. }
            | Self::RequestSensors { .. }
            | Self::StreamSensors { .. }
            | Self::SetAudioSilent { .. }
            | Self::Disarm => AuthorizationClass::Session,
            Self::Bootsel | Self::RestartCreate | Self::ResetMotherbrain => {
                AuthorizationClass::ServiceLease
            }
            _ => AuthorizationClass::ControlLease,
        }
    }

    pub fn requires_session(&self) -> bool {
        !matches!(
            self.authorization_class(),
            AuthorizationClass::ReadOnly | AuthorizationClass::Emergency
        )
    }

    pub fn requires_control_authority(&self) -> bool {
        self.authorization_class() == AuthorizationClass::ControlLease
    }

    fn bypasses_closed_motor_gate(&self) -> bool {
        matches!(
            self,
            Self::Stop
                | Self::EStop
                | Self::ClearEStop
                | Self::ClearSafetyLatch { .. }
                | Self::CarefulMode { .. }
                | Self::EscapeMotion { .. }
                | Self::BumpEscape { .. }
                | Self::Unstick { .. }
                | Self::CliffGuard { .. }
                | Self::OrientationProbe { .. }
                | Self::ZeroImuOrientation
                | Self::ClearImuOrientation
        )
    }

    pub fn verb(&self) -> &'static str {
        match self {
            Self::Ping => "ping",
            Self::Bootsel => "bootsel",
            Self::RestartCreate => "restart_create",
            Self::ResetMotherbrain => "reset_motherbrain",
            Self::GetStatus => "status",
            Self::GetCapabilities => "get_capabilities",
            Self::GetEvents { .. } => "get_events",
            Self::RegisterNetworkEndpoint(_) => "register_network_endpoint",
            Self::AcquireControlLease { .. } => "acquire_control_lease",
            Self::AcquireServiceLease { .. } => "acquire_service_lease",
            Self::Arm => "arm",
            Self::Disarm => "disarm",
            Self::Stop => "stop",
            Self::EStop => "estop",
            Self::ClearEStop => "clear_estop",
            Self::ClearSafetyLatch { .. } => "clear_safety_latch",
            Self::CarefulMode { .. } => "careful_mode",
            Self::EscapeMotion { .. } => "escape_motion",
            Self::CmdVel { .. } => "cmd_vel",
            Self::DriveDirect { .. } => "drive_direct",
            Self::DriveArc { .. } => "drive_arc",
            Self::FaceBearing { .. } => "face_bearing",
            Self::TrackBearing { .. } => "track_bearing",
            Self::TurnBy { .. } => "turn_by",
            Self::DriveFor { .. } => "drive_for",
            Self::BumpEscape { .. } => "bump_escape",
            Self::HoldHeading { .. } => "hold_heading",
            Self::TurnToHeading { .. } => "turn_to_heading",
            Self::ArcFor { .. } => "arc_for",
            Self::CreepUntil { .. } => "creep_until",
            Self::ScanArc { .. } => "scan_arc",
            Self::DockAlign { .. } => "dock_align",
            Self::WallFollow { .. } => "wall_follow",
            Self::WiggleAlign { .. } => "wiggle_align",
            Self::Unstick { .. } => "unstick",
            Self::CliffGuard { .. } => "cliff_guard",
            Self::HeartbeatStop { .. } => "heartbeat_stop",
            Self::RequestSensors { .. } => "request_sensors",
            Self::StreamSensors { .. } => "stream_sensors",
            Self::SetSafetyPolicy { .. } => "set_safety_policy",
            Self::ClearMotionQueue => "clear_motion_queue",
            Self::DefineChirp { .. } => "define_chirp",
            Self::PlayFeedback { .. } => "play_feedback",
            Self::SetAudioSilent { .. } => "set_silent",
            Self::PowerState { .. } => "power_state",
            Self::CreatePowerOn => "create_power_on",
            Self::CreatePowerOff => "create_power_off",
            Self::CalibrateTurn { .. } => "calibrate_turn",
            Self::OrientationProbe { .. } => "orientation_probe",
            Self::ResetOdometry => "reset_odometry",
            Self::ZeroImuOrientation => "zero_imu_orientation",
            Self::ClearImuOrientation => "clear_imu_orientation",
            Self::SetMode { .. } => "set_mode",
            Self::SongDefine { .. } => "song_define",
            Self::SongPlay { .. } => "song_play",
            Self::Dock => "dock",
            Self::SetLights { .. } => "set_lights",
        }
    }

    pub fn required_capability(&self) -> Option<&'static str> {
        match self {
            Self::Bootsel
            | Self::RegisterNetworkEndpoint(_)
            | Self::AcquireControlLease { .. }
            | Self::AcquireServiceLease { .. } => None,
            other => Some(other.verb()),
        }
    }

    pub fn capability_verbs() -> Vec<&'static str> {
        sample_cockpit_capability_verbs()
    }

    fn ttl_or_timeout_ms(&self) -> Option<u32> {
        match self {
            Self::CmdVel { ttl_ms, .. }
            | Self::DriveDirect { ttl_ms, .. }
            | Self::DriveArc { ttl_ms, .. }
            | Self::FaceBearing { ttl_ms, .. }
            | Self::TrackBearing { ttl_ms, .. }
            | Self::HoldHeading { ttl_ms, .. }
            | Self::DockAlign { ttl_ms, .. }
            | Self::WallFollow { ttl_ms, .. } => Some(*ttl_ms),
            Self::TurnBy { timeout_ms, .. }
            | Self::DriveFor { timeout_ms, .. }
            | Self::CreepUntil { timeout_ms, .. }
            | Self::ScanArc { timeout_ms, .. }
            | Self::TurnToHeading { timeout_ms, .. } => Some(*timeout_ms),
            Self::ArcFor { duration_ms, .. }
            | Self::CalibrateTurn { duration_ms, .. }
            | Self::OrientationProbe { duration_ms, .. } => Some(*duration_ms),
            Self::HeartbeatStop { timeout_ms } | Self::CarefulMode { ttl_ms: timeout_ms } => {
                Some(*timeout_ms)
            }
            Self::EscapeMotion { ttl_ms, .. } => Some(*ttl_ms),
            Self::StreamSensors { period_ms, .. } => Some(*period_ms),
            _ => None,
        }
    }

    pub fn apply<C: Cockpit>(&self, client: &mut C) -> Result<CockpitResponse> {
        match self {
            Self::Ping => client.ping().map(|()| CockpitResponse::Accepted),
            Self::Bootsel => client.bootsel().map(|()| CockpitResponse::Accepted),
            Self::RestartCreate | Self::ResetMotherbrain => client.execute(self.clone()),
            Self::GetStatus => Ok(CockpitResponse::Status(client.get_status()?)),
            Self::GetCapabilities => Ok(CockpitResponse::Capabilities(client.get_capabilities()?)),
            Self::GetEvents { since_seq } => Ok(CockpitResponse::Events(
                client.get_events_since(*since_seq)?,
            )),
            Self::RegisterNetworkEndpoint(_) => Err(CockpitError::SessionRequired),
            Self::AcquireControlLease { .. } => Err(CockpitError::SessionRequired),
            Self::AcquireServiceLease { .. } => Err(CockpitError::SessionRequired),
            Self::Arm => client.arm().map(|()| CockpitResponse::Accepted),
            Self::Disarm => client.disarm().map(|()| CockpitResponse::Accepted),
            Self::Stop => client.stop().map(|()| CockpitResponse::Accepted),
            Self::EStop => client.estop().map(|()| CockpitResponse::Accepted),
            Self::ClearEStop => client.clear_estop().map(|()| CockpitResponse::Accepted),
            Self::ClearSafetyLatch { latch } => client
                .clear_safety_latch(*latch)
                .map(|()| CockpitResponse::Accepted),
            Self::CarefulMode { ttl_ms } => client
                .careful_mode(*ttl_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::EscapeMotion {
                hazard,
                hazard_generation,
                linear_mm_s,
                angular_mrad_s,
                ttl_ms,
            } => client
                .escape_motion(
                    *hazard,
                    *hazard_generation,
                    *linear_mm_s,
                    *angular_mrad_s,
                    *ttl_ms,
                )
                .map(|()| CockpitResponse::Accepted),
            Self::CmdVel {
                linear_mm_s,
                angular_mrad_s,
                ttl_ms,
            } => client
                .cmd_vel(*linear_mm_s, *angular_mrad_s, *ttl_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::DriveDirect {
                left_mm_s,
                right_mm_s,
                ttl_ms,
            } => client
                .drive_direct(*left_mm_s, *right_mm_s, *ttl_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::DriveArc {
                velocity_mm_s,
                radius_mm,
                ttl_ms,
            } => client
                .drive_arc(*velocity_mm_s, *radius_mm, *ttl_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::FaceBearing {
                bearing_mrad,
                max_angular_mrad_s,
                tolerance_mrad,
                ttl_ms,
            } => client
                .face_bearing(*bearing_mrad, *max_angular_mrad_s, *tolerance_mrad, *ttl_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::TrackBearing {
                bearing_mrad,
                range_mm,
                max_linear_mm_s,
                max_angular_mrad_s,
                stop_range_mm,
                ttl_ms,
            } => client
                .track_bearing(
                    *bearing_mrad,
                    *range_mm,
                    *max_linear_mm_s,
                    *max_angular_mrad_s,
                    *stop_range_mm,
                    *ttl_ms,
                )
                .map(|()| CockpitResponse::Accepted),
            Self::TurnBy {
                angle_mrad,
                angular_mrad_s,
                timeout_ms,
            } => client
                .turn_by(*angle_mrad, *angular_mrad_s, *timeout_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::DriveFor {
                distance_mm,
                velocity_mm_s,
                timeout_ms,
            } => client
                .drive_for(*distance_mm, *velocity_mm_s, *timeout_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::BumpEscape {
                direction,
                backoff_mm_s,
                turn_angular_mrad_s,
            } => client
                .bump_escape(*direction, *backoff_mm_s, *turn_angular_mrad_s)
                .map(|()| CockpitResponse::Accepted),
            Self::HoldHeading {
                heading_error_mrad,
                velocity_mm_s,
                max_angular_mrad_s,
                ttl_ms,
            } => client
                .hold_heading(
                    *heading_error_mrad,
                    *velocity_mm_s,
                    *max_angular_mrad_s,
                    *ttl_ms,
                )
                .map(|()| CockpitResponse::Accepted),
            Self::TurnToHeading {
                heading_error_mrad,
                angular_mrad_s,
                tolerance_mrad,
                timeout_ms,
            } => client
                .turn_to_heading(
                    *heading_error_mrad,
                    *angular_mrad_s,
                    *tolerance_mrad,
                    *timeout_ms,
                )
                .map(|()| CockpitResponse::Accepted),
            Self::ArcFor {
                velocity_mm_s,
                radius_mm,
                duration_ms,
            } => client
                .arc_for(*velocity_mm_s, *radius_mm, *duration_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::CreepUntil {
                velocity_mm_s,
                angular_mrad_s,
                timeout_ms,
            } => client
                .creep_until(*velocity_mm_s, *angular_mrad_s, *timeout_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::ScanArc {
                angle_mrad,
                angular_mrad_s,
                timeout_ms,
            } => client
                .scan_arc(*angle_mrad, *angular_mrad_s, *timeout_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::DockAlign {
                bearing_mrad,
                range_mm,
                max_linear_mm_s,
                max_angular_mrad_s,
                stop_range_mm,
                ttl_ms,
            } => client
                .dock_align(
                    *bearing_mrad,
                    *range_mm,
                    *max_linear_mm_s,
                    *max_angular_mrad_s,
                    *stop_range_mm,
                    *ttl_ms,
                )
                .map(|()| CockpitResponse::Accepted),
            Self::WallFollow {
                distance_error_mm,
                velocity_mm_s,
                max_angular_mrad_s,
                ttl_ms,
            } => client
                .wall_follow(
                    *distance_error_mm,
                    *velocity_mm_s,
                    *max_angular_mrad_s,
                    *ttl_ms,
                )
                .map(|()| CockpitResponse::Accepted),
            Self::WiggleAlign {
                amplitude_mrad,
                angular_mrad_s,
                cycles,
            } => client
                .wiggle_align(*amplitude_mrad, *angular_mrad_s, *cycles)
                .map(|()| CockpitResponse::Accepted),
            Self::Unstick {
                direction,
                backoff_mm_s,
                turn_angular_mrad_s,
            } => client
                .unstick(*direction, *backoff_mm_s, *turn_angular_mrad_s)
                .map(|()| CockpitResponse::Accepted),
            Self::CliffGuard { clear } => client
                .cliff_guard(*clear)
                .map(|()| CockpitResponse::Accepted),
            Self::HeartbeatStop { timeout_ms } => client
                .heartbeat_stop(*timeout_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::RequestSensors { packet_id } => client
                .request_sensors(*packet_id)
                .map(|()| CockpitResponse::Accepted),
            Self::StreamSensors {
                enabled,
                packet_id,
                period_ms,
            } => client
                .stream_sensors(*enabled, *packet_id, *period_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::ResetOdometry => client.reset_odometry().map(|()| CockpitResponse::Accepted),
            Self::ZeroImuOrientation => client
                .zero_imu_orientation()
                .map(|()| CockpitResponse::Accepted),
            Self::ClearImuOrientation => client
                .clear_imu_orientation()
                .map(|()| CockpitResponse::Accepted),
            Self::SetSafetyPolicy { policy } => client
                .set_safety_policy(*policy)
                .map(|()| CockpitResponse::Accepted),
            Self::ClearMotionQueue => client
                .clear_motion_queue()
                .map(|()| CockpitResponse::Accepted),
            Self::DefineChirp { feedback, tones } => client
                .define_chirp(*feedback, tones)
                .map(|()| CockpitResponse::Accepted),
            Self::PlayFeedback { feedback } => client
                .play_feedback(*feedback)
                .map(|()| CockpitResponse::Accepted),
            Self::SetAudioSilent { silent } => client
                .set_audio_silent(*silent)
                .map(|()| CockpitResponse::Accepted),
            Self::PowerState { request } => client
                .power_state(*request)
                .map(|()| CockpitResponse::Accepted),
            Self::CreatePowerOn => client
                .power_state(PowerStateRequest::Wake)
                .map(|()| CockpitResponse::Accepted),
            Self::CreatePowerOff => client
                .power_state(PowerStateRequest::Sleep)
                .map(|()| CockpitResponse::Accepted),
            Self::CalibrateTurn {
                angular_mrad_s,
                duration_ms,
            } => client
                .calibrate_turn(*angular_mrad_s, *duration_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::OrientationProbe {
                angular_mrad_s,
                duration_ms,
            } => client
                .orientation_probe(*angular_mrad_s, *duration_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::SetMode { mode } => client.set_mode(*mode).map(|()| CockpitResponse::Accepted),
            Self::SongDefine { id, tones } => client
                .song_define(*id, tones)
                .map(|()| CockpitResponse::Accepted),
            Self::SongPlay { id } => client.song_play(*id).map(|()| CockpitResponse::Accepted),
            Self::Dock => client.dock().map(|()| CockpitResponse::Accepted),
            Self::SetLights { pattern } => client
                .set_lights(*pattern)
                .map(|()| CockpitResponse::Accepted),
        }
    }

    pub fn to_firmware_json(&self, command_id: u32) -> Result<String> {
        let mut value = serde_json::to_value(self)?;
        if let Some(object) = value.as_object_mut() {
            object.insert("command_id".to_owned(), command_id.into());
            if self.needs_seq() {
                object.insert("seq".to_owned(), command_id.into());
            }
            rewrite_for_firmware_json(self, object);
            if let Some(kind) = object.get_mut("kind") {
                if kind == "get_status" {
                    *kind = "status".into();
                } else if kind == "e_stop" {
                    *kind = "estop".into();
                } else if kind == "clear_e_stop" {
                    *kind = "clear_estop".into();
                } else if kind == "set_audio_silent" {
                    *kind = "set_silent".into();
                }
            }
        }
        Ok(serde_json::to_string(&value)?)
    }

    pub fn to_firmware_json_with_session(
        &self,
        command_id: u32,
        session_id: &str,
    ) -> Result<String> {
        let mut value: serde_json::Value =
            serde_json::from_str(&self.to_firmware_json(command_id)?)?;
        value
            .as_object_mut()
            .expect("request serializes as object")
            .insert(
                "session_id".to_owned(),
                serde_json::Value::String(session_id.to_owned()),
            );
        Ok(serde_json::to_string(&value)?)
    }

    pub fn to_firmware_json_with_authority(
        &self,
        command_id: u32,
        session_id: &str,
        lease_id: &str,
    ) -> Result<String> {
        let mut value: serde_json::Value =
            serde_json::from_str(&self.to_firmware_json_with_session(command_id, session_id)?)?;
        value
            .as_object_mut()
            .expect("request serializes as object")
            .insert("lease_id".into(), lease_id.into());
        Ok(serde_json::to_string(&value)?)
    }

    pub fn to_firmware_json_with_service_authority(
        &self,
        command_id: u32,
        session_id: &str,
        lease_id: &str,
    ) -> Result<String> {
        let mut value: serde_json::Value =
            serde_json::from_str(&self.to_firmware_json_with_session(command_id, session_id)?)?;
        value
            .as_object_mut()
            .expect("request serializes as object")
            .insert("service_lease_id".into(), lease_id.into());
        Ok(serde_json::to_string(&value)?)
    }

    fn needs_seq(&self) -> bool {
        !matches!(
            self,
            Self::Ping
                | Self::Bootsel
                | Self::RestartCreate
                | Self::ResetMotherbrain
                | Self::GetStatus
                | Self::GetCapabilities
                | Self::GetEvents { .. }
                | Self::Arm
                | Self::Disarm
                | Self::Stop
                | Self::EStop
                | Self::ClearEStop
                | Self::ClearSafetyLatch { .. }
                | Self::SetMode { .. }
                | Self::SongPlay { .. }
                | Self::Dock
                | Self::SetLights { .. }
        )
    }

    fn to_compact_line(&self, seq: u32) -> String {
        match self {
            Self::Ping => format!("PING {seq}\n"),
            Self::Bootsel => format!("BOOTSEL {seq}\n"),
            Self::RestartCreate => format!("RESTART_CREATE {seq}\n"),
            Self::ResetMotherbrain => format!("RESET_MOTHERBRAIN {seq}\n"),
            Self::GetStatus => format!("STATUS {seq}\n"),
            Self::GetCapabilities => format!("GET_CAPABILITIES {seq}\n"),
            Self::GetEvents { since_seq } => format!("GET_EVENTS {seq} {since_seq}\n"),
            Self::RegisterNetworkEndpoint(registration) => format!(
                "REGISTER_NETWORK_ENDPOINT {seq} {} {} {} {} {}\n",
                registration.interface_id,
                registration.address,
                registration.hostname,
                registration.lease_identity,
                registration.ttl_seconds,
            ),
            Self::AcquireControlLease { authority, ttl_ms } => format!(
                "ACQUIRE_CONTROL_LEASE {seq} {} {ttl_ms}\n",
                match authority {
                    ControlAuthority::Motherbrain => "motherbrain",
                    ControlAuthority::ForebrainRecovery => "forebrain_recovery",
                    ControlAuthority::OperatorDebug => "operator_debug",
                }
            ),
            Self::AcquireServiceLease { scope, ttl_ms } => format!(
                "ACQUIRE_SERVICE_LEASE {seq} {} {ttl_ms}\n",
                match scope {
                    ServiceScope::Bootsel => "bootsel",
                    ServiceScope::RestartCreate => "restart_create",
                    ServiceScope::ResetMotherbrain => "reset_motherbrain",
                }
            ),
            Self::Arm => format!("ARM {seq}\n"),
            Self::Disarm => format!("DISARM {seq}\n"),
            Self::Stop => format!("STOP {seq}\n"),
            Self::EStop => format!("ESTOP {seq}\n"),
            Self::ClearEStop => format!("CLEAR_ESTOP {seq}\n"),
            Self::ClearSafetyLatch { latch } => {
                format!("CLEAR_SAFETY_LATCH {seq} {}\n", latch.as_str())
            }
            Self::CarefulMode { ttl_ms } => format!("CAREFUL_MODE {seq} {ttl_ms}\n"),
            Self::EscapeMotion {
                hazard,
                hazard_generation,
                linear_mm_s,
                angular_mrad_s,
                ttl_ms,
            } => format!(
                "ESCAPE_MOTION {seq} {} {hazard_generation} {linear_mm_s} {angular_mrad_s} {ttl_ms}\n",
                hazard.as_str()
            ),
            Self::CmdVel {
                linear_mm_s,
                angular_mrad_s,
                ttl_ms,
            } => format!("CMD_VEL {seq} {linear_mm_s} {angular_mrad_s} {ttl_ms}\n"),
            Self::DriveDirect {
                left_mm_s,
                right_mm_s,
                ttl_ms,
            } => format!("DRIVE_DIRECT {seq} {left_mm_s} {right_mm_s} {ttl_ms}\n"),
            Self::DriveArc {
                velocity_mm_s,
                radius_mm,
                ttl_ms,
            } => format!("DRIVE_ARC {seq} {velocity_mm_s} {radius_mm} {ttl_ms}\n"),
            Self::FaceBearing {
                bearing_mrad,
                max_angular_mrad_s,
                tolerance_mrad,
                ttl_ms,
            } => format!(
                "FACE_BEARING {seq} {bearing_mrad} {max_angular_mrad_s} {tolerance_mrad} {ttl_ms}\n"
            ),
            Self::TrackBearing {
                bearing_mrad,
                range_mm,
                max_linear_mm_s,
                max_angular_mrad_s,
                stop_range_mm,
                ttl_ms,
            } => format!(
                "TRACK_BEARING {seq} {bearing_mrad} {range_mm} {max_linear_mm_s} {max_angular_mrad_s} {stop_range_mm} {ttl_ms}\n"
            ),
            Self::TurnBy {
                angle_mrad,
                angular_mrad_s,
                timeout_ms,
            } => format!("TURN_BY {seq} {angle_mrad} {angular_mrad_s} {timeout_ms}\n"),
            Self::DriveFor {
                distance_mm,
                velocity_mm_s,
                timeout_ms,
            } => format!("DRIVE_FOR {seq} {distance_mm} {velocity_mm_s} {timeout_ms}\n"),
            Self::BumpEscape {
                direction,
                backoff_mm_s,
                turn_angular_mrad_s,
            } => format!(
                "BUMP_ESCAPE {seq} {} {backoff_mm_s} {turn_angular_mrad_s}\n",
                direction.as_str()
            ),
            Self::HoldHeading {
                heading_error_mrad,
                velocity_mm_s,
                max_angular_mrad_s,
                ttl_ms,
            } => format!(
                "HOLD_HEADING {seq} {heading_error_mrad} {velocity_mm_s} {max_angular_mrad_s} {ttl_ms}\n"
            ),
            Self::TurnToHeading {
                heading_error_mrad,
                angular_mrad_s,
                tolerance_mrad,
                timeout_ms,
            } => format!(
                "TURN_TO_HEADING {seq} {heading_error_mrad} {angular_mrad_s} {tolerance_mrad} {timeout_ms}\n"
            ),
            Self::ArcFor {
                velocity_mm_s,
                radius_mm,
                duration_ms,
            } => format!("ARC_FOR {seq} {velocity_mm_s} {radius_mm} {duration_ms}\n"),
            Self::CreepUntil {
                velocity_mm_s,
                angular_mrad_s,
                timeout_ms,
            } => format!("CREEP_UNTIL {seq} {velocity_mm_s} {angular_mrad_s} {timeout_ms}\n"),
            Self::ScanArc {
                angle_mrad,
                angular_mrad_s,
                timeout_ms,
            } => format!("SCAN_ARC {seq} {angle_mrad} {angular_mrad_s} {timeout_ms}\n"),
            Self::DockAlign {
                bearing_mrad,
                range_mm,
                max_linear_mm_s,
                max_angular_mrad_s,
                stop_range_mm,
                ttl_ms,
            } => format!(
                "DOCK_ALIGN {seq} {bearing_mrad} {range_mm} {max_linear_mm_s} {max_angular_mrad_s} {stop_range_mm} {ttl_ms}\n"
            ),
            Self::WallFollow {
                distance_error_mm,
                velocity_mm_s,
                max_angular_mrad_s,
                ttl_ms,
            } => format!(
                "WALL_FOLLOW {seq} {distance_error_mm} {velocity_mm_s} {max_angular_mrad_s} {ttl_ms}\n"
            ),
            Self::WiggleAlign {
                amplitude_mrad,
                angular_mrad_s,
                cycles,
            } => format!("WIGGLE_ALIGN {seq} {amplitude_mrad} {angular_mrad_s} {cycles}\n"),
            Self::Unstick {
                direction,
                backoff_mm_s,
                turn_angular_mrad_s,
            } => format!(
                "UNSTICK {seq} {} {backoff_mm_s} {turn_angular_mrad_s}\n",
                direction.as_str()
            ),
            Self::CliffGuard { clear } => format!("CLIFF_GUARD {seq} {clear}\n"),
            Self::HeartbeatStop { timeout_ms } => format!("HEARTBEAT_STOP {seq} {timeout_ms}\n"),
            Self::RequestSensors { packet_id } => format!("REQUEST_SENSORS {seq} {packet_id}\n"),
            Self::StreamSensors {
                enabled,
                packet_id,
                period_ms,
            } => format!("STREAM_SENSORS {seq} {enabled} {packet_id} {period_ms}\n"),
            Self::SetSafetyPolicy { policy } => format!(
                "SET_SAFETY_POLICY {seq} {} {} {}\n",
                policy.bump.as_str(),
                policy.cliff.as_str(),
                policy.wheel_drop_latch
            ),
            Self::ClearMotionQueue => format!("CLEAR_MOTION_QUEUE {seq}\n"),
            Self::DefineChirp { feedback, tones } => {
                format!(
                    "DEFINE_CHIRP {seq} {}{}\n",
                    feedback.as_str(),
                    compact_tones(tones)
                )
            }
            Self::PlayFeedback { feedback } => {
                format!("PLAY_FEEDBACK {seq} {}\n", feedback.as_str())
            }
            Self::SetAudioSilent { silent } => format!("SET_SILENT {seq} {silent}\n"),
            Self::PowerState { request } => format!("POWER_STATE {seq} {}\n", request.as_str()),
            Self::CreatePowerOn => format!("CREATE_POWER_ON {seq}\n"),
            Self::CreatePowerOff => format!("CREATE_POWER_OFF {seq}\n"),
            Self::CalibrateTurn {
                angular_mrad_s,
                duration_ms,
            } => format!("CALIBRATE_TURN {seq} {angular_mrad_s} {duration_ms}\n"),
            Self::OrientationProbe {
                angular_mrad_s,
                duration_ms,
            } => format!("ORIENTATION_PROBE {seq} {angular_mrad_s} {duration_ms}\n"),
            Self::ResetOdometry => format!("RESET_ODOMETRY {seq}\n"),
            Self::ZeroImuOrientation => format!("ZERO_IMU_ORIENTATION {seq}\n"),
            Self::ClearImuOrientation => format!("CLEAR_IMU_ORIENTATION {seq}\n"),
            Self::SetMode { mode } => format!("SET_MODE {seq} {}\n", mode.as_str()),
            Self::SongDefine { id, tones } => {
                format!("SONG_DEFINE {seq} {id}{}\n", compact_tones(tones))
            }
            Self::SongPlay { id } => format!("SONG_PLAY {seq} {id}\n"),
            Self::Dock => format!("DOCK {seq}\n"),
            Self::SetLights { pattern } => format!("SET_LIGHTS {seq} {}\n", pattern.as_str()),
        }
    }

    pub fn to_compact_line_with_session(&self, seq: u32, session_id: &str) -> String {
        let mut line = self.to_compact_line(seq);
        line.pop();
        line.push_str(" session_id=");
        line.push_str(session_id);
        line.push('\n');
        line
    }

    pub fn to_compact_line_with_authority(
        &self,
        seq: u32,
        session_id: &str,
        lease_id: &str,
    ) -> String {
        let mut line = self.to_compact_line_with_session(seq, session_id);
        line.pop();
        line.push_str(" lease_id=");
        line.push_str(lease_id);
        line.push('\n');
        line
    }

    pub fn to_compact_line_with_service_authority(
        &self,
        seq: u32,
        session_id: &str,
        lease_id: &str,
    ) -> String {
        let mut line = self.to_compact_line_with_session(seq, session_id);
        line.pop();
        line.push_str(" service_lease_id=");
        line.push_str(lease_id);
        line.push('\n');
        line
    }
    pub fn to_bridge_json(&self, command_id: u32) -> Result<String> {
        self.to_firmware_json(command_id)
    }
}

fn sample_cockpit_capability_verbs() -> Vec<&'static str> {
    vec![
        "ping",
        "status",
        "get_capabilities",
        "get_events",
        "bootsel",
        "restart_create",
        "reset_motherbrain",
        "arm",
        "disarm",
        "stop",
        "estop",
        "clear_estop",
        "clear_safety_latch",
        "careful_mode",
        "escape_motion",
        "clear_motion_queue",
        "cmd_vel",
        "drive_direct",
        "drive_arc",
        "heartbeat_stop",
        "request_sensors",
        "stream_sensors",
        "song_define",
        "song_play",
        "define_chirp",
        "play_feedback",
        "set_silent",
        "power_state",
        "create_power_on",
        "create_power_off",
        "calibrate_turn",
        "orientation_probe",
        "reset_odometry",
        "zero_imu_orientation",
        "clear_imu_orientation",
        "dock",
        "set_lights",
        "set_mode",
    ]
}

fn optional_cockpit_verbs() -> Vec<&'static str> {
    // CAREFUL is a forward migration: current hosts must still accept and
    // service-flash brainstems that predate the explicit override lease.
    vec![
        "bootsel",
        "reset_motherbrain",
        "careful_mode",
        "escape_motion",
        "set_silent",
    ]
}

// Older brainstems advertised these convenience commands directly. Current
// firmware implements the primitives underneath them instead, but the host
// must continue accepting the old vocabulary so it can establish the service
// session needed to put pre-upgrade firmware into BOOTSEL.
fn legacy_brainstem_convenience_verbs() -> Vec<&'static str> {
    vec![
        "drive_for",
        "turn_by",
        "arc_for",
        "creep_until",
        "scan_arc",
        "face_bearing",
        "track_bearing",
        "hold_heading",
        "turn_to_heading",
        "dock_align",
        "wall_follow",
        "wiggle_align",
        "bump_escape",
        "unstick",
        "cliff_guard",
        "set_safety_policy",
    ]
}

fn tolerated_advertised_verbs() -> Vec<&'static str> {
    let mut verbs = vec!["restart_mpu"];
    verbs.extend(legacy_brainstem_convenience_verbs());
    verbs
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CockpitResponse {
    Accepted,
    Rejected { message: String },
    Status(CockpitStatus),
    Capabilities(CockpitCapabilities),
    Events(EventBatch),
    NetworkEndpointRegistered(NetworkEndpointRegistered),
    ControlLeaseGranted(ControlLease),
    ServiceLeaseGranted(ServiceLease),
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CommandAck {
    pub accepted: bool,
    pub command_id: u32,
    pub reason: String,
}

fn expect_accepted(response: CockpitResponse) -> Result<()> {
    match response {
        CockpitResponse::Accepted => Ok(()),
        CockpitResponse::Rejected { message } => Err(CockpitError::Rejected {
            command_id: 0,
            reason: message,
        }),
        other => Err(CockpitError::BadResponse(format!("{other:?}"))),
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct BrowserBridgeEnvelope {
    pub command_id: u32,
    pub request: CockpitRequest,
}
