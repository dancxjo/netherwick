#![no_std]

pub const CONTACT_WITHDRAWAL_DURATION_MS: u32 = 300;
pub const CONTACT_WITHDRAWAL_SPEED_MM_S: i16 = 80;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum EndpointRole {
    Brainstem,
    Motherbrain,
    Forebrain,
    Operator,
    Simulator,
    ServiceTool,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum SessionPurpose {
    #[default]
    Control,
    Diagnostic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum ControlAuthority {
    Motherbrain,
    ForebrainRecovery,
    OperatorDebug,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum AuthorizationClass {
    ReadOnly,
    Emergency,
    Session,
    ControlLease,
    ServiceLease,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum ServiceScope {
    Bootsel,
    RestartCreate,
    ResetMotherbrain,
}

pub const fn role_can_request_control(
    role: EndpointRole,
    purpose: SessionPurpose,
    authority: ControlAuthority,
) -> bool {
    if !matches!(purpose, SessionPurpose::Control) {
        return false;
    }
    matches!(
        (role, authority),
        (EndpointRole::Motherbrain, ControlAuthority::Motherbrain)
            | (EndpointRole::Forebrain, ControlAuthority::ForebrainRecovery)
            | (EndpointRole::Operator, ControlAuthority::OperatorDebug)
    )
}

pub const fn role_can_request_service(
    role: EndpointRole,
    purpose: SessionPurpose,
    transport: TransportKind,
) -> bool {
    matches!(
        (role, purpose, transport),
        (
            EndpointRole::Motherbrain,
            SessionPurpose::Control,
            TransportKind::UsbCdc | TransportKind::Http,
        ) | (EndpointRole::ServiceTool, _, TransportKind::UsbCdc,)
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum HandshakeRejectReason {
    WrongRole,
    ProtocolMajorMismatch,
    ProtocolMinorIncompatible,
    MissingRequiredFeature,
    InvalidIdentity,
    Busy,
    InternalError,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum CommandRejectReason {
    Busy,
    Charging,
    StaleSequence,
    Unsupported,
    HazardMismatch,
    EscapeEnvelope,
    AbsoluteHazard,
    Unknown(u8),
}

impl CommandRejectReason {
    pub const fn from_code(code: u8) -> Self {
        match code {
            1 => Self::Busy,
            2 => Self::Charging,
            3 => Self::StaleSequence,
            4 => Self::Unsupported,
            5 => Self::HazardMismatch,
            6 => Self::EscapeEnvelope,
            7 => Self::AbsoluteHazard,
            other => Self::Unknown(other),
        }
    }

    pub const fn code(self) -> u8 {
        match self {
            Self::Busy => 1,
            Self::Charging => 2,
            Self::StaleSequence => 3,
            Self::Unsupported => 4,
            Self::HazardMismatch => 5,
            Self::EscapeEnvelope => 6,
            Self::AbsoluteHazard => 7,
            Self::Unknown(code) => code,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Busy => "busy",
            Self::Charging => "charging_busy",
            Self::StaleSequence => "stale_sequence",
            Self::Unsupported => "unsupported",
            Self::HazardMismatch => "hazard_mismatch",
            Self::EscapeEnvelope => "escape_envelope",
            Self::AbsoluteHazard => "absolute_hazard",
            Self::Unknown(_) => "unknown",
        }
    }
}

impl HandshakeRejectReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::WrongRole => "wrong_role",
            Self::ProtocolMajorMismatch => "protocol_major_mismatch",
            Self::ProtocolMinorIncompatible => "protocol_minor_incompatible",
            Self::MissingRequiredFeature => "missing_required_feature",
            Self::InvalidIdentity => "invalid_identity",
            Self::Busy => "busy",
            Self::InternalError => "internal_error",
        }
    }
    pub const fn code(self) -> u32 {
        match self {
            Self::WrongRole => 1,
            Self::ProtocolMajorMismatch => 2,
            Self::ProtocolMinorIncompatible => 3,
            Self::MissingRequiredFeature => 4,
            Self::InvalidIdentity => 5,
            Self::Busy => 6,
            Self::InternalError => 7,
        }
    }
}

pub const PROTOCOL_MAJOR: u16 = 1;
pub const PROTOCOL_MINOR_MIN: u16 = 0;
pub const PROTOCOL_MINOR_MAX: u16 = 0;
pub const FEATURE_SESSION_IDS: &str = "session_ids";
pub const FEATURE_EVENT_CURSOR: &str = "event_cursor";
pub const FEATURE_HEARTBEAT: &str = "heartbeat";
pub const FEATURE_TRANSPORT_FAILOVER: &str = "transport_failover";
pub const FEATURE_CAPABILITY_DIGEST: &str = "capability_digest";

pub const fn brainstem_accepts_client_role(role: EndpointRole) -> bool {
    !matches!(role, EndpointRole::Brainstem | EndpointRole::Simulator)
}

pub const fn negotiate_minor(
    client_major: u16,
    client_minor_min: u16,
    client_minor_max: u16,
) -> Result<u16, HandshakeRejectReason> {
    if client_major != PROTOCOL_MAJOR {
        return Err(HandshakeRejectReason::ProtocolMajorMismatch);
    }
    let low = if client_minor_min > PROTOCOL_MINOR_MIN {
        client_minor_min
    } else {
        PROTOCOL_MINOR_MIN
    };
    let high = if client_minor_max < PROTOCOL_MINOR_MAX {
        client_minor_max
    } else {
        PROTOCOL_MINOR_MAX
    };
    if low > high {
        Err(HandshakeRejectReason::ProtocolMinorIncompatible)
    } else {
        Ok(high)
    }
}

pub fn valid_identity_token(value: &[u8], max_len: usize) -> bool {
    !value.is_empty()
        && value.len() <= max_len
        && value
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TransportKind {
    Http = 1,
    WebSocket = 2,
    HardwareUart = 3,
    Udp = 4,
    UsbCdc = 5,
    Simulator = 6,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_negotiation_and_identity_rules_are_fail_closed() {
        assert_eq!(negotiate_minor(1, 0, 0), Ok(0));
        assert_eq!(
            negotiate_minor(2, 0, 0),
            Err(HandshakeRejectReason::ProtocolMajorMismatch)
        );
        assert!(!brainstem_accepts_client_role(EndpointRole::Brainstem));
        assert!(brainstem_accepts_client_role(EndpointRole::Motherbrain));
        assert!(valid_identity_token(b"motherbrain-primary", 64));
        assert!(!valid_identity_token(b"motherbrain primary", 64));
        assert!(role_can_request_control(
            EndpointRole::Motherbrain,
            SessionPurpose::Control,
            ControlAuthority::Motherbrain
        ));
        assert!(!role_can_request_control(
            EndpointRole::Motherbrain,
            SessionPurpose::Diagnostic,
            ControlAuthority::Motherbrain
        ));
        assert!(role_can_request_service(
            EndpointRole::ServiceTool,
            SessionPurpose::Diagnostic,
            TransportKind::UsbCdc
        ));
        assert!(!role_can_request_service(
            EndpointRole::ServiceTool,
            SessionPurpose::Diagnostic,
            TransportKind::Http
        ));
    }

    #[test]
    fn command_rejection_codes_round_trip() {
        for reason in [
            CommandRejectReason::Busy,
            CommandRejectReason::Charging,
            CommandRejectReason::StaleSequence,
            CommandRejectReason::Unsupported,
        ] {
            assert_eq!(CommandRejectReason::from_code(reason.code()), reason);
        }
        assert_eq!(
            CommandRejectReason::from_code(99),
            CommandRejectReason::Unknown(99)
        );
    }
}
