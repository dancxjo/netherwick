#![no_std]

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
    matches!(transport, TransportKind::UsbCdc | TransportKind::Http)
        && (matches!(
            (role, purpose),
            (EndpointRole::Motherbrain, SessionPurpose::Control)
        ) || matches!(role, EndpointRole::ServiceTool))
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
}
