use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

pub use pete_cockpit_protocol::{
    AuthorizationClass, ControlAuthority, EndpointRole, HandshakeRejectReason, ServiceScope,
    SessionPurpose, PROTOCOL_MAJOR, PROTOCOL_MINOR_MAX, PROTOCOL_MINOR_MIN,
};
use serde::{Deserialize, Serialize};

use crate::{CockpitCapabilities, CockpitContract, CockpitError, Result};

pub const MAX_COMPACT_HANDSHAKE_FRAME_LEN: usize = 4096;

#[derive(Debug, Clone)]
pub struct CompactLineDecoder {
    buffer: Vec<u8>,
    max_len: usize,
}

impl CompactLineDecoder {
    pub fn new(max_len: usize) -> Self {
        Self {
            buffer: Vec::new(),
            max_len,
        }
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>> {
        let mut lines = Vec::new();
        for byte in bytes {
            match *byte {
                b'\r' => {}
                b'\n' => {
                    let line = std::str::from_utf8(&self.buffer)
                        .map_err(|_| {
                            CockpitError::BadResponse("compact frame was not utf-8".into())
                        })?
                        .to_owned();
                    self.buffer.clear();
                    if !line.is_empty() {
                        lines.push(line);
                    }
                }
                byte => {
                    if self.buffer.len() >= self.max_len {
                        self.buffer.clear();
                        return Err(CockpitError::FrameTooLarge { max: self.max_len });
                    }
                    self.buffer.push(byte);
                }
            }
        }
        Ok(lines)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum HandshakeFeature {
    SessionIds,
    EventCursor,
    Heartbeat,
    TransportFailover,
    CapabilityDigest,
    Unknown(String),
}

impl HandshakeFeature {
    pub fn as_str(&self) -> &str {
        match self {
            Self::SessionIds => pete_cockpit_protocol::FEATURE_SESSION_IDS,
            Self::EventCursor => pete_cockpit_protocol::FEATURE_EVENT_CURSOR,
            Self::Heartbeat => pete_cockpit_protocol::FEATURE_HEARTBEAT,
            Self::TransportFailover => pete_cockpit_protocol::FEATURE_TRANSPORT_FAILOVER,
            Self::CapabilityDigest => pete_cockpit_protocol::FEATURE_CAPABILITY_DIGEST,
            Self::Unknown(value) => value,
        }
    }
}

impl Serialize for HandshakeFeature {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for HandshakeFeature {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            pete_cockpit_protocol::FEATURE_SESSION_IDS => Self::SessionIds,
            pete_cockpit_protocol::FEATURE_EVENT_CURSOR => Self::EventCursor,
            pete_cockpit_protocol::FEATURE_HEARTBEAT => Self::Heartbeat,
            pete_cockpit_protocol::FEATURE_TRANSPORT_FAILOVER => Self::TransportFailover,
            pete_cockpit_protocol::FEATURE_CAPABILITY_DIGEST => Self::CapabilityDigest,
            _ => Self::Unknown(value),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SoftwareInfo {
    pub software_name: String,
    #[serde(default)]
    pub software_version: String,
    pub build_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HandshakeHello {
    pub role: EndpointRole,
    #[serde(default)]
    pub session_purpose: SessionPurpose,
    pub device_id: String,
    pub boot_id: String,
    pub handshake_nonce: String,
    pub protocol_major: u16,
    pub protocol_minor_min: u16,
    pub protocol_minor_max: u16,
    #[serde(default)]
    pub supported_features: Vec<HandshakeFeature>,
    #[serde(default)]
    pub required_features: Vec<HandshakeFeature>,
    pub preferred_heartbeat_ms: u32,
    pub software: SoftwareInfo,
}

impl HandshakeHello {
    pub fn default_motherbrain() -> Self {
        let configured = std::env::var("PETE_MOTHERBRAIN_DEVICE_ID")
            .ok()
            .filter(|value| valid_id(value));
        let device_id = configured.unwrap_or_else(|| {
            let installation =
                std::fs::read_to_string("/etc/machine-id").unwrap_or_else(|_| "primary".into());
            let mut hash = DefaultHasher::new();
            installation.trim().hash(&mut hash);
            format!("pete-motherbrain-{:08x}", hash.finish() as u32)
        });
        Self::motherbrain(device_id)
    }

    pub fn motherbrain(device_id: impl Into<String>) -> Self {
        Self {
            role: EndpointRole::Motherbrain,
            session_purpose: SessionPurpose::Control,
            device_id: device_id.into(),
            boot_id: process_boot_id(),
            handshake_nonce: fresh_id("hello"),
            protocol_major: PROTOCOL_MAJOR,
            protocol_minor_min: PROTOCOL_MINOR_MIN,
            protocol_minor_max: PROTOCOL_MINOR_MAX,
            supported_features: default_features(),
            required_features: vec![HandshakeFeature::SessionIds],
            preferred_heartbeat_ms: 500,
            software: SoftwareInfo {
                software_name: "pete-motherbrain".into(),
                software_version: env!("CARGO_PKG_VERSION").into(),
                build_id: option_env!("PETE_BUILD_ID").unwrap_or("development").into(),
            },
        }
    }

    pub fn forebrain(device_id: impl Into<String>) -> Self {
        let mut hello = Self::motherbrain(device_id);
        hello.role = EndpointRole::Forebrain;
        hello.session_purpose = SessionPurpose::Diagnostic;
        hello.software.software_name = "pete-forebrain".into();
        hello
    }

    pub fn operator(device_id: impl Into<String>) -> Self {
        let mut hello = Self::motherbrain(device_id);
        hello.role = EndpointRole::Operator;
        hello.session_purpose = SessionPurpose::Diagnostic;
        hello.software.software_name = "pete-operator".into();
        hello
    }

    pub fn new_attempt(&self) -> Self {
        let mut next = self.clone();
        next.handshake_nonce = fresh_id("hello");
        next
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SafetySnapshot {
    pub armed: bool,
    pub estop_latched: bool,
    pub safety_tripped: bool,
    pub active_motion: bool,
    pub runtime_state: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HandshakeWelcome {
    pub role: EndpointRole,
    pub device_id: String,
    pub boot_id: String,
    pub echoed_handshake_nonce: String,
    pub session_id: String,
    pub protocol_major: u16,
    pub protocol_minor: u16,
    pub supported_features: Vec<HandshakeFeature>,
    pub required_features: Vec<HandshakeFeature>,
    pub heartbeat_min_ms: u32,
    pub heartbeat_max_ms: u32,
    pub command_ttl_min_ms: u32,
    pub command_ttl_max_ms: u32,
    pub current_event_next_seq: u32,
    pub capability_contract: CockpitCapabilities,
    pub software: SoftwareInfo,
    pub safety_snapshot: SafetySnapshot,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HandshakeReject {
    pub echoed_handshake_nonce: String,
    pub reason_code: HandshakeRejectReason,
    pub message: String,
    pub supported_protocol_major: u16,
    pub supported_minor_min: u16,
    pub supported_minor_max: u16,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HandshakeResponse {
    Welcome(HandshakeWelcome),
    Reject(HandshakeReject),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CockpitSession {
    pub session_id: String,
    pub peer_device_id: String,
    pub peer_boot_id: String,
    pub local_device_id: String,
    pub local_boot_id: String,
    pub local_role: EndpointRole,
    #[serde(default)]
    pub local_purpose: SessionPurpose,
    pub protocol_major: u16,
    pub protocol_minor: u16,
    pub negotiated_features: Vec<HandshakeFeature>,
    pub heartbeat_ms: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ControlLease {
    pub lease_id: String,
    pub session_id: String,
    pub owner_role: EndpointRole,
    pub authority: ControlAuthority,
    pub ttl_ms: u32,
    pub generation: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ServiceLease {
    pub lease_id: String,
    pub session_id: String,
    pub owner_role: EndpointRole,
    pub scope: ServiceScope,
    pub ttl_ms: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReconnectClassification {
    Initial,
    TransportReconnect,
    BrainstemReboot,
    ReplacementBrainstem,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HandshakeOutcome {
    pub welcome: HandshakeWelcome,
    pub session: CockpitSession,
    pub contract: CockpitContract,
    pub event_cursor: u32,
    pub classification: ReconnectClassification,
}

impl HandshakeOutcome {
    pub fn validate(hello: &HandshakeHello, response: HandshakeResponse) -> Result<Self> {
        let welcome = match response {
            HandshakeResponse::Reject(reject) => {
                return Err(CockpitError::HandshakeRejected(reject))
            }
            HandshakeResponse::Welcome(welcome) => welcome,
        };
        if welcome.echoed_handshake_nonce != hello.handshake_nonce {
            return Err(CockpitError::StaleHandshake {
                expected: hello.handshake_nonce.clone(),
                received: welcome.echoed_handshake_nonce,
            });
        }
        if welcome.role != EndpointRole::Brainstem
            || welcome.protocol_major != hello.protocol_major
            || welcome.protocol_minor < hello.protocol_minor_min
            || welcome.protocol_minor > hello.protocol_minor_max
        {
            return Err(CockpitError::BadResponse(
                "welcome contains incompatible negotiated values".into(),
            ));
        }
        for feature in &welcome.required_features {
            if !hello.supported_features.contains(feature) {
                return Err(CockpitError::BadResponse(format!(
                    "brainstem requires unsupported feature {feature:?}"
                )));
            }
        }
        let heartbeat_ms = hello
            .preferred_heartbeat_ms
            .clamp(welcome.heartbeat_min_ms, welcome.heartbeat_max_ms);
        let contract = welcome.capability_contract.contract();
        let session = CockpitSession {
            session_id: welcome.session_id.clone(),
            peer_device_id: welcome.device_id.clone(),
            peer_boot_id: welcome.boot_id.clone(),
            local_device_id: hello.device_id.clone(),
            local_boot_id: hello.boot_id.clone(),
            local_role: hello.role,
            local_purpose: hello.session_purpose,
            protocol_major: welcome.protocol_major,
            protocol_minor: welcome.protocol_minor,
            negotiated_features: welcome
                .supported_features
                .iter()
                .filter(|feature| hello.supported_features.contains(feature))
                .cloned()
                .collect(),
            heartbeat_ms,
        };
        Ok(Self {
            event_cursor: welcome.current_event_next_seq,
            welcome,
            session,
            contract,
            classification: ReconnectClassification::Initial,
        })
    }

    pub fn classify_against(mut self, prior: Option<&CockpitSession>) -> Self {
        self.classification = match prior {
            None => ReconnectClassification::Initial,
            Some(prior) if prior.peer_device_id != self.session.peer_device_id => {
                ReconnectClassification::ReplacementBrainstem
            }
            Some(prior) if prior.peer_boot_id != self.session.peer_boot_id => {
                ReconnectClassification::BrainstemReboot
            }
            Some(_) => ReconnectClassification::TransportReconnect,
        };
        self
    }
}

pub fn default_features() -> Vec<HandshakeFeature> {
    vec![
        HandshakeFeature::SessionIds,
        HandshakeFeature::EventCursor,
        HandshakeFeature::Heartbeat,
        HandshakeFeature::TransportFailover,
    ]
}

pub fn encode_compact_hello(hello: &HandshakeHello) -> Result<String> {
    encode_compact("HELLO", hello)
}

pub fn decode_compact_hello(frame: &str) -> Result<HandshakeHello> {
    decode_compact("HELLO", frame)
}

pub fn encode_compact_response(response: &HandshakeResponse) -> Result<String> {
    let kind = match response {
        HandshakeResponse::Welcome(_) => "WELCOME",
        HandshakeResponse::Reject(_) => "REJECT",
    };
    encode_compact(kind, response)
}

pub fn decode_compact_response(frame: &str) -> Result<HandshakeResponse> {
    let kind = frame.split_ascii_whitespace().next().unwrap_or("");
    if kind != "WELCOME" && kind != "REJECT" {
        return Err(CockpitError::BadResponse(
            "expected WELCOME or REJECT frame".into(),
        ));
    }
    decode_compact(kind, frame)
}

fn encode_compact<T: Serialize>(kind: &str, value: &T) -> Result<String> {
    let frame = format!("{kind} {}\n", serde_json::to_string(value)?);
    if frame.len() > MAX_COMPACT_HANDSHAKE_FRAME_LEN {
        return Err(CockpitError::FrameTooLarge {
            max: MAX_COMPACT_HANDSHAKE_FRAME_LEN,
        });
    }
    Ok(frame)
}

fn decode_compact<T: for<'de> Deserialize<'de>>(kind: &str, frame: &str) -> Result<T> {
    if frame.len() > MAX_COMPACT_HANDSHAKE_FRAME_LEN {
        return Err(CockpitError::FrameTooLarge {
            max: MAX_COMPACT_HANDSHAKE_FRAME_LEN,
        });
    }
    let frame = frame.trim_end_matches(['\r', '\n']);
    let payload = frame
        .strip_prefix(kind)
        .and_then(|rest| rest.strip_prefix(' '))
        .ok_or_else(|| CockpitError::BadResponse(format!("expected {kind} frame")))?;
    Ok(serde_json::from_str(payload)?)
}

pub fn negotiate(
    hello: &HandshakeHello,
    brainstem_device_id: &str,
    brainstem_boot_id: &str,
    capabilities: CockpitCapabilities,
    safety_snapshot: SafetySnapshot,
    current_event_next_seq: u32,
    software: SoftwareInfo,
    session_serial: u64,
) -> HandshakeResponse {
    let reject = |reason_code, message: &str| {
        HandshakeResponse::Reject(HandshakeReject {
            echoed_handshake_nonce: hello.handshake_nonce.clone(),
            reason_code,
            message: message.into(),
            supported_protocol_major: PROTOCOL_MAJOR,
            supported_minor_min: PROTOCOL_MINOR_MIN,
            supported_minor_max: PROTOCOL_MINOR_MAX,
        })
    };
    if !pete_cockpit_protocol::brainstem_accepts_client_role(hello.role) {
        return reject(
            HandshakeRejectReason::WrongRole,
            "brainstem and simulator roles cannot claim a client session",
        );
    }
    if !valid_id(&hello.device_id) || !valid_id(&hello.boot_id) || !valid_id(&hello.handshake_nonce)
    {
        return reject(
            HandshakeRejectReason::InvalidIdentity,
            "identity fields must be bounded printable tokens",
        );
    }
    let high = match pete_cockpit_protocol::negotiate_minor(
        hello.protocol_major,
        hello.protocol_minor_min,
        hello.protocol_minor_max,
    ) {
        Ok(minor) => minor,
        Err(reason) => return reject(reason, "protocol versions are incompatible"),
    };
    let supported = default_features();
    if hello
        .required_features
        .iter()
        .any(|feature| !supported.contains(feature))
    {
        return reject(
            HandshakeRejectReason::MissingRequiredFeature,
            "required feature is unavailable",
        );
    }
    let session_id = derive_session_id(
        hello,
        brainstem_device_id,
        brainstem_boot_id,
        session_serial,
    );
    HandshakeResponse::Welcome(HandshakeWelcome {
        role: EndpointRole::Brainstem,
        device_id: brainstem_device_id.into(),
        boot_id: brainstem_boot_id.into(),
        echoed_handshake_nonce: hello.handshake_nonce.clone(),
        session_id,
        protocol_major: PROTOCOL_MAJOR,
        protocol_minor: high,
        supported_features: supported,
        required_features: vec![HandshakeFeature::SessionIds],
        heartbeat_min_ms: 250,
        heartbeat_max_ms: 2_000,
        command_ttl_min_ms: capabilities.limits.min_ttl_ms,
        command_ttl_max_ms: capabilities.limits.max_ttl_ms,
        current_event_next_seq,
        capability_contract: capabilities,
        software,
        safety_snapshot,
    })
}

fn valid_id(value: &str) -> bool {
    pete_cockpit_protocol::valid_identity_token(value.as_bytes(), 64)
}

fn derive_session_id(
    hello: &HandshakeHello,
    device_id: &str,
    boot_id: &str,
    serial: u64,
) -> String {
    let mut hash = DefaultHasher::new();
    hello.device_id.hash(&mut hash);
    hello.role.hash(&mut hash);
    hello.session_purpose.hash(&mut hash);
    hello.boot_id.hash(&mut hash);
    device_id.hash(&mut hash);
    boot_id.hash(&mut hash);
    hello.handshake_nonce.hash(&mut hash);
    serial.hash(&mut hash);
    format!("sess-{:016x}", hash.finish())
}

fn fresh_id(prefix: &str) -> String {
    static SERIAL: AtomicU64 = AtomicU64::new(1);
    let serial = SERIAL.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{}-{serial}", uuid::Uuid::new_v4().simple())
}

fn process_boot_id() -> String {
    static BOOT_ID: OnceLock<String> = OnceLock::new();
    BOOT_ID.get_or_init(|| fresh_id("mbboot")).clone()
}

#[cfg(test)]
mod tests {
    use super::SoftwareInfo;

    #[test]
    fn legacy_software_info_without_version_still_decodes() {
        let software: SoftwareInfo = serde_json::from_str(
            r#"{"software_name":"pete-brainstem","build_id":"legacy-build"}"#,
        )
        .unwrap();

        assert_eq!(software.software_name, "pete-brainstem");
        assert_eq!(software.software_version, "");
        assert_eq!(software.build_id, "legacy-build");
    }
}
