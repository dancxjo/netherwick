use core::fmt::Write;
#[cfg(not(feature = "pico-w"))]
use core::sync::atomic::{AtomicU32, Ordering};
#[cfg(feature = "pico-w")]
use portable_atomic::{AtomicU32, Ordering};

use heapless::{String, Vec};
pub use pete_cockpit_protocol::HandshakeRejectReason as RejectReason;
pub use pete_cockpit_protocol::{EndpointRole, SessionPurpose};
use serde::Deserialize;

pub const MAX_ID_LEN: usize = 64;
pub const MAX_FEATURES: usize = 8;

static GENERATION: AtomicU32 = AtomicU32::new(0);
static LAST_HELLO_HASH: AtomicU32 = AtomicU32::new(0);
static LAST_SESSION_HASH: AtomicU32 = AtomicU32::new(0);
static LAST_GENERATION: AtomicU32 = AtomicU32::new(0);

#[derive(Clone, Debug, Deserialize)]
pub struct Hello {
    pub role: EndpointRole,
    #[serde(default)]
    pub session_purpose: SessionPurpose,
    pub device_id: String<MAX_ID_LEN>,
    pub boot_id: String<MAX_ID_LEN>,
    pub handshake_nonce: String<MAX_ID_LEN>,
    pub protocol_major: u16,
    pub protocol_minor_min: u16,
    pub protocol_minor_max: u16,
    #[serde(default)]
    pub supported_features: Vec<String<32>, MAX_FEATURES>,
    #[serde(default)]
    pub required_features: Vec<String<32>, MAX_FEATURES>,
    pub preferred_heartbeat_ms: u32,
}

#[derive(Clone, Debug)]
pub struct AcceptedHello {
    pub generation: u32,
    pub session_hash: u32,
    pub session_id: String<24>,
    pub negotiated_minor: u16,
}

pub fn parse_json(body: &str) -> Result<Hello, RejectReason> {
    let (hello, used) =
        serde_json_core::from_str::<Hello>(body).map_err(|_| RejectReason::InvalidIdentity)?;
    if !body[used..].trim().is_empty() {
        return Err(RejectReason::InvalidIdentity);
    }
    Ok(hello)
}

pub fn validate(
    hello: &Hello,
    brainstem_device: &str,
    brainstem_boot: &str,
) -> Result<AcceptedHello, RejectReason> {
    if !pete_cockpit_protocol::brainstem_accepts_client_role(hello.role) {
        return Err(RejectReason::WrongRole);
    }
    if !valid_token(&hello.device_id)
        || !valid_token(&hello.boot_id)
        || !valid_token(&hello.handshake_nonce)
    {
        return Err(RejectReason::InvalidIdentity);
    }
    let negotiated_minor = pete_cockpit_protocol::negotiate_minor(
        hello.protocol_major,
        hello.protocol_minor_min,
        hello.protocol_minor_max,
    )?;
    if !hello
        .supported_features
        .iter()
        .any(|feature| feature.as_str() == pete_cockpit_protocol::FEATURE_SESSION_IDS)
    {
        return Err(RejectReason::MissingRequiredFeature);
    }
    for feature in &hello.required_features {
        if !matches!(
            feature.as_str(),
            "session_ids" | "event_cursor" | "heartbeat" | "transport_failover"
        ) {
            return Err(RejectReason::MissingRequiredFeature);
        }
    }
    let mut hash = 0x811c_9dc5;
    hash = fnv(hash, &[hello.role as u8, hello.session_purpose as u8]);
    for value in [
        hello.device_id.as_str(),
        hello.boot_id.as_str(),
        brainstem_device,
        brainstem_boot,
        hello.handshake_nonce.as_str(),
    ] {
        hash = fnv(hash, value.as_bytes());
    }
    let hello_hash = hash.max(1);
    let cached_generation = LAST_GENERATION.load(Ordering::Acquire);
    if LAST_HELLO_HASH.load(Ordering::Acquire) == hello_hash && cached_generation != 0 {
        let session_hash = LAST_SESSION_HASH.load(Ordering::Acquire);
        let mut session_id = String::new();
        let _ = write!(
            session_id,
            "sess-{session_hash:08x}-{cached_generation:08x}"
        );
        return Ok(AcceptedHello {
            generation: cached_generation,
            session_hash,
            session_id,
            negotiated_minor,
        });
    }
    let generation = GENERATION
        .fetch_add(1, Ordering::Relaxed)
        .wrapping_add(1)
        .max(1);
    hash = fnv(hash, &generation.to_le_bytes());
    let mut session_id = String::new();
    let _ = write!(session_id, "sess-{hash:08x}-{generation:08x}");
    LAST_SESSION_HASH.store(hash.max(1), Ordering::Release);
    LAST_GENERATION.store(generation, Ordering::Release);
    LAST_HELLO_HASH.store(hello_hash, Ordering::Release);
    Ok(AcceptedHello {
        generation,
        session_hash: hash.max(1),
        session_id,
        negotiated_minor,
    })
}

pub fn token_hash(token: &str) -> u32 {
    fnv(0x811c_9dc5, token.as_bytes()).max(1)
}

fn fnv(mut hash: u32, bytes: &[u8]) -> u32 {
    for byte in bytes {
        hash ^= *byte as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

fn valid_token<const N: usize>(value: &String<N>) -> bool {
    pete_cockpit_protocol::valid_identity_token(value.as_bytes(), N)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hello(role: &str) -> String<512> {
        let mut value = String::new();
        write!(value, "{{\"role\":\"{role}\",\"device_id\":\"motherbrain-primary\",\"boot_id\":\"boot-1\",\"handshake_nonce\":\"nonce-1\",\"protocol_major\":1,\"protocol_minor_min\":0,\"protocol_minor_max\":0,\"supported_features\":[\"session_ids\"],\"required_features\":[\"session_ids\"],\"preferred_heartbeat_ms\":500}}").unwrap();
        value
    }

    #[test]
    fn accepts_motherbrain_and_rejects_other_roles() {
        let parsed = parse_json(&hello("motherbrain")).unwrap();
        assert!(validate(&parsed, "brainstem-7k2m", "bsboot-1").is_ok());
        assert!(validate(
            &parse_json(&hello("forebrain")).unwrap(),
            "brainstem-7k2m",
            "bsboot-1"
        )
        .is_ok());
        let parsed = parse_json(&hello("brainstem")).unwrap();
        assert_eq!(
            validate(&parsed, "brainstem-7k2m", "bsboot-1").unwrap_err(),
            RejectReason::WrongRole
        );
    }

    #[test]
    fn duplicate_hello_reuses_session_and_missing_required_feature_is_rejected() {
        let parsed = parse_json(&hello("motherbrain")).unwrap();
        let first = validate(&parsed, "brainstem-7k2m", "bsboot-duplicate").unwrap();
        let duplicate = validate(&parsed, "brainstem-7k2m", "bsboot-duplicate").unwrap();
        assert_eq!(first.session_id, duplicate.session_id);
        assert_eq!(first.generation, duplicate.generation);

        let mut unsupported = parsed;
        unsupported.supported_features.clear();
        assert_eq!(
            validate(&unsupported, "brainstem-7k2m", "bsboot-feature").unwrap_err(),
            RejectReason::MissingRequiredFeature
        );
    }
}
