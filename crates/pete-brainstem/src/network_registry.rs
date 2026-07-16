use crate::status;
use portable_atomic::{AtomicU32, Ordering};

const LEASES: usize = 8;
static LEASE_ID: [AtomicU32; LEASES] = [const { AtomicU32::new(0) }; LEASES];
static LEASE_IP: [AtomicU32; LEASES] = [const { AtomicU32::new(0) }; LEASES];
static LEASE_EXPIRY_MS: [AtomicU32; LEASES] = [const { AtomicU32::new(0) }; LEASES];
static MOTHERBRAIN_IP: AtomicU32 = AtomicU32::new(0);
static MOTHERBRAIN_EXPIRY_MS: AtomicU32 = AtomicU32::new(0);
static REGISTRATION_GENERATION: AtomicU32 = AtomicU32::new(0);
static MOTHERBRAIN_LEASE_ID: AtomicU32 = AtomicU32::new(0);
static MOTHERBRAIN_DEVICE_HASH: AtomicU32 = AtomicU32::new(0);
static MOTHERBRAIN_BOOT_HASH: AtomicU32 = AtomicU32::new(0);

pub fn identity_hash(identity: &[u8]) -> u32 {
    let mut hash = 0x811c_9dc5u32;
    for byte in identity {
        hash ^= *byte as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash.max(1)
}

pub fn record_lease(identity: &[u8], ip: [u8; 4], expiry_ms: u32) {
    let id = dhcp_identity_hash(identity);
    record_lease_hash(id, ip, expiry_ms);
}

pub fn dhcp_identity_hash(identity: &[u8]) -> u32 {
    let mut id = 0x811c_9dc5u32;
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in identity {
        id = hash_byte(id, HEX[(byte >> 4) as usize]);
        id = hash_byte(id, HEX[(byte & 0x0f) as usize]);
    }
    id.max(1)
}

fn record_lease_hash(id: u32, ip: [u8; 4], expiry_ms: u32) {
    let slot = LEASE_ID
        .iter()
        .position(|value| value.load(Ordering::Acquire) == id)
        .or_else(|| {
            LEASE_ID
                .iter()
                .position(|value| value.load(Ordering::Acquire) == 0)
        })
        .unwrap_or((id as usize) % LEASES);
    LEASE_IP[slot].store(u32::from_be_bytes(ip), Ordering::Release);
    LEASE_EXPIRY_MS[slot].store(expiry_ms, Ordering::Release);
    LEASE_ID[slot].store(id, Ordering::Release);
    status::mark_dhcp_lease_changed(id, u32::from_be_bytes(ip));
}

pub fn lease_identity_is_motherbrain(identity: &[u8]) -> bool {
    let registered = MOTHERBRAIN_LEASE_ID.load(Ordering::Acquire);
    registered != 0 && registered == dhcp_identity_hash(identity)
}

fn hash_byte(mut hash: u32, byte: u8) -> u32 {
    hash ^= byte as u32;
    hash.wrapping_mul(0x0100_0193)
}

pub fn register_motherbrain(
    identity: &[u8],
    ip: [u8; 4],
    peer_device_hash: u32,
    peer_boot_hash: u32,
    requested_ttl_seconds: u32,
    now_ms: u32,
) -> Option<(u32, u32)> {
    let id = identity_hash(identity);
    let slot = LEASE_ID
        .iter()
        .position(|value| value.load(Ordering::Acquire) == id)?;
    if LEASE_IP[slot].load(Ordering::Acquire) != u32::from_be_bytes(ip) {
        return None;
    }
    let lease_expiry = LEASE_EXPIRY_MS[slot].load(Ordering::Acquire);
    if time_reached(now_ms, lease_expiry) {
        return None;
    }
    let lease_remaining = lease_expiry.wrapping_sub(now_ms) / 1_000;
    let ttl = requested_ttl_seconds.min(lease_remaining).max(1);
    let same_registration = MOTHERBRAIN_IP.load(Ordering::Acquire) == u32::from_be_bytes(ip)
        && MOTHERBRAIN_LEASE_ID.load(Ordering::Acquire) == id
        && MOTHERBRAIN_DEVICE_HASH.load(Ordering::Acquire) == peer_device_hash
        && MOTHERBRAIN_BOOT_HASH.load(Ordering::Acquire) == peer_boot_hash
        && !time_reached(now_ms, MOTHERBRAIN_EXPIRY_MS.load(Ordering::Acquire));
    let generation = if same_registration {
        REGISTRATION_GENERATION.load(Ordering::Acquire).max(1)
    } else {
        REGISTRATION_GENERATION
            .fetch_add(1, Ordering::Relaxed)
            .wrapping_add(1)
            .max(1)
    };
    MOTHERBRAIN_IP.store(u32::from_be_bytes(ip), Ordering::Release);
    MOTHERBRAIN_LEASE_ID.store(id, Ordering::Release);
    MOTHERBRAIN_DEVICE_HASH.store(peer_device_hash, Ordering::Release);
    MOTHERBRAIN_BOOT_HASH.store(peer_boot_hash, Ordering::Release);
    MOTHERBRAIN_EXPIRY_MS.store(now_ms.wrapping_add(ttl * 1_000), Ordering::Release);
    status::mark_dns_registration_changed(generation, u32::from_be_bytes(ip));
    Some((ttl, generation))
}

pub fn resolve_motherbrain(now_ms: u32) -> Option<[u8; 4]> {
    let ip = MOTHERBRAIN_IP.load(Ordering::Acquire);
    if ip == 0 || time_reached(now_ms, MOTHERBRAIN_EXPIRY_MS.load(Ordering::Acquire)) {
        return None;
    }
    Some(ip.to_be_bytes())
}
pub fn clear_motherbrain_registration() {
    let previous = MOTHERBRAIN_IP.load(Ordering::Acquire);
    MOTHERBRAIN_IP.store(0, Ordering::Release);
    MOTHERBRAIN_EXPIRY_MS.store(0, Ordering::Release);
    MOTHERBRAIN_LEASE_ID.store(0, Ordering::Release);
    MOTHERBRAIN_DEVICE_HASH.store(0, Ordering::Release);
    MOTHERBRAIN_BOOT_HASH.store(0, Ordering::Release);
    if previous != 0 {
        status::mark_dns_registration_changed(0, previous);
    }
}

fn time_reached(now: u32, deadline: u32) -> bool {
    now.wrapping_sub(deadline) < u32::MAX / 2
}

pub struct NetworkDiagnostics {
    pub active_leases: u8,
    pub motherbrain_ip: Option<[u8; 4]>,
    pub registration_generation: u32,
}

pub fn diagnostics(now_ms: u32) -> NetworkDiagnostics {
    let active_leases = LEASE_ID
        .iter()
        .zip(LEASE_EXPIRY_MS.iter())
        .filter(|(id, expiry)| {
            id.load(Ordering::Acquire) != 0 && !time_reached(now_ms, expiry.load(Ordering::Acquire))
        })
        .count() as u8;
    NetworkDiagnostics {
        active_leases,
        motherbrain_ip: resolve_motherbrain(now_ms),
        registration_generation: REGISTRATION_GENERATION.load(Ordering::Acquire),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn registration_must_match_live_lease_and_expires() {
        record_lease(b"client-a", [192, 168, 4, 2], 10_000);
        assert!(register_motherbrain(b"wrong", [192, 168, 4, 2], 1, 1, 5, 1_000).is_none());
        let identity = b"636c69656e742d61";
        assert!(register_motherbrain(identity, [192, 168, 4, 3], 1, 1, 5, 1_000).is_none());
        let first = register_motherbrain(identity, [192, 168, 4, 2], 1, 1, 5, 1_000).unwrap();
        let duplicate = register_motherbrain(identity, [192, 168, 4, 2], 1, 1, 5, 1_100).unwrap();
        let rebooted = register_motherbrain(identity, [192, 168, 4, 2], 1, 2, 5, 1_200).unwrap();
        assert_eq!(first.1, duplicate.1);
        assert_ne!(duplicate.1, rebooted.1);
        assert_eq!(resolve_motherbrain(6_199), Some([192, 168, 4, 2]));
        assert_eq!(resolve_motherbrain(6_200), None);
    }
}
