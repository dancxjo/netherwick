const AP_NETWORK: [u8; 3] = [192, 168, 4];
const DHCP_POOL_FIRST_HOST: u8 = 2;
const DHCP_POOL_SIZE: usize = 8;
pub const DHCP_LEASE_SECONDS: u32 = 3_600;
const DHCP_OFFER_HOLD_SECONDS: u32 = 30;
use crate::network_registry;

#[derive(Clone, Copy, Eq)]
pub struct DhcpClient {
    hardware_address: [u8; 6],
    client_identifier: [u8; 32],
    client_identifier_len: u8,
    requested_hostname: [u8; 32],
    requested_hostname_len: u8,
}

impl PartialEq for DhcpClient {
    fn eq(&self, other: &Self) -> bool {
        self.lease_identity() == other.lease_identity()
    }
}

pub fn hostname_is_reserved(hostname: &[u8]) -> bool {
    [
        b"pete".as_slice(),
        b"brainstem",
        b"motherbrain",
        b"forebrain",
        b"gateway",
        b"control",
    ]
    .iter()
    .any(|reserved| hostname.eq_ignore_ascii_case(reserved))
}

impl DhcpClient {
    pub const fn new(hardware_address: [u8; 6]) -> Self {
        Self {
            hardware_address,
            client_identifier: [0; 32],
            client_identifier_len: 0,
            requested_hostname: [0; 32],
            requested_hostname_len: 0,
        }
    }

    pub fn with_metadata(mut self, client_identifier: &[u8], requested_hostname: &[u8]) -> Self {
        let id_len = client_identifier.len().min(self.client_identifier.len());
        self.client_identifier[..id_len].copy_from_slice(&client_identifier[..id_len]);
        self.client_identifier_len = id_len as u8;
        let hostname_len = requested_hostname.len().min(self.requested_hostname.len());
        self.requested_hostname[..hostname_len]
            .copy_from_slice(&requested_hostname[..hostname_len]);
        self.requested_hostname_len = hostname_len as u8;
        self
    }

    pub fn lease_identity(&self) -> &[u8] {
        if self.client_identifier_len == 0 {
            &self.hardware_address
        } else {
            &self.client_identifier[..self.client_identifier_len as usize]
        }
    }

    pub fn requested_hostname(&self) -> &[u8] {
        &self.requested_hostname[..self.requested_hostname_len as usize]
    }
}

#[derive(Clone, Copy)]
pub struct DhcpRequest {
    message_type: u8,
    client: DhcpClient,
}

impl DhcpRequest {
    pub const fn new(message_type: u8, client: DhcpClient) -> Self {
        Self {
            message_type,
            client,
        }
    }

    pub fn client(self) -> DhcpClient {
        self.client
    }
}

#[derive(Clone, Copy)]
struct DhcpLease {
    client: DhcpClient,
    ip: [u8; 4],
    expires_at_ms: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DhcpGrant {
    message_type: u8,
    lease_ip: [u8; 4],
}

impl DhcpGrant {
    pub const fn reply_message_type(self) -> u8 {
        self.message_type
    }

    pub const fn lease_ip(self) -> [u8; 4] {
        self.lease_ip
    }
}

pub struct DhcpLeaseState {
    active: [Option<DhcpLease>; DHCP_POOL_SIZE],
}

impl DhcpLeaseState {
    pub const fn new() -> Self {
        Self {
            active: [None; DHCP_POOL_SIZE],
        }
    }

    pub fn grant(&mut self, request: DhcpRequest, now_ms: u64) -> Option<DhcpGrant> {
        self.clear_expired(now_ms);

        match request.message_type {
            1 => self
                .reserve(request.client, now_ms, DHCP_OFFER_HOLD_SECONDS)
                .map(|lease_ip| DhcpGrant {
                    message_type: 2,
                    lease_ip,
                }),
            3 => self
                .reserve(request.client, now_ms, DHCP_LEASE_SECONDS)
                .map(|lease_ip| DhcpGrant {
                    message_type: 5,
                    lease_ip,
                }),
            7 => {
                self.release(request.client);
                None
            }
            _ => None,
        }
    }

    fn reserve(&mut self, client: DhcpClient, now_ms: u64, seconds: u32) -> Option<[u8; 4]> {
        let expires_at_ms = now_ms.saturating_add(seconds as u64 * 1_000);

        for lease in self.active.iter_mut().flatten() {
            if lease.client == client {
                lease.expires_at_ms = expires_at_ms;
                return Some(lease.ip);
            }
        }

        let preferred = network_registry::lease_identity_is_motherbrain(client.lease_identity())
            .then_some(0)
            .filter(|index| self.active[*index].is_none());
        let slot_index =
            preferred.or_else(|| self.active.iter().position(|lease| lease.is_none()))?;
        let ip = [
            AP_NETWORK[0],
            AP_NETWORK[1],
            AP_NETWORK[2],
            DHCP_POOL_FIRST_HOST + slot_index as u8,
        ];
        self.active[slot_index] = Some(DhcpLease {
            client,
            ip,
            expires_at_ms,
        });
        Some(ip)
    }

    fn release(&mut self, client: DhcpClient) {
        for slot in &mut self.active {
            if slot.map(|lease| lease.client == client).unwrap_or(false) {
                *slot = None;
                return;
            }
        }
    }

    fn clear_expired(&mut self, now_ms: u64) {
        for slot in &mut self.active {
            if slot
                .map(|lease| now_ms >= lease.expires_at_ms)
                .unwrap_or(false)
            {
                *slot = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(message_type: u8, client_id: u8) -> DhcpRequest {
        DhcpRequest::new(message_type, DhcpClient::new([0, 0, 0, 0, 0, client_id]))
    }

    #[test]
    fn assigns_a_distinct_address_to_each_client() {
        let mut leases = DhcpLeaseState::new();

        for client_id in 0..DHCP_POOL_SIZE as u8 {
            let grant = leases.grant(request(1, client_id), 0).unwrap();
            assert_eq!(grant.reply_message_type(), 2);
            assert_eq!(
                grant.lease_ip(),
                [192, 168, 4, DHCP_POOL_FIRST_HOST + client_id]
            );
        }

        assert!(leases.grant(request(1, DHCP_POOL_SIZE as u8), 0).is_none());
    }

    #[test]
    fn keeps_a_clients_address_for_request_and_renewal() {
        let mut leases = DhcpLeaseState::new();
        let offered = leases.grant(request(1, 7), 1_000).unwrap();
        let acknowledged = leases.grant(request(3, 7), 2_000).unwrap();

        assert_eq!(offered.lease_ip(), acknowledged.lease_ip());
        assert_eq!(acknowledged.reply_message_type(), 5);
        assert_eq!(acknowledged.lease_ip(), [192, 168, 4, 2]);
    }

    #[test]
    fn reuses_released_and_expired_addresses() {
        let mut leases = DhcpLeaseState::new();
        let first = leases.grant(request(1, 1), 0).unwrap();
        leases.grant(request(7, 1), 1_000);
        let after_release = leases.grant(request(1, 2), 1_000).unwrap();
        assert_eq!(after_release.lease_ip(), first.lease_ip());

        let after_expiry = leases
            .grant(
                request(1, 3),
                1_000 + DHCP_OFFER_HOLD_SECONDS as u64 * 1_000,
            )
            .unwrap();
        assert_eq!(after_expiry.lease_ip(), first.lease_ip());
    }

    #[test]
    fn reserved_names_are_not_ordinary_dhcp_names() {
        for name in [b"pete".as_slice(), b"MOTHERBRAIN", b"control"] {
            assert!(hostname_is_reserved(name));
        }
        assert!(!hostname_is_reserved(b"operator-laptop"));
    }
}
