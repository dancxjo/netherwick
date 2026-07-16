const AP_IP_OCTETS: [u8; 4] = [192, 168, 4, 1];
const ICMP_ETHERNET_HEADER_LEN: usize = 14;
const ICMP_IPV4_HEADER_LEN: usize = 20;
const ICMP_HEADER_LEN: usize = 8;
pub(crate) const NETWORK_FRAME_CAPACITY: usize = 1_536;
const ICMP_MAX_FRAME_LEN: usize = 1_280;
const ICMP_ECHO_REPLY_LIMIT_PER_SECOND: u8 = 4;

#[derive(Clone, Copy)]
pub(crate) struct IcmpRateLimit {
    window_started_ms: u64,
    replies_in_window: u8,
}

impl IcmpRateLimit {
    pub(crate) const fn new() -> Self {
        Self {
            window_started_ms: 0,
            replies_in_window: 0,
        }
    }

    fn allow(&mut self, now_ms: u64) -> bool {
        if now_ms.wrapping_sub(self.window_started_ms) >= 1_000 {
            self.window_started_ms = now_ms;
            self.replies_in_window = 0;
        }
        if self.replies_in_window >= ICMP_ECHO_REPLY_LIMIT_PER_SECOND {
            return false;
        }
        self.replies_in_window += 1;
        true
    }
}

pub(crate) enum IcmpEchoDisposition {
    NotIcmp,
    Reply(usize),
    RateLimited,
    Dropped,
}

pub(crate) fn process_icmp_echo_frame(
    frame: &mut [u8],
    now_ms: u64,
    rate_limit: &mut IcmpRateLimit,
) -> IcmpEchoDisposition {
    if frame.len() < ICMP_ETHERNET_HEADER_LEN || frame[12..14] != [0x08, 0x00] {
        return IcmpEchoDisposition::NotIcmp;
    }
    let ip = &frame[ICMP_ETHERNET_HEADER_LEN..];
    if ip.len() < ICMP_IPV4_HEADER_LEN || ip[0] >> 4 != 4 || ip[9] != 1 {
        return IcmpEchoDisposition::NotIcmp;
    }
    if frame.len() > ICMP_MAX_FRAME_LEN {
        return IcmpEchoDisposition::Dropped;
    }
    let header_len = (ip[0] as usize & 0x0f) * 4;
    if header_len < ICMP_IPV4_HEADER_LEN || header_len > ip.len() {
        return IcmpEchoDisposition::Dropped;
    }
    let total_len = u16::from_be_bytes([ip[2], ip[3]]) as usize;
    if total_len < header_len + ICMP_HEADER_LEN
        || total_len > ip.len()
        || ipv4_checksum(&ip[..header_len]) != 0
    {
        return IcmpEchoDisposition::Dropped;
    }
    if ip[16..20] != AP_IP_OCTETS || ip[12..16] == [0, 0, 0, 0] || ip[12] >= 224 {
        return IcmpEchoDisposition::Dropped;
    }
    if u16::from_be_bytes([ip[6], ip[7]]) & 0x3fff != 0 {
        return IcmpEchoDisposition::Dropped;
    }
    let icmp_end = ICMP_ETHERNET_HEADER_LEN + total_len;
    let icmp_start = ICMP_ETHERNET_HEADER_LEN + header_len;
    if ipv4_checksum(&frame[icmp_start..icmp_end]) != 0
        || frame[icmp_start] != 8
        || frame[icmp_start + 1] != 0
    {
        return IcmpEchoDisposition::Dropped;
    }
    if !rate_limit.allow(now_ms) {
        return IcmpEchoDisposition::RateLimited;
    }

    let icmp_len = total_len - header_len;
    let source_mac = [frame[0], frame[1], frame[2], frame[3], frame[4], frame[5]];
    let destination_mac = [frame[6], frame[7], frame[8], frame[9], frame[10], frame[11]];
    let source_ip = [frame[26], frame[27], frame[28], frame[29]];
    frame[0..6].copy_from_slice(&destination_mac);
    frame[6..12].copy_from_slice(&source_mac);
    let reply_ip =
        &mut frame[ICMP_ETHERNET_HEADER_LEN..ICMP_ETHERNET_HEADER_LEN + ICMP_IPV4_HEADER_LEN];
    reply_ip.fill(0);
    reply_ip[0] = 0x45;
    reply_ip[2..4].copy_from_slice(&((ICMP_IPV4_HEADER_LEN + icmp_len) as u16).to_be_bytes());
    reply_ip[8] = 64;
    reply_ip[9] = 1;
    reply_ip[12..16].copy_from_slice(&AP_IP_OCTETS);
    reply_ip[16..20].copy_from_slice(&source_ip);
    let header_checksum = ipv4_checksum(reply_ip);
    reply_ip[10..12].copy_from_slice(&header_checksum.to_be_bytes());
    let reply_icmp_start = ICMP_ETHERNET_HEADER_LEN + ICMP_IPV4_HEADER_LEN;
    frame.copy_within(icmp_start + 4..icmp_end, reply_icmp_start + 4);
    frame[reply_icmp_start] = 0;
    frame[reply_icmp_start + 1] = 0;
    frame[reply_icmp_start + 2..reply_icmp_start + 4].fill(0);
    let reply_end = reply_icmp_start + icmp_len;
    let checksum = ipv4_checksum(&frame[reply_icmp_start..reply_end]);
    frame[reply_icmp_start + 2..reply_icmp_start + 4].copy_from_slice(&checksum.to_be_bytes());
    IcmpEchoDisposition::Reply(ICMP_ETHERNET_HEADER_LEN + ICMP_IPV4_HEADER_LEN + icmp_len)
}

fn ipv4_checksum(bytes: &[u8]) -> u16 {
    let mut sum = 0u32;
    let mut chunks = bytes.chunks_exact(2);
    for chunk in &mut chunks {
        sum += u16::from_be_bytes([chunk[0], chunk[1]]) as u32;
    }
    if let Some(&byte) = chunks.remainder().first() {
        sum += (byte as u32) << 8;
    }
    while sum >> 16 != 0 {
        sum = (sum & 0xffff) + (sum >> 16);
    }
    !(sum as u16)
}

#[cfg(test)]
mod icmp_tests {
    use super::*;

    fn echo_request(
        payload: &[u8],
        destination: [u8; 4],
    ) -> heapless::Vec<u8, NETWORK_FRAME_CAPACITY> {
        let mut frame = heapless::Vec::<u8, NETWORK_FRAME_CAPACITY>::new();
        let total_len = ICMP_IPV4_HEADER_LEN + ICMP_HEADER_LEN + payload.len();
        frame
            .resize_default(ICMP_ETHERNET_HEADER_LEN + total_len)
            .unwrap();
        frame[0..6].copy_from_slice(&[0x02, 0, 0, 0, 0, 1]);
        frame[6..12].copy_from_slice(&[0x02, 0, 0, 0, 0, 2]);
        frame[12..14].copy_from_slice(&[0x08, 0x00]);
        let ip =
            &mut frame[ICMP_ETHERNET_HEADER_LEN..ICMP_ETHERNET_HEADER_LEN + ICMP_IPV4_HEADER_LEN];
        ip[0] = 0x45;
        ip[2..4].copy_from_slice(&(total_len as u16).to_be_bytes());
        ip[4..6].copy_from_slice(&0x0123u16.to_be_bytes());
        ip[8] = 64;
        ip[9] = 1;
        ip[12..16].copy_from_slice(&[192, 168, 4, 2]);
        ip[16..20].copy_from_slice(&destination);
        let checksum = ipv4_checksum(ip);
        ip[10..12].copy_from_slice(&checksum.to_be_bytes());
        let icmp_start = ICMP_ETHERNET_HEADER_LEN + ICMP_IPV4_HEADER_LEN;
        frame[icmp_start] = 8;
        frame[icmp_start + 1] = 0;
        frame[icmp_start + 4..icmp_start + 6].copy_from_slice(&0x1234u16.to_be_bytes());
        frame[icmp_start + 6..icmp_start + 8].copy_from_slice(&0x5678u16.to_be_bytes());
        frame[icmp_start + ICMP_HEADER_LEN..].copy_from_slice(payload);
        let checksum = ipv4_checksum(&frame[icmp_start..]);
        frame[icmp_start + 2..icmp_start + 4].copy_from_slice(&checksum.to_be_bytes());
        frame
    }

    #[test]
    fn echo_reply_preserves_identifier_sequence_and_payload_sizes() {
        for payload in [
            b"".as_slice(),
            b"platform-default-ping-payload-0123456789abcdefghijklmnopqrstuv".as_slice(),
            &[0xa5; 512][..],
        ] {
            let mut frame = echo_request(payload, AP_IP_OCTETS);
            let mut rate_limit = IcmpRateLimit::new();
            let reply_len = match process_icmp_echo_frame(&mut frame, 1_000, &mut rate_limit) {
                IcmpEchoDisposition::Reply(len) => len,
                _ => panic!("expected echo reply"),
            };
            let icmp_start = ICMP_ETHERNET_HEADER_LEN + ICMP_IPV4_HEADER_LEN;
            assert_eq!(
                reply_len,
                ICMP_ETHERNET_HEADER_LEN + ICMP_IPV4_HEADER_LEN + ICMP_HEADER_LEN + payload.len()
            );
            assert_eq!(&frame[0..6], &[0x02, 0, 0, 0, 0, 2]);
            assert_eq!(&frame[6..12], &[0x02, 0, 0, 0, 0, 1]);
            assert_eq!(&frame[26..30], &AP_IP_OCTETS);
            assert_eq!(&frame[30..34], &[192, 168, 4, 2]);
            assert_eq!(
                ipv4_checksum(&frame[ICMP_ETHERNET_HEADER_LEN..icmp_start]),
                0
            );
            assert_eq!(frame[icmp_start], 0);
            assert_eq!(
                &frame[icmp_start + 4..icmp_start + 8],
                &0x1234_5678u32.to_be_bytes()
            );
            assert_eq!(&frame[icmp_start + ICMP_HEADER_LEN..reply_len], payload);
            assert_eq!(ipv4_checksum(&frame[icmp_start..reply_len]), 0);
        }
    }

    #[test]
    fn malformed_length_checksum_and_unowned_destination_do_not_reply() {
        let mut rate_limit = IcmpRateLimit::new();
        let mut truncated = echo_request(b"payload", AP_IP_OCTETS);
        truncated[16..18].copy_from_slice(&0xffffu16.to_be_bytes());
        assert!(matches!(
            process_icmp_echo_frame(&mut truncated, 1_000, &mut rate_limit),
            IcmpEchoDisposition::Dropped
        ));

        let mut bad_checksum = echo_request(b"payload", AP_IP_OCTETS);
        bad_checksum[ICMP_ETHERNET_HEADER_LEN + ICMP_IPV4_HEADER_LEN + 2] ^= 1;
        assert!(matches!(
            process_icmp_echo_frame(&mut bad_checksum, 1_000, &mut rate_limit),
            IcmpEchoDisposition::Dropped
        ));

        let mut unowned = echo_request(b"payload", [192, 168, 4, 99]);
        assert!(matches!(
            process_icmp_echo_frame(&mut unowned, 1_000, &mut rate_limit),
            IcmpEchoDisposition::Dropped
        ));
    }

    #[test]
    fn echo_replies_are_rate_limited_per_second() {
        let mut rate_limit = IcmpRateLimit::new();
        for _ in 0..ICMP_ECHO_REPLY_LIMIT_PER_SECOND {
            let mut frame = echo_request(b"payload", AP_IP_OCTETS);
            assert!(matches!(
                process_icmp_echo_frame(&mut frame, 1_000, &mut rate_limit),
                IcmpEchoDisposition::Reply(_)
            ));
        }
        let mut limited = echo_request(b"payload", AP_IP_OCTETS);
        assert!(matches!(
            process_icmp_echo_frame(&mut limited, 1_000, &mut rate_limit),
            IcmpEchoDisposition::RateLimited
        ));
        let mut next_window = echo_request(b"payload", AP_IP_OCTETS);
        assert!(matches!(
            process_icmp_echo_frame(&mut next_window, 2_000, &mut rate_limit),
            IcmpEchoDisposition::Reply(_)
        ));
    }
}
