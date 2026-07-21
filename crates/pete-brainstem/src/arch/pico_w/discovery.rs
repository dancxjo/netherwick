fn build_mdns_announcement(packet: &mut [u8; 768]) -> usize {
    packet.fill(0);
    packet[2..4].copy_from_slice(&[0x84, 0x00]);
    packet[6..8].copy_from_slice(&9u16.to_be_bytes());
    let mut i = 12;
    i = mdns_a(packet, i, MDNS_NAME, AP_IP_OCTETS).unwrap_or(12);
    for (service, port) in [
        (
            b"\x0f_pete-brainstem\x04_tcp\x05local\x00".as_slice(),
            HTTP_PORT,
        ),
        (
            b"\x0b_pete-debug\x04_tcp\x05local\x00".as_slice(),
            HTTP_PORT,
        ),
        (
            b"\x0d_pete-control\x04_tcp\x05local\x00".as_slice(),
            WS_CONTROL_PORT,
        ),
        (
            b"\x0d_pete-control\x04_udp\x05local\x00".as_slice(),
            UDP_CONTROL_PORT,
        ),
    ] {
        let mut instance = heapless::Vec::<u8, 64>::new();
        let _ = instance.extend_from_slice(b"\x09brainstem");
        let _ = instance.extend_from_slice(service);
        i = mdns_ptr(packet, i, service, &instance).unwrap_or(i);
        i = mdns_srv(packet, i, &instance, port, MDNS_NAME).unwrap_or(i);
    }
    i
}

fn mdns_a(packet: &mut [u8], mut i: usize, name: &[u8], ip: [u8; 4]) -> Option<usize> {
    i = put_bytes(packet, i, name)?;
    i = put_bytes(packet, i, &[0, 1, 0x80, 1, 0, 0, 0, 120, 0, 4])?;
    put_bytes(packet, i, &ip)
}
fn mdns_ptr(packet: &mut [u8], mut i: usize, name: &[u8], target: &[u8]) -> Option<usize> {
    i = put_bytes(packet, i, name)?;
    i = put_bytes(packet, i, &[0, 12, 0, 1, 0, 0, 0, 120])?;
    i = put_bytes(packet, i, &(target.len() as u16).to_be_bytes())?;
    put_bytes(packet, i, target)
}
fn mdns_srv(
    packet: &mut [u8],
    mut i: usize,
    name: &[u8],
    port: u16,
    target: &[u8],
) -> Option<usize> {
    i = put_bytes(packet, i, name)?;
    i = put_bytes(packet, i, &[0, 33, 0x80, 1, 0, 0, 0, 120])?;
    i = put_bytes(packet, i, &((6 + target.len()) as u16).to_be_bytes())?;
    i = put_bytes(packet, i, &[0, 0, 0, 0])?;
    i = put_bytes(packet, i, &port.to_be_bytes())?;
    put_bytes(packet, i, target)
}
fn put_bytes(packet: &mut [u8], offset: usize, bytes: &[u8]) -> Option<usize> {
    let end = offset.checked_add(bytes.len())?;
    packet.get_mut(offset..end)?.copy_from_slice(bytes);
    Some(end)
}

fn build_dns_reply<'a>(query: &[u8], response: &'a mut [u8; 512]) -> Option<&'a [u8]> {
    let question = parse_dns_question(query)?;
    let answer_ip = dns_answer_ip(
        &query[12..question.name_end],
        Instant::now().as_millis() as u32,
    )?;
    if !matches!(question.qtype, 1 | 255) || !matches!(question.qclass, 1 | 255) {
        return None;
    }

    response[..question.end].copy_from_slice(&query[..question.end]);
    response[2] = 0x84 | (query[2] & 0x01); // response, authoritative, preserve RD
    response[3] = 0x00; // no error
    response[4] = 0x00;
    response[5] = 0x01; // echo only the first question
    response[6] = 0x00;
    response[7] = 0x01; // one answer
    response[8] = 0x00;
    response[9] = 0x00;
    response[10] = 0x00;
    response[11] = 0x00;

    let mut i = question.end;
    let answer = [
        0xc0,
        0x0c, // compressed name pointer to the original question name
        0x00,
        0x01, // A
        0x00,
        0x01, // IN
        0x00,
        0x00,
        0x00,
        0x3c, // TTL 60s
        0x00,
        0x04, // IPv4 length
        answer_ip[0],
        answer_ip[1],
        answer_ip[2],
        answer_ip[3],
    ];
    if i + answer.len() > response.len() {
        return None;
    }
    response[i..i + answer.len()].copy_from_slice(&answer);
    i += answer.len();
    Some(&response[..i])
}

struct DnsQuestion {
    name_end: usize,
    end: usize,
    qtype: u16,
    qclass: u16,
}

fn parse_dns_question(packet: &[u8]) -> Option<DnsQuestion> {
    if packet.len() < 17 || packet[2] & 0x80 != 0 {
        return None;
    }
    let question_count = u16::from_be_bytes([packet[4], packet[5]]);
    if question_count == 0 {
        return None;
    }

    let mut i = 12;
    loop {
        let len = *packet.get(i)? as usize;
        if len & 0xc0 != 0 {
            return None;
        }
        i += 1;
        if len == 0 {
            break;
        }
        i = i.checked_add(len)?;
        if i > packet.len() {
            return None;
        }
    }

    let name_end = i;
    let end = i.checked_add(4)?;
    if end > packet.len() {
        return None;
    }
    Some(DnsQuestion {
        name_end,
        end,
        qtype: u16::from_be_bytes([packet[i], packet[i + 1]]),
        qclass: u16::from_be_bytes([packet[i + 2], packet[i + 3]]),
    })
}

fn dns_answer_ip(name: &[u8], now_ms: u32) -> Option<[u8; 4]> {
    const PETE_INTERNAL: &[u8] = b"\x04pete\x08internal\x00";
    const BRAINSTEM_INTERNAL: &[u8] = b"\x09brainstem\x04pete\x08internal\x00";
    const GATEWAY_INTERNAL: &[u8] = b"\x07gateway\x04pete\x08internal\x00";
    const MOTHERBRAIN_INTERNAL: &[u8] = b"\x0bmotherbrain\x04pete\x08internal\x00";
    if dns_name_eq(name, PETE_INTERNAL)
        || dns_name_eq(name, BRAINSTEM_INTERNAL)
        || dns_name_eq(name, GATEWAY_INTERNAL)
        || dns_name_eq(name, MDNS_NAME)
    {
        return Some(AP_IP_OCTETS);
    }
    if dns_name_eq(name, MOTHERBRAIN_INTERNAL) {
        return network_registry::resolve_motherbrain(now_ms);
    }
    None
}

fn dns_name_eq(left: &[u8], right: &[u8]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| dns_byte_eq(*left, *right))
}

fn dns_byte_eq(left: u8, right: u8) -> bool {
    if left.is_ascii_alphabetic() && right.is_ascii_alphabetic() {
        left.to_ascii_lowercase() == right.to_ascii_lowercase()
    } else {
        left == right
    }
}

fn build_dhcp_reply<'a>(
    grant: DhcpGrant,
    request: &[u8],
    response: &'a mut [u8; 576],
) -> Option<&'a [u8]> {
    response.fill(0);
    response[0] = 2;
    response[1] = request[1];
    response[2] = request[2];
    response[3] = request[3];
    response[4..8].copy_from_slice(&request[4..8]);
    response[10..12].copy_from_slice(&request[10..12]);
    response[16..20].copy_from_slice(&grant.lease_ip());
    response[20..24].copy_from_slice(&AP_IP_OCTETS);
    response[28..44].copy_from_slice(&request[28..44]);
    response[236..240].copy_from_slice(&[99, 130, 83, 99]);

    let mut i = 240;
    i = write_dhcp_option(i, response, 53, &[grant.reply_message_type()])?;
    i = write_dhcp_option(i, response, 54, &AP_IP_OCTETS)?;
    i = write_dhcp_option(i, response, 51, &DHCP_LEASE_SECONDS.to_be_bytes())?;
    i = write_dhcp_option(i, response, 1, &[255, 255, 255, 0])?;
    i = write_dhcp_option(i, response, 3, &AP_IP_OCTETS)?;
    i = write_dhcp_option(i, response, 6, &AP_IP_OCTETS)?;
    response[i] = 255;
    Some(&response[..i + 1])
}

impl DhcpRequest {
    fn parse(packet: &[u8]) -> Option<Self> {
        if packet.len() < 240 || packet[0] != 1 || packet[1] != 1 || packet[2] < 6 {
            return None;
        }

        let mut hardware_address = [0; 6];
        hardware_address.copy_from_slice(&packet[28..34]);

        let client_identifier = dhcp_option(packet, 61).unwrap_or(&[]);
        let requested_hostname = dhcp_option(packet, 12).unwrap_or(&[]);
        Some(Self::new(
            dhcp_message_type(packet)?,
            DhcpClient::new(hardware_address).with_metadata(client_identifier, requested_hostname),
        ))
    }
}

fn dhcp_message_type(packet: &[u8]) -> Option<u8> {
    dhcp_option(packet, 53).and_then(|value| (value.len() == 1).then_some(value[0]))
}

fn dhcp_option(packet: &[u8], wanted: u8) -> Option<&[u8]> {
    if packet.len() < 240 || packet[236..240] != [99, 130, 83, 99] {
        return None;
    }

    let mut i = 240;
    while i < packet.len() {
        let option = packet[i];
        i += 1;
        match option {
            0 => continue,
            255 => return None,
            _ => {
                if i >= packet.len() {
                    return None;
                }
                let len = packet[i] as usize;
                i += 1;
                if i + len > packet.len() {
                    return None;
                }
                if option == wanted {
                    return Some(&packet[i..i + len]);
                }
                i += len;
            }
        }
    }
    None
}

fn write_dhcp_option(
    offset: usize,
    packet: &mut [u8; 576],
    option: u8,
    value: &[u8],
) -> Option<usize> {
    let end = offset.checked_add(2)?.checked_add(value.len())?;
    if end >= packet.len() || value.len() > u8::MAX as usize {
        return None;
    }
    packet[offset] = option;
    packet[offset + 1] = value.len() as u8;
    packet[offset + 2..end].copy_from_slice(value);
    Some(end)
}

fn level(high: bool) -> Level {
    if high {
        Level::High
    } else {
        Level::Low
    }
}
