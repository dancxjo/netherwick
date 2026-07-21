pub fn signal_event(event: &BrainstemEvent) {
    record_public_event_from_brainstem_event(event);
    let blinks = match event {
        BrainstemEvent::Boot => 1,
        BrainstemEvent::CreatePowerOnRequested | BrainstemEvent::CreatePowerOffRequested => 2,
        BrainstemEvent::CreatePowerToggled => 3,
        BrainstemEvent::CreateOiStartRequested | BrainstemEvent::CreateOiModeRequested(_) => 5,
        BrainstemEvent::CreatePacketReceived { .. }
        | BrainstemEvent::CreateSensorPacketDecoded { .. } => 6,
        BrainstemEvent::DriveRequested { .. } | BrainstemEvent::DriveStopped => 7,
        BrainstemEvent::Error(_) => 8,
        BrainstemEvent::TickMs(_) => return,
    };
    request_led_blinks(blinks);
}

pub fn event_next_seq() -> u32 {
    EVENT_NEXT_SEQ.load(Ordering::Relaxed)
}

pub fn event_oldest_seq() -> u32 {
    event_next_seq()
        .saturating_sub(EVENT_LOG_CAPACITY as u32)
        .max(1)
}

pub fn event_dropped_before_seq(since_seq: u32) -> u32 {
    let oldest_seq = event_oldest_seq();
    if since_seq.saturating_add(1) < oldest_seq {
        oldest_seq
    } else {
        0
    }
}

pub fn collect_events_since<const N: usize>(
    since_seq: u32,
    out: &mut heapless::Vec<PublicEventRecord, N>,
) {
    let next_seq = EVENT_NEXT_SEQ.load(Ordering::Relaxed);
    let since_seq = since_seq.max(event_oldest_seq().saturating_sub(1));
    for seq in since_seq.saturating_add(1)..next_seq {
        let index = event_index(seq);
        if EVENT_SEQ[index].load(Ordering::Relaxed) != seq {
            continue;
        }
        let _ = out.push(PublicEventRecord {
            seq,
            kind: EVENT_KIND[index].load(Ordering::Relaxed),
            a: EVENT_A[index].load(Ordering::Relaxed),
            b: EVENT_B[index].load(Ordering::Relaxed),
            c: EVENT_C[index].load(Ordering::Relaxed),
        });
    }
}

fn event_batch_next_seq<const N: usize>(records: &heapless::Vec<PublicEventRecord, N>) -> u32 {
    records
        .last()
        .map(|record| record.seq.saturating_add(1))
        .unwrap_or_else(event_next_seq)
}

#[cfg(feature = "pico-w")]
pub fn render_events_json<'a>(since_seq: u32, buffer: &'a mut [u8]) -> Option<&'a str> {
    let mut response = heapless::String::<2048>::new();
    let mut records = heapless::Vec::<PublicEventRecord, EVENT_RESPONSE_CAPACITY>::new();
    collect_events_since(since_seq, &mut records);
    let batch_next_seq = event_batch_next_seq(&records);
    write!(
        response,
        "{{\"type\":\"events\",\"since_seq\":{},\"oldest_seq\":{},\"next_seq\":{},\"dropped_before_seq\":{},\"events\":[",
        since_seq,
        event_oldest_seq(),
        batch_next_seq,
        event_dropped_before_seq(since_seq)
    )
    .ok()?;
    for (index, record) in records.iter().enumerate() {
        if index > 0 {
            response.push(',').ok()?;
        }
        write!(
            response,
            "{{\"seq\":{},\"kind\":\"{}\",\"a\":{},\"b\":{},\"c\":{}}}",
            record.seq,
            public_event_kind_text(record.kind),
            record.a,
            record.b,
            record.c
        )
        .ok()?;
    }
    response.push_str("]}\n").ok()?;
    let bytes = response.as_bytes();
    if bytes.len() > buffer.len() {
        return None;
    }
    buffer[..bytes.len()].copy_from_slice(bytes);
    core::str::from_utf8(&buffer[..bytes.len()]).ok()
}

pub fn write_compact_events<const N: usize>(
    response: &mut heapless::String<N>,
    since_seq: u32,
) -> core::fmt::Result {
    let mut records = heapless::Vec::<PublicEventRecord, EVENT_RESPONSE_CAPACITY>::new();
    collect_events_since(since_seq, &mut records);
    let batch_next_seq = event_batch_next_seq(&records);
    write!(
        response,
        "EVENTS since={} oldest={} next={} dropped_before={} count={}",
        since_seq,
        event_oldest_seq(),
        batch_next_seq,
        event_dropped_before_seq(since_seq),
        records.len()
    )?;
    for record in records {
        write!(
            response,
            " | {}:{}:{},{},{}",
            record.seq,
            public_event_kind_text(record.kind),
            record.a,
            record.b,
            record.c
        )?;
    }
    response.push('\n').map_err(|_| core::fmt::Error)
}

#[cfg(feature = "pico-w")]
pub fn take_led_blinks() -> Option<u8> {
    let blinks = PENDING_LED_BLINKS.load(Ordering::Relaxed);
    PENDING_LED_BLINKS.store(0, Ordering::Relaxed);
    match blinks {
        0 => None,
        blinks => Some(blinks),
    }
}

fn request_led_blinks(blinks: u8) {
    let blinks = blinks.min(9);
    if blinks > PENDING_LED_BLINKS.load(Ordering::Relaxed) {
        PENDING_LED_BLINKS.store(blinks, Ordering::Relaxed);
    }
}

fn increment(counter: &AtomicU32) {
    increment_by(counter, 1);
}

fn increment_by(counter: &AtomicU32, amount: u32) {
    counter.store(
        counter.load(Ordering::Relaxed).saturating_add(amount),
        Ordering::Relaxed,
    );
}

fn add_signed(counter: &AtomicU32, amount: i32) {
    let current = decode_signed_i32(counter.load(Ordering::Relaxed));
    counter.store(
        encode_signed_i32(current.saturating_add(amount)),
        Ordering::Relaxed,
    );
}

fn record_public_event(kind: PublicEventKind, a: u32, b: u32, c: u32) -> u32 {
    let seq = EVENT_NEXT_SEQ.load(Ordering::Relaxed);
    EVENT_NEXT_SEQ.store(seq.wrapping_add(1).max(1), Ordering::Relaxed);
    let index = event_index(seq);
    EVENT_A[index].store(a, Ordering::Relaxed);
    EVENT_B[index].store(b, Ordering::Relaxed);
    EVENT_C[index].store(c, Ordering::Relaxed);
    EVENT_KIND[index].store(kind as u8, Ordering::Relaxed);
    EVENT_SEQ[index].store(seq, Ordering::Relaxed);
    seq
}

fn record_public_event_from_brainstem_event(event: &BrainstemEvent) {
    match event {
        BrainstemEvent::Boot => record_public_event(PublicEventKind::Boot, 0, 0, 0),
        BrainstemEvent::CreatePowerOnRequested => {
            record_public_event(PublicEventKind::BodyPowerRequested, 1, 0, 0)
        }
        BrainstemEvent::CreatePowerOffRequested => {
            record_public_event(PublicEventKind::BodyPowerRequested, 0, 0, 0)
        }
        BrainstemEvent::CreatePowerToggled => {
            record_public_event(PublicEventKind::BodyPowerChanged, 0, 0, 0)
        }
        BrainstemEvent::CreateOiStartRequested => {
            record_public_event(PublicEventKind::BodyModeRequested, 0, 0, 0)
        }
        BrainstemEvent::CreateOiModeRequested(mode) => record_public_event(
            PublicEventKind::BodyModeRequested,
            encode_oi_mode_public(*mode),
            0,
            0,
        ),
        BrainstemEvent::CreatePacketReceived { .. }
        | BrainstemEvent::CreateSensorPacketDecoded { .. } => 0,
        BrainstemEvent::DriveRequested {
            left_mm_s,
            right_mm_s,
            duration_ms,
        } => record_public_event(
            PublicEventKind::MotionRequested,
            pack_i16_pair(*left_mm_s, *right_mm_s),
            *duration_ms,
            0,
        ),
        BrainstemEvent::DriveStopped => {
            record_public_event(PublicEventKind::MotionStopped, 0, 0, 0)
        }
        BrainstemEvent::Error(error) => {
            record_public_event(PublicEventKind::Error, encode_error_public(*error), 0, 0)
        }
        BrainstemEvent::TickMs(_) => 0,
    };
}

const fn event_index(seq: u32) -> usize {
    seq as usize % EVENT_LOG_CAPACITY
}

#[cfg(test)]
pub(crate) fn reset_event_log_for_test() {
    EVENT_NEXT_SEQ.store(1, Ordering::Relaxed);
    SAFETY_HAZARD_GENERATION.store(0, Ordering::Relaxed);
    clear_velocity_stream();
    PENDING_VELOCITY_IS_RENEWAL.store(OFF, Ordering::Relaxed);
    for index in 0..EVENT_LOG_CAPACITY {
        EVENT_SEQ[index].store(0, Ordering::Relaxed);
        EVENT_KIND[index].store(PublicEventKind::None as u8, Ordering::Relaxed);
        EVENT_A[index].store(0, Ordering::Relaxed);
        EVENT_B[index].store(0, Ordering::Relaxed);
        EVENT_C[index].store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
static STATUS_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub(crate) fn status_test_guard() -> std::sync::MutexGuard<'static, ()> {
    STATUS_TEST_LOCK.lock().unwrap()
}

fn create_sensor_flags_bits(sensors: CreateSensorPacket) -> u32 {
    let flags = sensors.flags;
    (flags.bump_left as u32)
        | ((flags.bump_right as u32) << 1)
        | ((flags.wheel_drop as u32) << 2)
        | ((flags.wall as u32) << 3)
        | ((flags.cliff_left as u32) << 4)
        | ((flags.cliff_front_left as u32) << 5)
        | ((flags.cliff_front_right as u32) << 6)
        | ((flags.cliff_right as u32) << 7)
        | ((flags.virtual_wall as u32) << 8)
        | ((flags.overcurrent as u32) << 9)
}

fn merge_create_sensor_flags(packet_id: u8, old_flags: u32, packet_flags: u32) -> u32 {
    let mask = match packet_id {
        0 => 0b11_1111_1111,
        7 => (1 << 0) | (1 << 1) | (1 << 2),
        8 => 1 << 3,
        9 => 1 << 4,
        10 => 1 << 5,
        11 => 1 << 6,
        12 => 1 << 7,
        13 => 1 << 8,
        14 => 1 << 9,
        _ => 0,
    };
    (old_flags & !mask) | (packet_flags & mask)
}

#[allow(clippy::too_many_arguments)]
fn record_sensor_edge_events(
    packet_id: u8,
    old_flags: u32,
    new_flags: u32,
    old_ir_byte: u8,
    new_ir_byte: u8,
    old_buttons: u8,
    new_buttons: u8,
    old_charging_state: u8,
    new_charging_state: u8,
    old_charge: u16,
    old_capacity: u16,
    new_charge: u16,
    new_capacity: u16,
) {
    if changed(old_flags, new_flags, 1 << 3) {
        mark_wall_changed(new_flags & (1 << 3) != 0);
    }
    if changed(old_flags, new_flags, 1 << 8) {
        mark_virtual_wall_changed(new_flags & (1 << 8) != 0);
    }
    if create_packet_has_charging_state(packet_id) && old_charging_state != new_charging_state {
        mark_charging_state_changed(new_charging_state);
    }
    if create_packet_has_buttons(packet_id) && old_buttons != new_buttons {
        mark_buttons_changed(new_buttons);
    }
    if create_packet_has_ir(packet_id) && old_ir_byte != new_ir_byte {
        mark_ir_changed(new_ir_byte);
    }

    let old_percent = battery_percent(old_charge, old_capacity);
    let new_percent = battery_percent(new_charge, new_capacity);
    if let Some(percent) = new_percent {
        let old_low = old_percent.is_some_and(|value| value <= 20);
        let new_low = percent <= 20;
        let latched = BATTERY_LOW_LATCHED.load(Ordering::Relaxed) != 0;
        if new_low && (!old_low || !latched) {
            mark_battery_low(percent);
            BATTERY_LOW_LATCHED.store(1, Ordering::Relaxed);
        } else if !new_low {
            BATTERY_LOW_LATCHED.store(0, Ordering::Relaxed);
        }
    }
}

fn changed(old_flags: u32, new_flags: u32, mask: u32) -> bool {
    old_flags & mask != new_flags & mask
}

fn battery_percent(charge_mah: u16, capacity_mah: u16) -> Option<u8> {
    if capacity_mah == 0 {
        None
    } else {
        Some(((charge_mah as u32 * 100) / capacity_mah as u32).min(100) as u8)
    }
}

fn create_packet_has_distance_delta(packet_id: u8) -> bool {
    matches!(packet_id, 0 | 19)
}

fn create_packet_has_angle_delta(packet_id: u8) -> bool {
    matches!(packet_id, 0 | 20)
}

fn create_packet_has_ir(packet_id: u8) -> bool {
    matches!(packet_id, 0 | 17)
}

fn create_packet_has_buttons(packet_id: u8) -> bool {
    matches!(packet_id, 0 | 18)
}

fn create_packet_has_charging_state(packet_id: u8) -> bool {
    matches!(packet_id, 0 | 21)
}

fn create_packet_has_charging_sources(packet_id: u8) -> bool {
    packet_id == 34
}

fn create_packet_has_voltage(packet_id: u8) -> bool {
    matches!(packet_id, 0 | 22)
}

fn create_packet_has_current(packet_id: u8) -> bool {
    matches!(packet_id, 0 | 23)
}

fn create_packet_has_temperature(packet_id: u8) -> bool {
    matches!(packet_id, 0 | 24)
}

fn create_packet_has_charge(packet_id: u8) -> bool {
    matches!(packet_id, 0 | 25)
}

fn create_packet_has_capacity(packet_id: u8) -> bool {
    matches!(packet_id, 0 | 26)
}

fn create_packet_has_cliff_left_signal(packet_id: u8) -> bool {
    packet_id == 28
}

fn create_packet_has_cliff_front_left_signal(packet_id: u8) -> bool {
    packet_id == 29
}

fn create_packet_has_cliff_front_right_signal(packet_id: u8) -> bool {
    packet_id == 30
}

fn create_packet_has_cliff_right_signal(packet_id: u8) -> bool {
    packet_id == 31
}

fn encode_signed_i16(value: i16) -> u32 {
    value as u16 as u32
}

fn decode_signed_i16(value: u32) -> i16 {
    value as u16 as i16
}

fn encode_signed_i8(value: i8) -> u32 {
    value as u8 as u32
}

fn decode_signed_i8(value: u32) -> i8 {
    value as u8 as i8
}

fn encode_signed_i32(value: i32) -> u32 {
    value as u32
}

fn decode_signed_i32(value: u32) -> i32 {
    value as i32
}

fn pack_i16_pair(left: i16, right: i16) -> u32 {
    ((left as u16 as u32) << 16) | right as u16 as u32
}

fn encode_oi_mode_public(mode: CreateOiMode) -> u32 {
    match mode {
        CreateOiMode::Passive => 1,
        CreateOiMode::Safe => 2,
        CreateOiMode::Full => 3,
    }
}

fn encode_error_public(error: BrainstemError) -> u32 {
    match error {
        BrainstemError::CreateNoResponse => ErrorCode::CreateNoResponse as u32,
        BrainstemError::UartFraming => ErrorCode::UartFraming as u32,
        BrainstemError::Timeout => ErrorCode::Timeout as u32,
        BrainstemError::InvalidPacket => ErrorCode::InvalidPacket as u32,
    }
}
