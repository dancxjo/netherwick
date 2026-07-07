use core::fmt::{self, Write as _};

use heapless::String;

use crate::body;

pub struct BrainstemCapabilities {
    pub firmware_name: &'static str,
    pub firmware_version: &'static str,
    pub body_name: &'static str,
    pub body_kind: &'static str,
    pub drive: &'static str,
    pub verbs: &'static [&'static str],
    pub sensors: &'static [&'static str],
    pub outputs: &'static [&'static str],
    pub safety: &'static [&'static str],
    pub feedback: &'static [&'static str],
    pub events: &'static [&'static str],
    pub sensor_packets: &'static str,
    pub max_song_tones: usize,
    pub song_slots: u8,
    pub max_linear_mm_s: i16,
    pub max_angular_mrad_s: i16,
    pub min_ttl_ms: u32,
    pub max_ttl_ms: u32,
    #[allow(dead_code)]
    pub driver: BodyDriverCapabilities,
}

#[allow(dead_code)]
pub struct BodyDriverCapabilities {
    pub modes: &'static [&'static str],
    pub sensor_packets: &'static str,
    pub has_brc: bool,
    pub has_power_toggle: bool,
    pub has_lights: bool,
    pub has_songs: bool,
    pub has_dock: bool,
}

pub fn current() -> BrainstemCapabilities {
    body::current_capabilities()
}

pub fn render_json<'a>(
    capabilities: &BrainstemCapabilities,
    command_id: u32,
    buffer: &'a mut [u8],
) -> Option<&'a str> {
    let mut response = String::<3072>::new();
    write!(
        response,
        "{{\"accepted\":true,\"command_id\":{},\"firmware\":\"{}\",\"version\":\"{}\",\"body\":\"{}\",\"body_kind\":\"{}\",\"drive\":\"{}\",",
        command_id,
        capabilities.firmware_name,
        capabilities.firmware_version,
        capabilities.body_name,
        capabilities.body_kind,
        capabilities.drive
    )
    .ok()?;
    write_json_str_array(&mut response, "verbs", capabilities.verbs).ok()?;
    write_json_str_array(&mut response, "sensors", capabilities.sensors).ok()?;
    write_json_str_array(&mut response, "outputs", capabilities.outputs).ok()?;
    write_json_str_array(&mut response, "safety", capabilities.safety).ok()?;
    write_json_str_array(&mut response, "feedback", capabilities.feedback).ok()?;
    write_json_str_array(&mut response, "events", capabilities.events).ok()?;
    write!(
        response,
        "\"limits\":{{\"max_linear_mm_s\":{},\"max_angular_mrad_s\":{},\"min_ttl_ms\":{},\"max_ttl_ms\":{}}},\"sensor_packets\":\"{}\",\"max_song_tones\":{},\"song_slots\":{}}}\n",
        capabilities.max_linear_mm_s,
        capabilities.max_angular_mrad_s,
        capabilities.min_ttl_ms,
        capabilities.max_ttl_ms,
        capabilities.sensor_packets,
        capabilities.max_song_tones,
        capabilities.song_slots
    )
    .ok()?;

    let bytes = response.as_bytes();
    if bytes.len() > buffer.len() {
        return None;
    }
    buffer[..bytes.len()].copy_from_slice(bytes);
    core::str::from_utf8(&buffer[..bytes.len()]).ok()
}

pub fn write_compact<const N: usize>(
    response: &mut String<N>,
    capabilities: &BrainstemCapabilities,
    seq: u32,
) -> fmt::Result {
    write!(
        response,
        "OK {seq} CAPABILITIES body_kind={} drive={} verbs=",
        capabilities.body_kind, capabilities.drive
    )?;
    write_csv(response, capabilities.verbs)?;
    write!(response, " sensors=")?;
    write_csv(response, capabilities.sensors)?;
    write!(response, " outputs=")?;
    write_csv(response, capabilities.outputs)?;
    write!(response, " safety=")?;
    write_csv(response, capabilities.safety)?;
    write!(response, " events=")?;
    write_csv(response, capabilities.events)?;
    write!(
        response,
        " limits=max_linear_mm_s:{},max_angular_mrad_s:{},min_ttl_ms:{},max_ttl_ms:{} max_tones={} song_slots={} feedback_slots={} sensor_packets={}\n",
        capabilities.max_linear_mm_s,
        capabilities.max_angular_mrad_s,
        capabilities.min_ttl_ms,
        capabilities.max_ttl_ms,
        capabilities.max_song_tones,
        capabilities.song_slots,
        capabilities.feedback.len(),
        capabilities.sensor_packets
    )
}

fn write_json_str_array<const N: usize>(
    response: &mut String<N>,
    key: &str,
    values: &[&str],
) -> fmt::Result {
    write!(response, "\"{}\":[", key)?;
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            response.push(',').map_err(|_| fmt::Error)?;
        }
        write!(response, "\"{}\"", value)?;
    }
    response.push_str("],").map_err(|_| fmt::Error)
}

fn write_csv<const N: usize>(response: &mut String<N>, values: &[&str]) -> fmt::Result {
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            response.push(',').map_err(|_| fmt::Error)?;
        }
        response.push_str(value).map_err(|_| fmt::Error)?;
    }
    Ok(())
}
