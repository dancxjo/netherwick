fn run_mouth(args: MouthArgs) -> Result<()> {
    let Some(mouth) = QueuedPiperCpalMouth::from_env()? else {
        anyhow::bail!(
            "robot mouth disabled: no Piper voice found; set PETE_TTS_PIPER_VOICE and PETE_TTS_PIPER_CONFIG"
        );
    };
    let outcome = mouth.enqueue_and_wait_timeout(args.text, Some(Duration::from_secs(60)))?;
    println!(
        "robot mouth diagnostic complete: device {}, duration {} ms",
        outcome.device.as_deref().unwrap_or("<unknown>"),
        outcome.duration_ms.unwrap_or_default()
    );
    Ok(())
}

fn run_whisper_transcribe(args: WhisperTranscribeArgs) -> Result<()> {
    use speaking::{AudioFrame, SpeechRecognizer, WhisperSpeechRecognizer};

    let model = args
        .model
        .or_else(|| env_path("PETE_WHISPER_MODEL"))
        .or_else(default_whisper_model_path)
        .context("missing Whisper model path; run `just setup-whisper`, set PETE_WHISPER_MODEL, or pass --model")?;
    let samples = read_wav_as_16khz_mono_f32(&args.wav)
        .with_context(|| format!("failed to read {}", args.wav.display()))?;
    if samples.is_empty() {
        return Ok(());
    }
    let mut recognizer = WhisperSpeechRecognizer::new_quiet_without_input_padding(&model)
        .with_context(|| format!("loading Whisper model {}", model.display()))?;
    recognizer.push_frame(&AudioFrame {
        sample_rate_hz: 16_000,
        channels: 1,
        samples,
    })?;
    let chunks = recognizer.poll_chunks()?;
    let transcript = chunks
        .into_iter()
        .map(|chunk| chunk.text)
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string();
    if !transcript.is_empty() {
        println!("{transcript}");
    }
    Ok(())
}

fn read_wav_as_16khz_mono_f32(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels.max(1));
    let source_rate = spec.sample_rate.max(1);
    let mut interleaved = Vec::new();
    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => {
            for sample in reader.samples::<f32>() {
                interleaved.push(sample?);
            }
        }
        (hound::SampleFormat::Int, 8) => {
            for sample in reader.samples::<i8>() {
                interleaved.push(sample? as f32 / i8::MAX as f32);
            }
        }
        (hound::SampleFormat::Int, 16) => {
            for sample in reader.samples::<i16>() {
                interleaved.push(sample? as f32 / i16::MAX as f32);
            }
        }
        (hound::SampleFormat::Int, 24 | 32) => {
            let scale = ((1_i64 << (spec.bits_per_sample - 1)) - 1) as f32;
            for sample in reader.samples::<i32>() {
                interleaved.push(sample? as f32 / scale);
            }
        }
        _ => anyhow::bail!(
            "unsupported WAV format: {:?} {} bits",
            spec.sample_format,
            spec.bits_per_sample
        ),
    }
    let mono = interleaved
        .chunks(channels)
        .map(|frame| frame.iter().copied().sum::<f32>() / frame.len().max(1) as f32)
        .collect::<Vec<_>>();
    Ok(resample_mono_linear(&mono, source_rate, 16_000))
}

fn resample_mono_linear(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if samples.is_empty() || source_rate == 0 || target_rate == 0 {
        return Vec::new();
    }
    if source_rate == target_rate {
        return samples.to_vec();
    }
    let output_len = (samples.len() as u64)
        .saturating_mul(u64::from(target_rate))
        .div_ceil(u64::from(source_rate)) as usize;
    let ratio = source_rate as f64 / target_rate as f64;
    let mut output = Vec::with_capacity(output_len);
    for index in 0..output_len {
        let pos = index as f64 * ratio;
        let left = pos.floor() as usize;
        let right = (left + 1).min(samples.len() - 1);
        let fraction = (pos - left as f64) as f32;
        let sample = samples[left] * (1.0 - fraction) + samples[right] * fraction;
        output.push(sample.clamp(-1.0, 1.0));
    }
    output
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn default_whisper_model_path() -> Option<PathBuf> {
    let data_home = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share"))
        })?;
    Some(
        data_home
            .join("pete")
            .join("models")
            .join("whisper")
            .join(DEFAULT_WHISPER_MODEL_FILENAME),
    )
}

fn robot_initialization_metadata(
    robot_mode: RobotMode,
    args: &RobotArgs,
    is_mock_body: bool,
    create_port: Option<&str>,
    active_sensor_count: usize,
    init_body: Option<&BodySense>,
    brainstem_capabilities: &pete_cockpit::CockpitCapabilities,
) -> serde_json::Value {
    let body_status = if is_mock_body {
        "mock Create body connected".to_string()
    } else if let Some(port) = create_port {
        format!("Create body connected on {port}")
    } else {
        "Create body connected".to_string()
    };
    let mode = match robot_mode {
        RobotMode::ReadOnly => "read-only",
        RobotMode::Slow => "slow",
        RobotMode::Disabled => "disabled",
    };
    serde_json::json!({
        "mode": mode,
        "body": body_status,
        "brainstem_capabilities": brainstem_capabilities,
        "battery_percent": init_body.map(|body| {
            (body.battery_level.clamp(0.0, 1.0) * 100.0).round() as u32
        }),
        "charging": init_body.map(|body| body.charging),
        "active_sensors": active_sensor_count,
        "requested_sensors": requested_robot_sensor_count(args),
        "ledger": args.ledger.clone(),
        "tick_ms": args.tick_ms,
        "dashboard": args.dashboard.map(|addr| addr.to_string()),
        "capture": args.capture.clone(),
        "brainstem_device_id": args.brainstem_device_id.clone(),
        "brainstem_boot_id": args.brainstem_boot_id.clone(),
        "reconnect_initial_backoff_ms": args.reconnect_initial_backoff_ms,
        "reconnect_max_backoff_ms": args.reconnect_max_backoff_ms,
    })
}

fn brainstem_firmware_identity(cockpit: &mut dyn Cockpit) -> Option<serde_json::Value> {
    let status = cockpit.get_status().ok()?;
    let fields = [
        "firmware_name",
        "firmware_version",
        "git_commit",
        "git_commit_short",
        "git_dirty",
        "build_timestamp",
        "build_profile",
        "build_target",
        "build_backend",
        "build_id",
    ];
    if let Ok(status) = serde_json::from_str::<serde_json::Value>(&status.raw) {
        let mut identity = serde_json::Map::new();
        for field in fields {
            if let Some(value) = status.get(field) {
                identity.insert(field.into(), value.clone());
            }
        }
        return (!identity.is_empty()).then_some(serde_json::Value::Object(identity));
    }
    let mut identity = serde_json::Map::new();
    for item in status.raw.split_ascii_whitespace() {
        let Some((key, value)) = item.split_once('=') else {
            continue;
        };
        if fields.contains(&key) {
            let value = match key {
                "git_dirty" => serde_json::Value::Bool(value == "true"),
                _ => serde_json::Value::String(value.to_string()),
            };
            identity.insert(key.into(), value);
        }
    }
    (!identity.is_empty()).then_some(serde_json::Value::Object(identity))
}

fn enqueue_default_bringup_outputs(
    mouth: &Option<QueuedPiperCpalMouth>,
    cockpit: &mut dyn Cockpit,
    initialization: &serde_json::Value,
) {
    play_robot_song(cockpit, "bring_up");
    play_robot_chirp(cockpit, "Confirm");
    let Some(mouth) = mouth.as_ref() else {
        return;
    };
    if let Some(mode) = initialization.get("mode").and_then(|value| value.as_str()) {
        enqueue_robot_mouth_text(
            mouth,
            &format!("Pete robot initialization complete in {mode} mode."),
        );
    }
    if let Some(body) = initialization.get("body").and_then(|value| value.as_str()) {
        enqueue_robot_mouth_text(mouth, &format!("{body}."));
    }
    match (
        initialization
            .get("battery_percent")
            .and_then(|value| value.as_u64()),
        initialization
            .get("charging")
            .and_then(|value| value.as_bool()),
    ) {
        (Some(percent), Some(charging)) => {
            let charging = if charging { "charging" } else { "not charging" };
            enqueue_robot_mouth_text(
                mouth,
                &format!("Battery is {percent} percent and {charging}."),
            );
        }
        _ => enqueue_robot_mouth_text(mouth, "Battery status is unavailable."),
    }
}

fn speak_robot_mouth_text_before_status(mouth: &QueuedPiperCpalMouth, text: &str) -> bool {
    println!("robot mouth speaking before body status: {text:?}");
    match mouth.enqueue_and_wait_timeout(text.to_string(), Some(Duration::from_secs(20))) {
        Ok(outcome) => {
            println!(
                "robot mouth completed before body status: device {}, duration {} ms",
                outcome.device.as_deref().unwrap_or("<unknown>"),
                outcome.duration_ms.unwrap_or_default()
            );
            true
        }
        Err(error) => {
            println!(
                "robot mouth pre-status speech failed; disabling mouth for this run and continuing: {error}"
            );
            false
        }
    }
}

fn play_event_script_outputs(
    mouth: &Option<QueuedPiperCpalMouth>,
    cockpit: &mut dyn Cockpit,
    tick: &RuntimeTick,
) {
    let Some(scripts) = tick.frame.now.extensions.get("event_scripts") else {
        return;
    };
    let Some(object) = scripts.as_object() else {
        return;
    };
    for sequence in object.values() {
        let Some(actions) = sequence.get("actions").and_then(|value| value.as_array()) else {
            continue;
        };
        for action in actions {
            let requested = action.get("requested").unwrap_or(action);
            if let Some(text) = requested.get("text").and_then(|value| value.as_str()) {
                if let Some(mouth) = mouth.as_ref() {
                    enqueue_robot_mouth_text(mouth, text);
                }
            } else if let Some(pattern) = requested.get("pattern").and_then(|value| value.as_str())
            {
                play_robot_chirp(cockpit, pattern);
            } else if let Some(name) = requested.get("name").and_then(|value| value.as_str()) {
                play_robot_song(cockpit, name);
            }
        }
    }
}

fn play_reign_audio_action(
    mouth: &Option<QueuedPiperCpalMouth>,
    cockpit: &mut dyn Cockpit,
    tick: &RuntimeTick,
    played: &mut HashSet<String>,
) {
    if tick.skill_request.is_some() {
        return;
    }
    let Some(action) = tick.chosen_action.as_ref() else {
        return;
    };
    let action_key = match action {
        ActionPrimitive::Speak { text } => format!("speak:{text}"),
        ActionPrimitive::Chirp { pattern } => format!("chirp:{pattern:?}"),
        _ => return,
    };
    let key = tick
        .frame
        .reign_input
        .as_ref()
        .map(|input| format!("reign:{}:{action_key}", input.id))
        .unwrap_or_else(|| format!("frame:{}:{action_key}", tick.frame.id));
    if !played.insert(key) {
        return;
    }
    match action {
        ActionPrimitive::Speak { text } => {
            if let Some(mouth) = mouth.as_ref() {
                enqueue_robot_mouth_text(mouth, text);
            } else {
                println!("robot mouth unavailable; skipped Reign speech {text:?}");
            }
        }
        ActionPrimitive::Chirp { pattern } => {
            play_robot_chirp(cockpit, &format!("{pattern:?}"));
        }
        _ => {}
    }
}

fn play_lua_skill_audio(
    mouth: &Option<QueuedPiperCpalMouth>,
    tick: &RuntimeTick,
    played: &mut HashSet<String>,
) {
    let Some(record) = tick.frame.now.extensions.get("motherbrain.skill_execution") else {
        return;
    };
    for (key, text) in lua_skill_speech_intents(record) {
        if !played.insert(key) {
            continue;
        }
        if let Some(mouth) = mouth.as_ref() {
            enqueue_robot_mouth_text(mouth, &text);
        } else {
            println!("robot mouth unavailable; skipped Lua skill speech {text:?}");
        }
    }
}

fn lua_skill_speech_intents(record: &serde_json::Value) -> Vec<(String, String)> {
    let execution_id = record
        .get("execution_id")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_default();
    let Some(trace) = record.get("trace").and_then(serde_json::Value::as_array) else {
        return Vec::new();
    };
    let mut intents = Vec::new();
    for event in trace {
        if event.get("kind").and_then(serde_json::Value::as_str) != Some("primitive")
            || event.get("operation").and_then(serde_json::Value::as_str) != Some("say")
        {
            continue;
        }
        let operation_id = event
            .get("operation_id")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or_default();
        let Some(text) = event
            .pointer("/detail/text")
            .and_then(serde_json::Value::as_str)
        else {
            continue;
        };
        let key = format!("lua:{execution_id}:{operation_id}");
        intents.push((key, text.to_string()));
    }
    intents
}

fn enqueue_robot_mouth_text(mouth: &QueuedPiperCpalMouth, text: &str) {
    match mouth.enqueue(text.to_string()) {
        Ok(()) => println!("robot mouth queued: {text:?}"),
        Err(error) => println!("robot mouth queue failed: {error}; text {text:?}"),
    }
}

fn play_robot_chirp(cockpit: &mut dyn Cockpit, pattern: &str) {
    play_body_song(
        cockpit,
        &format!("chirp {pattern}"),
        chirp_pattern_song(pattern),
    );
}

fn play_robot_song(cockpit: &mut dyn Cockpit, name: &str) {
    play_body_song(cockpit, name, robot_song(name));
}

fn play_body_song(cockpit: &mut dyn Cockpit, label: &str, song: BodySong) {
    let tones = song
        .tones
        .iter()
        .map(|tone| SongTone {
            note: tone.note,
            duration_64ths: tone.duration_64ths,
        })
        .collect::<Vec<_>>();
    match cockpit
        .song_define(0, &tones)
        .and_then(|()| cockpit.song_play(0))
    {
        Ok(()) => println!("robot cockpit song played: {label}"),
        Err(error) => println!("robot cockpit song skipped: {error}; song {label}"),
    }
}

fn chirp_pattern_song(pattern: &str) -> BodySong {
    BodySong::new(
        chirp_pattern_notes(pattern)
            .iter()
            .enumerate()
            .map(|(index, note)| {
                tone(
                    *note,
                    if index + 1 == chirp_pattern_notes(pattern).len() {
                        8
                    } else {
                        6
                    },
                )
            })
            .collect::<Vec<_>>(),
    )
}

fn chirp_pattern_notes(pattern: &str) -> &'static [u8] {
    match normalized_chirp_pattern(pattern).as_str() {
        "confirm" => &[79, 84, 79],
        "warning" => &[79, 75],
        "hello" => &[72, 76, 79],
        "goodbye" => &[79, 76, 72],
        "curious" => &[72, 76, 74],
        "idea" => &[76, 81, 84],
        "goalacquired" => &[72, 79, 84, 91],
        "searching" => &[72, 74, 76, 74],
        "sawsomething" => &[84, 91],
        "surprise" => &[72, 84],
        "learned" => &[74, 79, 83],
        "personrecognized" => &[76, 79, 84, 79],
        "objectrecognized" => &[79, 84, 76],
        "placerecognized" => &[79, 84, 72],
        "didntunderstand" => &[79, 81, 78],
        "docking" => &[67, 72, 76, 79],
        "chargingstarted" => &[60, 67, 72],
        "sleep" => &[79, 76, 72, 67],
        "wake" => &[67, 72, 79],
        _ => &[72],
    }
}

fn normalized_chirp_pattern(pattern: &str) -> String {
    pattern
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase()
}

fn robot_song(name: &str) -> BodySong {
    match name {
        "bring_up" => BodySong::new([
            tone(60, 8),
            tone(64, 8),
            tone(67, 8),
            tone(72, 12),
            tone(67, 6),
            tone(72, 14),
        ]),
        "mournful_bump" => BodySong::new([tone(64, 12), tone(63, 12), tone(60, 16), tone(55, 20)]),
        _ => BodySong::new([tone(60, 8), tone(67, 8), tone(72, 12)]),
    }
}

fn tone(note: u8, duration_64ths: u8) -> BodyTone {
    BodyTone::new(note, duration_64ths)
}
