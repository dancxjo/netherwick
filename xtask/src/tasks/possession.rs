fn skull() -> Result<()> {
    let python = if Path::new("skeleton/brainstem/.venv/bin/python").is_file() {
        "skeleton/brainstem/.venv/bin/python"
    } else {
        "python"
    };
    run_program(
        ProcessCommand::new(python)
            .arg("skull.py")
            .current_dir("skeleton/brainstem"),
    )
}

fn cockpit_backend() -> Result<String> {
    if let Ok(value) = env::var("PETE_COCKPIT_BACKEND") {
        if !value.is_empty() {
            return Ok(value);
        }
    }
    let configured = env_or("PETE_COCKPIT_PORT", "auto");
    let port = if configured == "auto" {
        serial_candidates().into_iter().next().unwrap_or_default()
    } else {
        configured
    };
    if !env_flag("PETE_SKIP_COCKPIT_UART") && !port.is_empty() {
        let status = ProcessCommand::new("cargo")
            .args([
                "run",
                "-q",
                "-p",
                "pete-cockpit",
                "--example",
                "contract_check",
                "--",
                "uart",
                &port,
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()?;
        if status.success() {
            return Ok("uart".to_owned());
        }
    }
    let wifi = output("nmcli", &["-t", "-f", "active,ssid", "dev", "wifi"])
        .unwrap_or_default()
        .lines()
        .any(|line| line.to_ascii_lowercase().starts_with("yes:pete-"));
    if wifi
        && ProcessCommand::new("curl")
            .args([
                "-fsS",
                "--connect-timeout",
                "1",
                "--max-time",
                "2",
                &format!(
                    "http://{}/status.json",
                    env_or("PETE_BRAINSTEM_HTTP_HOST", "192.168.4.1:80")
                ),
            ])
            .status()
            .is_ok_and(|s| s.success())
    {
        return Ok("wifi".to_owned());
    }
    Ok(if env_flag("PETE_SKIP_COCKPIT_UART") {
        "none"
    } else {
        "uart"
    }
    .to_owned())
}

fn serial_candidates() -> Vec<String> {
    let mut candidates = Vec::new();
    for directory in ["/dev/serial/by-id", "/dev"] {
        if let Ok(entries) = fs::read_dir(directory) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.contains("Pete_Brainstem")
                    || (directory == "/dev"
                        && (name.starts_with("ttyACM") || name.starts_with("ttyUSB")))
                {
                    candidates.push(entry.path().display().to_string());
                }
            }
        }
    }
    candidates.sort();
    candidates
}

fn robot(args: &[String], overrides: &[(&str, String)]) -> Result<()> {
    run_program(&mut robot_process(args, overrides)?)
}

fn robot_process(args: &[String], overrides: &[(&str, String)]) -> Result<ProcessCommand> {
    ensure_memory_servers()?;
    dev_cert()?;
    let mut command = vec![
        "robot".to_owned(),
        "--mode".to_owned(),
        override_or_env(overrides, "PETE_ROBOT_MODE", "read-only"),
        "--cockpit".to_owned(),
        match override_value(overrides, "PETE_COCKPIT_BACKEND") {
            Some(backend) => backend,
            None => cockpit_backend()?,
        },
        "--create-port".to_owned(),
        override_or_env(overrides, "PETE_COCKPIT_PORT", "auto"),
        "--ledger".to_owned(),
        override_or_env(overrides, "PETE_ROBOT_LEDGER", "data/ledger/real/robot"),
    ];
    optional_arg(&mut command, "CAMERA_DEVICE", "/dev/video0", "--camera");
    optional_arg(&mut command, "MIC_DEVICE", "", "--mic");
    optional_arg(&mut command, "IMU_DEVICE", "", "--imu");
    optional_arg(&mut command, "GPS_SERIAL_PORT", "", "--gps");
    if let Ok(lidar) = env::var("LIDAR_SERIAL_PORT") {
        if !lidar.is_empty() {
            command.extend([
                "--lidar".to_owned(),
                lidar,
                "--lidar-yaw-deg".to_owned(),
                env_or("LIDAR_YAW_DEG", "0"),
                "--lidar-pitch-deg".to_owned(),
                env_or("LIDAR_PITCH_DEG", "0"),
                "--lidar-roll-deg".to_owned(),
                env_or("LIDAR_ROLL_DEG", "0"),
                "--lidar-height-m".to_owned(),
                env_or("LIDAR_HEIGHT_M", "0"),
                "--lidar-forward-m".to_owned(),
                env_or("LIDAR_FORWARD_M", "0"),
                "--lidar-left-m".to_owned(),
                env_or("LIDAR_LEFT_M", "0"),
            ]);
        }
    }
    if env_flag("PETE_KINECT_DEPTH")
        || env::var_os("PETE_KINECT_DEPTH").is_none_or(|value| value != OsStr::new("0"))
    {
        command.extend([
            "--kinect-depth".to_owned(),
            "--kinect-rgb-target-luma".to_owned(),
            env_or("KINECT_RGB_TARGET_LUMA", "0.32"),
            "--kinect-rgb-auto-gain-max".to_owned(),
            env_or("KINECT_RGB_AUTO_GAIN_MAX", "3.0"),
            "--kinect-rgb-gain".to_owned(),
            env_or("KINECT_RGB_GAIN", "1.0"),
            "--kinect-rgb-gamma".to_owned(),
            env_or("KINECT_RGB_GAMMA", "0.80"),
            "--kinect-rgb-brightness".to_owned(),
            env_or("KINECT_RGB_BRIGHTNESS", "0.0"),
        ]);
    }
    command.extend([
        "--dashboard".to_owned(),
        env_or("PETE_ROBOT_DASHBOARD", "0.0.0.0:3000"),
        "--dashboard-tls".to_owned(),
        "--dashboard-tls-cert".to_owned(),
        env_or("PETE_ROBOT_DASHBOARD_TLS_CERT", "certs/pete-dev.crt"),
        "--dashboard-tls-key".to_owned(),
        env_or("PETE_ROBOT_DASHBOARD_TLS_KEY", "certs/pete-dev.key"),
    ]);
    command.extend(args.iter().cloned());
    let mut process = ProcessCommand::new("cargo");
    process.args(["run", "-p", "pete-tools", "--"]);
    process.args(&command);
    process.env(
        "PETE_TTS_OUTPUT_DEVICE",
        env_or("PETE_TTS_OUTPUT_DEVICE", ""),
    );
    for (key, value) in overrides {
        process.env(key, value);
    }
    Ok(process)
}

fn override_value(overrides: &[(&str, String)], name: &str) -> Option<String> {
    overrides
        .iter()
        .find_map(|(key, value)| (*key == name).then(|| value.clone()))
}

fn override_or_env(overrides: &[(&str, String)], name: &str, default: &str) -> String {
    override_value(overrides, name).unwrap_or_else(|| env_or(name, default))
}

fn optional_arg(command: &mut Vec<String>, env_name: &str, default: &str, flag: &str) {
    let value = env_or(env_name, default);
    if !value.is_empty() {
        command.extend([flag.to_owned(), value]);
    }
}

fn possess(args: &[String]) -> Result<()> {
    let (robot_args, robot_mode) = split_mode_override(args);
    let robot_mode = normalize_possession_mode(&robot_mode);
    let backend_was_explicit =
        env::var("PETE_COCKPIT_BACKEND").is_ok_and(|value| !value.is_empty());
    let mut backend = if backend_was_explicit {
        env_or("PETE_COCKPIT_BACKEND", "uart")
    } else {
        String::new()
    };
    let mut device = env_or("PETE_BRAINSTEM_DEVICE_ID", "");
    let mut boot = env_or("PETE_BRAINSTEM_BOOT_ID", "unknown");
    let mut port = env_or("PETE_COCKPIT_PORT", "auto");
    if backend == "local" {
        let (live_device, live_boot) = bootstrap_brainstem(None)?;
        if !device.is_empty()
            && live_device != device
            && !env_flag("PETE_ACCEPT_BRAINSTEM_REPLACEMENT")
        {
            return fail(format!(
                "local brainstem {live_device} does not match pinned {device}; set PETE_ACCEPT_BRAINSTEM_REPLACEMENT=1 only if this is the intended RPi 5"
            ));
        }
        device = live_device;
        boot = live_boot;
        set_dotenv("PETE_BRAINSTEM_DEVICE_ID", &device)?;
        set_dotenv("PETE_BRAINSTEM_BOOT_ID", &boot)?;
    } else if port != "auto" && !port.is_empty() && !Path::new(&port).exists() {
        if let Some(detected) = single_brainstem_port() {
            println!(
                "Configured PETE_COCKPIT_PORT is missing: {port}\nDetected one wired brainstem candidate: {}",
                detected.display()
            );
            let (live_device, live_boot) = bootstrap_brainstem(Some(&detected))?;
            if live_device != device && !env_flag("PETE_ACCEPT_BRAINSTEM_REPLACEMENT") {
                return fail(format!(
                    "detected brainstem {live_device}, but .env pins {device}; rerun with PETE_ACCEPT_BRAINSTEM_REPLACEMENT=1 to accept the wired replacement"
                ));
            }
            device = live_device;
            boot = live_boot;
            port = detected.display().to_string();
            set_dotenv("PETE_BRAINSTEM_DEVICE_ID", &device)?;
            set_dotenv("PETE_BRAINSTEM_BOOT_ID", &boot)?;
            set_dotenv("PETE_COCKPIT_PORT", &port)?;
            println!(
                "Updated .env brainstem pin from wired USB: device={device} boot={boot} port={port}"
            );
        }
    }
    if device.is_empty() {
        return fail("set PETE_BRAINSTEM_DEVICE_ID in .env");
    }
    if !backend_was_explicit && port != "auto" && Path::new(&port).exists() {
        backend = "uart".to_owned();
    } else if !backend_was_explicit {
        backend = cockpit_backend()?;
    }
    if backend.is_empty() {
        backend = "uart".to_owned();
    }
    let (mut status, mut log) =
        possession_attempt(&robot_args, &robot_mode, &device, &boot, &backend, &port)?;
    if status.success() {
        return Ok(());
    }
    let host = env_or("PETE_BRAINSTEM_HTTP_HOST", "192.168.4.1:80");
    if !backend_was_explicit && backend == "uart" && connected_to_brainstem_wifi(&host) {
        backend = "wifi".to_owned();
        println!("Brainstem USB/UART failed; retrying possession over Pete Wi-Fi at {host}.");
        (status, log) =
            possession_attempt(&robot_args, &robot_mode, &device, &boot, &backend, &port)?;
        if status.success() {
            return Ok(());
        }
    }
    if backend == "wifi" && log.contains("reason_code: InvalidIdentity") {
        println!("Wi-Fi identity continuity is not established; bootstrapping the pinned brainstem over USB.");
        bootstrap_brainstem(single_brainstem_port().as_deref())?;
        (status, log) =
            possession_attempt(&robot_args, &robot_mode, &device, &boot, &backend, &port)?;
        if status.success() {
            return Ok(());
        }
    }
    if let Some(live_boot) = boot_identity_mismatch(&log) {
        set_dotenv("PETE_BRAINSTEM_BOOT_ID", &live_boot)?;
        println!(
            "Accepted current boot identity for pinned device {device}; updated .env and retrying."
        );
        (status, _) = possession_attempt(
            &robot_args,
            &robot_mode,
            &device,
            &live_boot,
            &backend,
            &port,
        )?;
    }
    if status.success() {
        Ok(())
    } else {
        fail(format!("possession exited with {status}"))
    }
}

fn possession_attempt(
    args: &[String],
    robot_mode: &str,
    device: &str,
    boot: &str,
    backend: &str,
    port: &str,
) -> Result<(std::process::ExitStatus, String)> {
    let endpoint = match backend {
        "wifi" => env_or("PETE_BRAINSTEM_HTTP_HOST", "192.168.4.1:80"),
        "local" => env_or("PETE_BRAINSTEM_LOCAL_ADDR", "127.0.0.1:8787"),
        _ => port.to_owned(),
    };
    let explicit_tick_ms = long_option_value(args, "--tick-ms");
    let tick_ms = explicit_tick_ms
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| env_or("PETE_POSSESSION_TICK_MS", DEFAULT_POSSESSION_TICK_MS));
    println!(
        "Taking brainstem possession over {backend} at {endpoint}\ndevice={device} boot={boot}\ncontrol target: {tick_ms} ms; limits: 50 mm/s linear, 500 mrad/s angular; exit performs STOP then exorcize"
    );
    let mut robot_args = vec![
        "--brainstem-device-id".to_owned(),
        device.to_owned(),
        "--brainstem-boot-id".to_owned(),
        boot.to_owned(),
        "--max-linear-mm-s".to_owned(),
        "50".to_owned(),
        "--max-angular-mrad-s".to_owned(),
        "500".to_owned(),
        "--autonomous-motion".to_owned(),
        "--imu".to_owned(),
        "none".to_owned(),
        "--gps".to_owned(),
        "none".to_owned(),
        "--llm-provider".to_owned(),
        "disabled".to_owned(),
        "--capture".to_owned(),
        env_or("PETE_POSSESSION_CAPTURE", "data/captures/real/possession"),
    ];
    if explicit_tick_ms.is_none() {
        robot_args.extend(["--tick-ms".to_owned(), tick_ms]);
    }
    robot_args.extend(args.iter().cloned());
    let mut command = robot_process(
        &robot_args,
        &[
            ("PETE_ROBOT_MODE", robot_mode.to_owned()),
            ("PETE_COCKPIT_BACKEND", backend.to_owned()),
            ("PETE_COCKPIT_PORT", port.to_owned()),
            (
                "PETE_ROBOT_LEDGER",
                env_or("PETE_POSSESSION_LEDGER", "data/ledger/real/possession"),
            ),
            ("CAMERA_DEVICE", "".to_owned()),
            ("MIC_DEVICE", "".to_owned()),
            ("IMU_DEVICE", "".to_owned()),
            ("GPS_SERIAL_PORT", "".to_owned()),
            ("PETE_KINECT_DEPTH", "0".to_owned()),
        ],
    )?;
    run_program_captured(&mut command)
}

fn split_mode_override(args: &[String]) -> (Vec<String>, String) {
    let mut robot_mode = "regular".to_owned();
    let mut robot_args = Vec::new();
    let mut i = 0usize;
    while i < args.len() {
        if args[i] == "--mode" {
            if let Some(next) = args.get(i + 1) {
                robot_mode = next.to_owned();
                i += 2;
                continue;
            }
            robot_args.push(args[i].to_owned());
            i += 1;
            continue;
        }
        if let Some(mode) = args[i].strip_prefix("--mode=") {
            robot_mode = mode.to_owned();
            i += 1;
            continue;
        }
        robot_args.push(args[i].to_owned());
        i += 1;
    }
    (robot_args, robot_mode)
}

fn normalize_possession_mode(mode: &str) -> String {
    match mode {
        "regular" => "regular".to_owned(),
        _ => mode.to_owned(),
    }
}

fn long_option_value<'a>(args: &'a [String], option: &str) -> Option<&'a str> {
    args.iter().enumerate().find_map(|(index, arg)| {
        if arg == option {
            args.get(index + 1).map(String::as_str)
        } else {
            arg.strip_prefix(option)
                .and_then(|value| value.strip_prefix('='))
        }
    })
}

fn run_program_captured(
    command: &mut ProcessCommand,
) -> Result<(std::process::ExitStatus, String)> {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let log = Arc::new(Mutex::new(String::new()));
    let stdout = stream_and_capture(
        child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("missing child stdout"))?,
        false,
        Arc::clone(&log),
    );
    let stderr = stream_and_capture(
        child
            .stderr
            .take()
            .ok_or_else(|| io::Error::other("missing child stderr"))?,
        true,
        Arc::clone(&log),
    );
    let status = child.wait()?;
    stdout
        .join()
        .map_err(|_| io::Error::other("stdout reader panicked"))??;
    stderr
        .join()
        .map_err(|_| io::Error::other("stderr reader panicked"))??;
    let captured = log
        .lock()
        .map_err(|_| io::Error::other("captured output lock poisoned"))?
        .clone();
    Ok((status, captured))
}

fn stream_and_capture<R: Read + Send + 'static>(
    reader: R,
    is_stderr: bool,
    log: Arc<Mutex<String>>,
) -> thread::JoinHandle<io::Result<()>> {
    thread::spawn(move || {
        for line in BufReader::new(reader).lines() {
            let line = line?;
            if is_stderr {
                eprintln!("{line}");
            } else {
                println!("{line}");
            }
            let mut captured = log
                .lock()
                .map_err(|_| io::Error::other("captured output lock poisoned"))?;
            captured.push_str(&line);
            captured.push('\n');
        }
        Ok(())
    })
}

fn single_brainstem_port() -> Option<PathBuf> {
    let candidates = fs::read_dir("/dev/serial/by-id")
        .ok()?
        .flatten()
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .contains("Pete_Brainstem")
        })
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    (candidates.len() == 1).then(|| candidates[0].clone())
}

fn bootstrap_brainstem(port: Option<&Path>) -> Result<(String, String)> {
    let mut command = ProcessCommand::new("cargo");
    command
        .args([
            "run",
            "-q",
            "-p",
            "pete-cockpit",
            "--example",
            "motherbrain_bootstrap",
            "--",
            "--identity-only",
        ])
        .env("PETE_BRAINSTEM_DEVICE_ID", "");
    if let Some(port) = port {
        command.env("PETE_COCKPIT_PORT", port);
    }
    let output = command.output()?;
    print!("{}", String::from_utf8_lossy(&output.stdout));
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    if !output.status.success() {
        return fail(format!(
            "brainstem identity bootstrap failed with {}",
            output.status
        ));
    }
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let device = prefixed_value(&text, "brainstem identity: ")
        .ok_or_else(|| io::Error::other("bootstrap did not report a brainstem identity"))?;
    let boot = prefixed_value(&text, "brainstem boot: ")
        .ok_or_else(|| io::Error::other("bootstrap did not report a brainstem boot identity"))?;
    Ok((device, boot))
}

fn prefixed_value(text: &str, prefix: &str) -> Option<String> {
    text.lines()
        .filter_map(|line| line.strip_prefix(prefix))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .next_back()
        .map(str::to_owned)
}

fn boot_identity_mismatch(log: &str) -> Option<String> {
    log.lines()
        .filter_map(|line| {
            line.strip_prefix("Error: brainstem boot identity mismatch: expected ")?
                .split_once(" received ")
                .map(|(_, received)| {
                    received
                        .split_whitespace()
                        .next()
                        .unwrap_or(received)
                        .to_owned()
                })
        })
        .next_back()
}

fn set_dotenv(key: &str, value: &str) -> Result<()> {
    let path = Path::new(".env");
    let existing = fs::read_to_string(path).unwrap_or_default();
    let mut replaced = false;
    let mut lines = existing
        .lines()
        .map(|line| {
            if line.starts_with(&format!("{key}=")) {
                replaced = true;
                format!("{key}={value}")
            } else {
                line.to_owned()
            }
        })
        .collect::<Vec<_>>();
    if !replaced {
        lines.push(format!("{key}={value}"));
    }
    fs::write(path, format!("{}\n", lines.join("\n")))?;
    Ok(())
}
