fn ensure_memory_servers() -> Result<()> {
    let neo4j = env_or(
        "PETE_NEO4J_HTTP_URL",
        &format!(
            "http://127.0.0.1:{}",
            env_or("PETE_NEO4J_HTTP_PORT", "7474")
        ),
    );
    let qdrant = env_or(
        "PETE_QDRANT_URL",
        &format!(
            "http://127.0.0.1:{}",
            env_or("PETE_QDRANT_HTTP_PORT", "6333")
        ),
    );
    if curl_ok(&neo4j)
        && (curl_ok(&format!("{}/readyz", qdrant.trim_end_matches('/')))
            || curl_ok(&format!("{}/collections", qdrant.trim_end_matches('/'))))
    {
        return Ok(());
    }
    run(Command::Servers)
}

fn curl_ok(url: &str) -> bool {
    ProcessCommand::new("curl")
        .args(["-fsS", "--max-time", "2", url])
        .status()
        .is_ok_and(|s| s.success())
}

fn dev_cert() -> Result<()> {
    let cert = PathBuf::from(env_or(
        "PETE_ROBOT_DASHBOARD_TLS_CERT",
        "certs/pete-dev.crt",
    ));
    let key = PathBuf::from(env_or("PETE_ROBOT_DASHBOARD_TLS_KEY", "certs/pete-dev.key"));
    if cert.is_file() && key.is_file() {
        return Ok(());
    }
    fs::create_dir_all(cert.parent().unwrap_or(Path::new(".")))?;
    fs::create_dir_all(key.parent().unwrap_or(Path::new(".")))?;
    let san = if lan_ip() == "127.0.0.1" {
        String::new()
    } else {
        format!(",IP:{}", lan_ip())
    };
    run_program(ProcessCommand::new("openssl").args([
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        path(&key)?,
        "-out",
        path(&cert)?,
        "-days",
        "365",
        "-subj",
        "/CN=pete.local",
        "-addext",
        &format!("subjectAltName=DNS:localhost,DNS:pete.local,IP:127.0.0.1{san}"),
    ]))
}

fn lan_ip() -> String {
    output("hostname", &["-I"])
        .ok()
        .and_then(|ips| ips.split_whitespace().next().map(str::to_owned))
        .unwrap_or_else(|| "127.0.0.1".to_owned())
}

fn go(target: &str) -> Result<()> {
    if target != "virtual" {
        return fail("usage: just go virtual");
    }
    dev_cert()?;
    let port = env_or("PETE_LIVE_PORT", "8787");
    let lan_ip = lan_ip();
    println!(
        "Pete Dream World is starting.\nVirtual training theater is collecting experience.\nDesktop: https://127.0.0.1:{port}/view/3d\nHeadset/LAN: https://{lan_ip}:{port}/view/3d\nScene JSON: https://{lan_ip}:{port}/view/scene\nThis serves robot/dream-world sensor data on the LAN; use only on trusted networks."
    );
    if program_exists("qrencode") {
        let _ = ProcessCommand::new("qrencode")
            .args([
                "-t",
                "ANSIUTF8",
                &format!("https://{lan_ip}:{port}/view/3d"),
            ])
            .status();
    }
    pete(["build", "-p", "pete-tools"])?;
    let mut child = ProcessCommand::new("target/debug/pete")
        .args([
            "sim",
            "--live",
            "--live-tls",
            "--live-addr",
            &format!("0.0.0.0:{port}"),
            "--live-tls-cert",
            "certs/pete-dev.crt",
            "--live-tls-key",
            "certs/pete-dev.key",
            "--action-selector",
            &env_or("PETE_ACTION_SELECTOR", "goal"),
            "--scenario",
            &env_or("PETE_SCENARIO", "dream"),
            "--steps",
            &env_or("PETE_SIM_STEPS", "1000000000"),
            "--tick-delay-ms",
            &env_or("PETE_TICK_DELAY_MS", "100"),
            "--ledger",
            &env_or("PETE_LEDGER", "data/ledger/virtual-live"),
        ])
        .spawn()?;
    if env_flag("PETE_OPEN_BROWSER") && program_exists("xdg-open") {
        if program_exists("curl") {
            for _ in 0..80 {
                if ProcessCommand::new("curl")
                    .args(["-kfsS", &format!("https://127.0.0.1:{port}/view/scene")])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .is_ok_and(|status| status.success())
                {
                    break;
                }
                if child.try_wait()?.is_some() {
                    break;
                }
                thread::sleep(Duration::from_millis(250));
            }
        } else {
            thread::sleep(Duration::from_secs(2));
        }
        let _ = ProcessCommand::new("xdg-open")
            .arg(format!("https://127.0.0.1:{port}/view/3d"))
            .spawn();
    }
    let status = child.wait()?;
    if status.success() {
        Ok(())
    } else {
        fail(format!("virtual world exited with {status}"))
    }
}
