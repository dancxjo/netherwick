fn setup_pico_bootsel() -> Result<()> {
    pete(["build", "--release", "-p", "xtask"])?;
    let user = env_or("SUDO_USER", &env_or("USER", "root"));
    let group = output("id", &["-gn", &user])?;
    run_program(ProcessCommand::new("sudo").args([
        "install",
        "-d",
        "-m",
        "0755",
        "/usr/local/lib/netherwick",
    ]))?;
    run_program(ProcessCommand::new("sudo").args([
        "install",
        "-m",
        "0755",
        "target/release/xtask",
        "/usr/local/lib/netherwick/pico-bootsel-mount",
    ]))?;
    run_program(ProcessCommand::new("sudo").args([
        "install",
        "-m",
        "0644",
        "configs/systemd/netherwick-pico-bootsel-mount@.service",
        "/etc/systemd/system/netherwick-pico-bootsel-mount@.service",
    ]))?;
    run_program(ProcessCommand::new("sudo").args([
        "install",
        "-m",
        "0644",
        "configs/udev/99-netherwick-pico-bootsel.rules",
        "/etc/udev/rules.d/99-netherwick-pico-bootsel.rules",
    ]))?;
    let defaults = env::temp_dir().join(format!("netherwick-pico-bootsel-{}", std::process::id()));
    fs::write(
        &defaults,
        format!(
            "PICO_BOOTSEL_USER={user}\nPICO_BOOTSEL_GROUP={group}\nPICO_BOOTSEL_MOUNT_BASE=/media\n"
        ),
    )?;
    run_program(ProcessCommand::new("sudo").args([
        "install",
        "-m",
        "0644",
        path(&defaults)?,
        "/etc/default/netherwick-pico-bootsel",
    ]))?;
    fs::remove_file(defaults)?;
    run_program(ProcessCommand::new("sudo").args(["systemctl", "daemon-reload"]))?;
    run_program(ProcessCommand::new("sudo").args(["udevadm", "control", "--reload-rules"]))?;
    let _ = ProcessCommand::new("sudo")
        .args([
            "udevadm",
            "trigger",
            "--subsystem-match=block",
            "--property-match=ID_FS_LABEL=RPI-RP2",
        ])
        .status();
    println!("Pico BOOTSEL automount installed for {user}:{group}.");
    Ok(())
}

fn setup_kinect_from_source() -> Result<()> {
    let source = Path::new(".vendor/libfreenect");
    if !source.join(".git").is_dir() {
        run_program(ProcessCommand::new("git").args([
            "clone",
            "https://github.com/OpenKinect/libfreenect.git",
            path(source)?,
        ]))?;
    }
    run_program(ProcessCommand::new("cmake").args([
        "-S",
        path(source)?,
        "-B",
        ".vendor/libfreenect/build",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_CPP=ON",
        "-DBUILD_AUDIO=ON",
        "-DBUILD_EXAMPLES=OFF",
        "-DBUILD_OPENNI2_DRIVER=OFF",
    ]))?;
    run_program(ProcessCommand::new("cmake").args(["--build", ".vendor/libfreenect/build", "-j"]))?;
    run_program(ProcessCommand::new("sudo").args([
        "cmake",
        "--install",
        ".vendor/libfreenect/build",
    ]))
}

fn fetch_cyw43() -> Result<()> {
    let directory = Path::new("crates/pete-brainstem/firmware/cyw43");
    fs::create_dir_all(directory)?;
    let base = format!(
        "https://raw.githubusercontent.com/embassy-rs/embassy/{}/cyw43-firmware",
        env_or("CYW43_FIRMWARE_REF", "main")
    );
    for file in [
        "43439A0.bin",
        "43439A0_clm.bin",
        "nvram_rp2040.bin",
        "LICENSE-permissive-binary-license-1.0.txt",
    ] {
        run_program(ProcessCommand::new("curl").args([
            "-fL",
            "--retry",
            "3",
            "--retry-delay",
            "2",
            "-o",
            path(&directory.join(file))?,
            &format!("{base}/{file}"),
        ]))?;
    }
    Ok(())
}

fn elf_to_uf2(name: &str) -> Result<()> {
    run_program(ProcessCommand::new("elf2uf2-rs").args([
        "crates/pete-brainstem/target/thumbv6m-none-eabi/release/pete-brainstem",
        &format!("crates/pete-brainstem/target/thumbv6m-none-eabi/release/{name}"),
    ]))
}

fn flash() -> Result<()> {
    run(Command::BrainstemPicoWUf2)?;
    let uf2 = Path::new(
        "crates/pete-brainstem/target/thumbv6m-none-eabi/release/pete-brainstem-pico-w.uf2",
    );
    if !uf2.is_file() || uf2.metadata()?.len() == 0 {
        return fail(format!("UF2 not found: {}", uf2.display()));
    }
    let mut mount = bootsel_mount().or_else(mount_bootsel_block);
    if mount.is_none() {
        println!("Requesting authorized BOOTSEL via USB CDC");
        let mut request = ProcessCommand::new("cargo");
        request
            .args([
                "run",
                "-q",
                "-p",
                "pete-cockpit",
                "--example",
                "service_bootsel",
            ])
            .env("PETE_BOOTSEL_USB", "1");
        let mut requested = request.status()?.success();
        let bootsel_url = env_or("PICO_W_BOOTSEL_URL", "http://192.168.4.1/command");
        let host = bootsel_host(&bootsel_url);
        if !requested && connected_to_brainstem_wifi(&host) {
            println!("USB BOOTSEL failed; requesting authorized BOOTSEL via {bootsel_url}");
            let mut request = ProcessCommand::new("cargo");
            request
                .args([
                    "run",
                    "-q",
                    "-p",
                    "pete-cockpit",
                    "--example",
                    "service_bootsel",
                ])
                .env("PETE_BRAINSTEM_HTTP_HOST", &host);
            requested = request.status()?.success();
        }
        if !requested {
            if !env_flag("PETE_ALLOW_LEGACY_BOOTSEL") {
                return fail(
                    "authorized BOOTSEL failed; set PETE_ALLOW_LEGACY_BOOTSEL=1 only for explicit development recovery",
                );
            }
            if !connected_to_brainstem_wifi(&host) {
                return fail("legacy BOOTSEL requires an active Pete brainstem Wi-Fi connection");
            }
            eprintln!("WARNING: USING UNAUDITED LEGACY BOOTSEL DEVELOPMENT FALLBACK");
            run_program(ProcessCommand::new("curl").args([
                "-fsS",
                "--max-time",
                "3",
                "-H",
                "Content-Type: application/json",
                "-d",
                r#"{"kind":"bootsel","command_id":1}"#,
                &bootsel_url,
            ]))?;
        }
        println!("Waiting for RPI-RP2 mount");
        let timeout = env_or("PICO_W_MOUNT_TIMEOUT_SECS", "30")
            .parse::<u64>()
            .unwrap_or(30);
        for _ in 0..timeout {
            mount = bootsel_mount().or_else(mount_bootsel_block);
            if mount.is_some() {
                break;
            }
            thread::sleep(Duration::from_secs(1));
        }
    }
    let mount =
        mount.ok_or_else(|| io::Error::other("RPI-RP2 mount was not found; set PICO_W_MOUNT"))?;
    println!("Copying {} to {}", uf2.display(), mount.display());
    fs::copy(uf2, mount.join("pete-brainstem-pico-w.uf2"))?;
    run_program(&mut ProcessCommand::new("sync"))?;
    println!("Flash copy complete");
    Ok(())
}

fn bootsel_mount() -> Option<PathBuf> {
    let explicit = env_or("PICO_W_MOUNT", "");
    let candidates = [
        PathBuf::from(explicit),
        PathBuf::from(format!("/media/{}/RPI-RP2", env_or("USER", "root"))),
        PathBuf::from(format!("/run/media/{}/RPI-RP2", env_or("USER", "root"))),
        PathBuf::from("/media/RPI-RP2"),
        PathBuf::from("/Volumes/RPI-RP2"),
    ];
    candidates
        .into_iter()
        .find(|candidate| {
            !candidate.as_os_str().is_empty()
                && candidate.is_dir()
                && ProcessCommand::new("test")
                    .args(["-w", path(candidate).unwrap_or("")])
                    .status()
                    .is_ok_and(|status| status.success())
        })
        .or_else(|| {
            output("lsblk", &["-rpo", "LABEL,MOUNTPOINT"])
                .ok()?
                .lines()
                .find_map(|line| {
                    let mut fields = line.split_whitespace();
                    if fields.next()? != "RPI-RP2" {
                        return None;
                    }
                    Some(PathBuf::from(fields.next()?))
                })
                .filter(|candidate| candidate.is_dir())
        })
}

fn mount_bootsel_block() -> Option<PathBuf> {
    let listing = output("lsblk", &["-rnpo", "LABEL,PATH,FSTYPE,MOUNTPOINT"]).ok()?;
    let block = listing.lines().find_map(|line| {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        (fields.len() >= 3 && fields[0] == "RPI-RP2" && fields[2] == "vfat")
            .then(|| fields[1].to_owned())
    })?;
    if program_exists("udisksctl")
        && ProcessCommand::new("udisksctl")
            .args(["mount", "-b", &block])
            .status()
            .is_ok_and(|status| status.success())
    {
        return bootsel_mount();
    }
    let mount = PathBuf::from(format!("/media/{}/RPI-RP2", env_or("USER", "root")));
    if run_program(ProcessCommand::new("sudo").args(["mkdir", "-p", path(&mount).ok()?])).is_err()
        || run_program(ProcessCommand::new("sudo").args([
            "mount",
            "-t",
            "vfat",
            "-o",
            &format!(
                "uid={},gid={},umask=022",
                output("id", &["-u"]).ok()?,
                output("id", &["-g"]).ok()?
            ),
            &block,
            path(&mount).ok()?,
        ]))
        .is_err()
    {
        return None;
    }
    bootsel_mount()
}

fn bootsel_host(url: &str) -> String {
    let host = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .unwrap_or(url)
        .split('/')
        .next()
        .unwrap_or(url);
    if host.contains(':') {
        host.to_owned()
    } else {
        format!("{host}:80")
    }
}

fn connected_to_brainstem_wifi(host: &str) -> bool {
    let pete_network = if program_exists("nmcli") {
        output("nmcli", &["-t", "-f", "active,ssid", "dev", "wifi"])
            .unwrap_or_default()
            .lines()
            .any(|line| line.to_ascii_lowercase().starts_with("yes:pete-"))
    } else if program_exists("iwgetid") {
        output("iwgetid", &["-r"])
            .unwrap_or_default()
            .to_ascii_lowercase()
            .starts_with("pete-")
    } else {
        false
    };
    pete_network && curl_ok(&format!("http://{host}/status.json"))
}
