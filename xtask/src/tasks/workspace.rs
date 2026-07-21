fn eval_scenario_smoke() -> Result<()> {
    for args in [
        [
            "empty-room",
            "2",
            "10",
            "data/reports/scenario/empty-smoke.json",
        ],
        [
            "obstacle-avoidance",
            "2",
            "10",
            "data/reports/scenario/obstacle-smoke.json",
        ],
        [
            "corner-trap",
            "1",
            "40",
            "data/reports/scenario/corner-trap-smoke.json",
        ],
        [
            "charger-seeking",
            "2",
            "10",
            "data/reports/scenario/charge-smoke.json",
        ],
    ] {
        pete_tools(
            [
                "eval-scenario",
                "--scenario",
                args[0],
                "--episodes",
                args[1],
                "--steps",
                args[2],
                "--out",
                args[3],
            ],
            &[],
        )?;
    }
    Ok(())
}

fn pico_bootsel_mount(umount: bool, kernel_name: &str) -> Result<()> {
    let user = env_or(
        "PICO_BOOTSEL_USER",
        &env_or("SUDO_USER", &env_or("USER", "")),
    );
    let uid = output("id", &["-u", &user]).unwrap_or_else(|_| "0".to_owned());
    let group = env_or("PICO_BOOTSEL_GROUP", &user);
    let gid = output("getent", &["group", &group])
        .ok()
        .and_then(|entry| entry.split(':').nth(2).map(str::to_owned))
        .unwrap_or_else(|| uid.clone());
    let mount = env::var("PICO_BOOTSEL_MOUNT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env_or("PICO_BOOTSEL_MOUNT_BASE", "/media")).join(if uid == "0" {
                "RPI-RP2".to_owned()
            } else {
                format!("{user}/RPI-RP2")
            })
        });
    if umount {
        if is_mountpoint(&mount) {
            run_program(ProcessCommand::new("umount").arg(&mount))?;
        }
        let _ = fs::remove_dir(&mount);
        return Ok(());
    }
    let device = format!("/dev/{kernel_name}");
    #[cfg(unix)]
    let is_block_device = fs::metadata(&device)
        .map(|metadata| metadata.file_type().is_block_device())
        .unwrap_or(false);
    #[cfg(not(unix))]
    let is_block_device = Path::new(&device).exists();
    if !is_block_device {
        return fail(format!("BOOTSEL block device not found: {device}"));
    }
    fs::create_dir_all(&mount)?;
    if !is_mountpoint(&mount) {
        run_program(ProcessCommand::new("mount").args([
            "-t",
            "vfat",
            "-o",
            &format!("uid={uid},gid={gid},umask=022,noatime,flush"),
            &device,
            path(&mount)?,
        ]))?;
    }
    #[cfg(unix)]
    fs::set_permissions(&mount, fs::Permissions::from_mode(0o755))?;
    println!(
        "Mounted RPI-RP2 at {} for uid={uid} gid={gid}",
        mount.display()
    );
    Ok(())
}

fn is_mountpoint(path: &Path) -> bool {
    ProcessCommand::new("mountpoint")
        .args(["-q", path.to_str().unwrap_or("")])
        .status()
        .is_ok_and(|status| status.success())
}

fn codex_sync() -> Result<()> {
    if !Path::new(".git").is_dir() {
        return fail("codex-sync: no git repository in the current directory");
    }
    let status = output("git", &["status", "--short", "--branch"])?;
    if status.lines().count() <= 1 {
        run_program(ProcessCommand::new("git").args(["pull", "--ff-only"]))?;
        if status.contains("[ahead ") {
            run_program(ProcessCommand::new("git").arg("push"))?;
        }
        return Ok(());
    }
    let staged_diff = output("git", &["diff", "--cached"])?;
    let unstaged_diff = output("git", &["diff"])?;
    let short_log = output("git", &["log", "--oneline", "--decorate", "-n", "20"])?;
    let prompt = format!(
        "\
Context for this sync:\n\
`git status --short --branch`:\n{status}\n\n\
`git diff --cached`:\n{staged_diff}\n\n\
`git diff`:\n{unstaged_diff}\n\n\
Recent commits (`git log --oneline --decorate -n 20`):\n{short_log}\n\n\
Treat already staged changes as candidate work even when another agent or person staged them: include every ready staged or unstaged semantic change in CHANGELOG.md under Unreleased without removing releases. Summarize and classify each change as ready or ongoing, commit only ready semantic groups, then git pull --ff-only and git push. Use git commands to carry out that workflow. Do not run CI or create extra files; leave uncertain work uncommitted."
    );
    let summary_path =
        env::temp_dir().join(format!("netherwick-codex-sync-{}.md", std::process::id()));
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {msg}").expect("valid spinner template"),
    );
    spinner.set_message("Codex is reviewing and syncing the worktree");
    spinner.enable_steady_tick(Duration::from_millis(120));
    let terminal_title = TerminalTitleSpinner::start("Codex is reviewing and syncing the worktree");
    let mut child = ProcessCommand::new("codex")
        .args([
            "--ask-for-approval",
            "never",
            "exec",
            "--sandbox",
            "danger-full-access",
            "--ephemeral",
            "--model",
            "gpt-5.3-codex-spark",
            "-c",
            "model_reasoning_effort=high",
            "--output-last-message",
            path(&summary_path)?,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    child
        .stdin
        .take()
        .ok_or_else(|| io::Error::other("failed to open codex stdin"))?
        .write_all(prompt.as_bytes())?;
    let status = child.wait()?;
    spinner.finish_and_clear();
    drop(terminal_title);
    let summary = fs::read_to_string(&summary_path).unwrap_or_default();
    let _ = fs::remove_file(&summary_path);
    if !status.success() {
        return fail(format!("codex-sync failed with {status}"));
    }
    if !summary.trim().is_empty() {
        println!("{summary}");
    }
    Ok(())
}

struct TerminalTitleSpinner {
    running: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl TerminalTitleSpinner {
    fn start(message: &str) -> Option<Self> {
        if !io::stderr().is_terminal() {
            return None;
        }

        let running = Arc::new(AtomicBool::new(true));
        let spinner_running = Arc::clone(&running);
        let message = terminal_title_text(message);
        let thread = thread::spawn(move || {
            const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            let mut frame = 0;
            while spinner_running.load(Ordering::Relaxed) {
                let mut stderr = io::stderr().lock();
                let _ = write!(stderr, "\x1b]0;{} {}\x07", FRAMES[frame], message);
                let _ = stderr.flush();
                frame = (frame + 1) % FRAMES.len();
                thread::sleep(Duration::from_millis(120));
            }
        });

        Some(Self {
            running,
            thread: Some(thread),
        })
    }
}

impl Drop for TerminalTitleSpinner {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        let mut stderr = io::stderr().lock();
        let _ = write!(stderr, "\x1b]0;Netherwick\x07");
        let _ = stderr.flush();
    }
}

fn terminal_title_text(value: &str) -> String {
    value
        .chars()
        .filter(|character| !character.is_control())
        .collect()
}

fn output(program: &str, args: &[&str]) -> Result<String> {
    let result = ProcessCommand::new(program).args(args).output()?;
    if result.status.success() {
        Ok(String::from_utf8_lossy(&result.stdout).trim().to_owned())
    } else {
        fail(format!("{program} failed with {}", result.status))
    }
}
