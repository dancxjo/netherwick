fn pete<const N: usize>(args: [&str; N]) -> Result<()> {
    run_program(ProcessCommand::new("cargo").args(args))
}
fn ansible<const N: usize>(args: [&str; N]) -> Result<()> {
    run_program(ProcessCommand::new("ansible-playbook").args(args))
}
fn docker<const N: usize>(args: [&str; N]) -> Result<()> {
    run_program(ProcessCommand::new("docker").args(args))
}
fn brainstem<const N: usize>(args: [&str; N]) -> Result<()> {
    run_program(
        ProcessCommand::new("cargo")
            .args(args)
            .current_dir("crates/pete-brainstem"),
    )
}
fn path(value: &Path) -> Result<&str> {
    value
        .to_str()
        .ok_or_else(|| io::Error::other("path is not UTF-8").into())
}
fn fail<T>(message: impl Into<String>) -> Result<T> {
    Err(io::Error::other(message.into()).into())
}

fn run_program(command: &mut ProcessCommand) -> Result<()> {
    let status = command.status()?;
    if status.success() {
        Ok(())
    } else {
        fail(format!("command failed with {status}"))
    }
}

fn pete_tools<'a>(args: impl IntoIterator<Item = &'a str>, envs: &[(&str, String)]) -> Result<()> {
    let mut command = ProcessCommand::new("cargo");
    command.args(["run", "-p", "pete-tools", "--"]);
    command.args(args);
    for (key, value) in envs {
        command.env(key, value);
    }
    run_program(&mut command)
}

fn pete_cockpit<'a>(args: impl IntoIterator<Item = &'a str>) -> Result<()> {
    let mut command = ProcessCommand::new("cargo");
    command.args([
        "run",
        "-q",
        "-p",
        "pete-cockpit",
        "--bin",
        "pete-cockpit",
        "--",
    ]);
    command.args(args);
    command.env("CARGO_BUILD_JOBS", env_or("CARGO_BUILD_JOBS", "1"));
    run_program(&mut command)
}

fn env_or(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_owned())
}
fn env_flag(name: &str) -> bool {
    matches!(env_or(name, "").as_str(), "1" | "true" | "on" | "yes")
}
fn program_exists(name: &str) -> bool {
    ProcessCommand::new("sh")
        .args(["-c", &format!("command -v {name} >/dev/null 2>&1")])
        .status()
        .is_ok_and(|s| s.success())
}

fn rust_tool(name: &str) -> ProcessCommand {
    if program_exists(name) {
        ProcessCommand::new(name)
    } else {
        ProcessCommand::new(
            PathBuf::from(env_or("HOME", "."))
                .join(".cargo/bin")
                .join(name),
        )
    }
}
fn data_home() -> PathBuf {
    env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env_or("HOME", ".")).join(".local/share"))
}

fn fetch_asset(destination: &Path, url: &str) -> Result<()> {
    if destination.is_file() && destination.metadata()?.len() > 0 {
        return Ok(());
    }
    fs::create_dir_all(
        destination
            .parent()
            .ok_or_else(|| io::Error::other("asset has no parent"))?,
    )?;
    run_program(ProcessCommand::new("curl").args([
        "-fL",
        "--retry",
        "3",
        "--retry-delay",
        "2",
        "-o",
        path(destination)?,
        url,
    ]))
}
