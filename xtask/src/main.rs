use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
mod physical_qa;
#[cfg(unix)]
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::{
    env,
    error::Error,
    ffi::OsStr,
    fs,
    io::{self, BufRead, BufReader, IsTerminal, Read, Write},
    path::{Path, PathBuf},
    process::{Command as ProcessCommand, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, SystemTime},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const DEFAULT_POSSESSION_TICK_MS: &str = "20";

include!("tasks/cli.rs");
include!("tasks/process.rs");
include!("tasks/firmware.rs");
include!("tasks/possession.rs");
include!("tasks/services.rs");
include!("tasks/training.rs");
include!("tasks/workspace.rs");

#[cfg(test)]
#[path = "main_tests.rs"]
mod tests;
