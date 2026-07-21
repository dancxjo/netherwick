
use super::{
    boot_identity_mismatch, bootsel_host, evolution_dataset_files, long_option_value,
    neat_generation_limit, prefixed_value, terminal_title_text, Cli,
};
use clap::Parser;
use std::fs;

#[test]
fn continuation_limit_extends_completed_generation() {
    let state = std::env::temp_dir().join(format!("xtask-neat-limit-{}", std::process::id()));
    fs::write(&state, r#"{"generation_in_stage":243}"#).unwrap();
    assert_eq!(neat_generation_limit(&state, None, 120, 120), 363);
    fs::write(&state, r#"{"generation_in_stage":363}"#).unwrap();
    assert_eq!(neat_generation_limit(&state, None, 120, 120), 483);
    assert_eq!(neat_generation_limit(&state, Some(77), 120, 120), 77);
    let _ = fs::remove_file(state);
}

#[test]
fn terminal_title_text_omits_terminal_control_characters() {
    assert_eq!(
        terminal_title_text("sync\u{1b}]0;spoof\u{7}"),
        "sync]0;spoof"
    );
}

#[test]
fn hardware_identity_output_is_parsed_conservatively() {
    let output = "brainstem identity: pete-17\nbrainstem boot: boot-4\n";
    assert_eq!(
        prefixed_value(output, "brainstem identity: ").as_deref(),
        Some("pete-17")
    );
    assert_eq!(
        boot_identity_mismatch(
            "Error: brainstem boot identity mismatch: expected boot-3 received boot-4"
        )
        .as_deref(),
        Some("boot-4")
    );
    assert_eq!(bootsel_host("http://192.168.4.1/command"), "192.168.4.1:80");
}

#[test]
fn dataset_retention_only_claims_evolution_episodes() {
    let root = std::env::temp_dir().join(format!("xtask-dataset-files-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("nested")).unwrap();
    fs::write(root.join("nested/level-1-seed-2-genome-3.jsonl"), "episode").unwrap();
    fs::write(root.join("keep-me.jsonl"), "other").unwrap();
    let files = evolution_dataset_files(&root).unwrap();
    assert_eq!(files.len(), 1);
    assert!(files[0].0.ends_with("level-1-seed-2-genome-3.jsonl"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn public_alias_commands_remain_parseable() {
    for command in [
        "check",
        "flash",
        "possess",
        "go",
        "train",
        "evolve-infinite",
        "codex-sync",
    ] {
        Cli::try_parse_from(["xtask", command]).unwrap();
    }
}

#[test]
fn possession_tick_override_accepts_split_and_joined_forms() {
    assert_eq!(
        long_option_value(&["--tick-ms".to_owned(), "12".to_owned()], "--tick-ms"),
        Some("12")
    );
    assert_eq!(
        long_option_value(&["--tick-ms=8".to_owned()], "--tick-ms"),
        Some("8")
    );
    assert_eq!(long_option_value(&[], "--tick-ms"), None);
}
