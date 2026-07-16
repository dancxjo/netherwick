# 025 Workspace Automation

Repository automation is implemented by the workspace `xtask` crate. The
`Justfile` is a compatibility and discoverability layer: its recipes forward to
the corresponding `cargo xtask` subcommand instead of maintaining a second
shell implementation.

Use the familiar entrypoints for ordinary work:

```bash
just --list
just check
just go virtual
just possess
just train --neat locomotion
just sup
```

The equivalent direct interface is useful in scripts and when passing arguments
that a `just` recipe intentionally constrains:

```bash
cargo run -q -p xtask -- --help
cargo run -q -p xtask -- check
```

## Operational boundaries

Hardware workflows retain their safety checks in `xtask`:

- `flash` requests authenticated BOOTSEL through the cockpit, permits the
  unaudited HTTP fallback only when explicitly enabled, waits for a writable
  `RPI-RP2` mount, and then copies the UF2.
- `possess` requires a pinned brainstem identity, accepts replacement hardware
  only through an explicit wired opt-in, maintains the boot identity, and may
  fall back from failed UART to an available Pete Wi-Fi link.
- `setup-pico-bootsel` installs the built `xtask` binary as the systemd/udev
  mount helper. The internal `pico-bootsel-mount` command is not an operator
  workflow.

Training continuation also lives in `xtask`. Locomotion NEAT training chooses,
in order, an explicit resume checkpoint, an explicit founders report, a current
trainer state, a historical founders report, or a fresh run. Its generation
limit extends the completed generation in the current stage unless explicitly
overridden.

The removed helper scripts map to these commands:

| Former helper | Replacement |
| --- | --- |
| `scripts/codex-sync.sh` | `cargo xtask codex-sync` / `just sup` |
| `scripts/pico-bootsel-mount.sh` | internal `cargo xtask pico-bootsel-mount` |
| `scripts/neat-generation-limit.sh` | internal `cargo xtask neat-generation-limit` |

The executable repository-root `wuzzup` wrapper is a portable shorthand for
`just sup`.
