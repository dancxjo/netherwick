# Working Agreement

- Commit completed, coherent slices of work as you go. Keep unrelated worktree
  changes out of those commits.
- If someone wants to control a commit themselves, they should make that commit
  themselves; do not hold completed agent work for manual commit choreography.
- Do not wait for CI/CD or a release process to make progress. Run focused local
  checks when they help; do not run the full test suite before every commit.
- End the task with `just sup` so staged work, unstaged work, and the Unreleased
  changelog are reconciled before handoff.
- Keep `just flash` and its authorized BOOTSEL handshake backwards compatible
  with previously flashed brainstem firmware. Capability-vocabulary, handshake,
  and service-authorization changes are migrations: retain host acceptance for
  older advertised contracts, add a regression test for the prior contract, and
  validate against attached pre-upgrade firmware before flashing when available.

# Development workflow

- Make your own commits as you make progress. Keep them small, coherent, and easy to review instead of accumulating one large end-of-task change.
- Commit only the files and hunks that belong to your work. Preserve unrelated user changes in the worktree, and use concise, descriptive commit messages.
- During development, validate the package or behavior you are changing with focused tests and checks. Netherwick's full test suite is slow, so do not feel compelled to run every workspace test after every small change.
- Prefer the narrowest useful command, such as a package-level test or check. Remember that `crates/pete-brainstem` is excluded from the root workspace and must be tested with `cargo test --manifest-path crates/pete-brainstem/Cargo.toml ...`.
- At the end of the task, run `just sup` as the broad final verification pass to catch formatting, lint, test, or integration issues that focused checks may have missed. Fix any failures attributable to your work and commit those fixes before handoff.
