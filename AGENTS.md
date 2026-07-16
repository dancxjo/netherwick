# Working Agreement

- Commit completed, coherent slices of work as you go. Keep unrelated worktree
  changes out of those commits.
- If someone wants to control a commit themselves, they should make that commit
  themselves; do not hold completed agent work for manual commit choreography.
- Do not wait for CI/CD or a release process to make progress. Run focused local
  checks when they help; do not run the full test suite before every commit.
- End the task with `just sup` so staged work, unstaged work, and the Unreleased
  changelog are reconciled before handoff.
