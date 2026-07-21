# Higher-brain provisioning and exchange

Netherwick now has three deliberately separate roles and one versioned data
exchange skeleton.

- The Pico W brainstem exclusively owns Create OI, physical safety, reflexes,
  motion possession, heartbeats, emergency recovery, and the separately wired
  Pi RUN reset authority. Nothing in `pete-higher-brain` imports the Cockpit
  protocol or can acquire a motion lease.
- The Pi 5 motherbrain remains canonical for Pete's current graph, vector
  stores, ledger, experience epoch, live inference, and bounded online
  learning. It exports immutable snapshots; it does not replicate its live
  databases or surrender their authoritative copy.
- A forebrain is a stable enrolled node identity plus a fresh boot identity and
  advertised capabilities. The role can move to any compatible Linux GPU or
  CPU machine.

The implementation lives in `crates/pete-higher-brain`. Its protocol versions
are `netherwick-higher-brain/1`, `experience_bundle/1`, `job/1`, and
`model_candidate/1`.

## Network separation and fallback

Host control-path failover is specified in
[`026-brainstem-transit-failover.md`](026-brainstem-transit-failover.md).
Joining the brainstem network changes reachability only: it neither changes a
host's role nor grants possession. Forebrain searches direct Ethernet and
ordinary Wi-Fi before brainstem transit, retains a reachable motherbrain as the
body-facing host, and becomes a takeover candidate only after a fresh
authoritative no-controller observation and a bounded grace interval. An
atomic brainstem acquisition remains the sole transition into controlling
fallback. Motherbrain USB loss changes transport, not role.

The Pico-hosted Wi-Fi remains the bodily control plane: possession, leases,
heartbeats, bounded commands, acknowledgements, safety/mode status, modest
telemetry, events, and motherbrain recovery requests. Bundles, database
exports, Kinect data, model weights, images, packages, and software updates
never use that service.

The data plane is independently configurable. Eligible addresses are selected
in this order by default:

1. working Ethernet links;
2. working ordinary or dedicated Wi-Fi links;
3. other explicitly eligible links.

Route metric breaks ties. `allowed_interfaces` can narrow the set. Every name
in `brainstem_interfaces` is denied for binding and advertising unless the
operator explicitly sets `allow_brainstem_interface = true`. Empty allowed and
seed lists do not mean “use the Pico”; they mean enumerate any non-loopback,
up, non-brainstem interface. A direct Ethernet cable, isolated switch, normal
LAN, or dedicated Wi-Fi all work without internet access. Multicast discovery
is supplemented by multiple configurable unicast seeds for networks that
filter multicast. Each selected address is advertised, so a caller can try
Ethernet and then the next endpoint after a connection failure.

See `configs/higher-brain/forebrain.example.toml`. The older
`configs/network/20-interbrain.network` remains a useful direct-link profile,
not an interface-name requirement.

Discovery is advisory JSON over trusted-local UDP. An advertisement contains
protocol version, node/boot identities, CPU architecture and cores, total and
available RAM, GPU vendor/model/runtime/usable memory when detected, workspace
capacity, runtimes, supported jobs, load, readiness, software version, schema
versions, and eligible endpoints. Discovery and authorization are separate.

## Authorization and transfer

The library defines independent scopes for discovery, experience transfer, job
submission, job cancellation, candidate return, candidate staging, activation,
rollback, and provisioning. A motion possession lease grants none of them.
The example token authorization file stores only SHA-256 fingerprints, never
plaintext credentials.

Provisioned hosts use OpenSSH as the initial secure noninteractive transport.
An enrolled public key is forced into `internal-sftp` rooted at
`/var/lib/netherwick/forebrain`, with `restrict` disabling shell, forwarding,
and agent features. Use separate enrolled keys where operators need different
administrative scopes. Bundle/job/candidate library entry points still require
their specific `Principal` scope; filesystem service commands run as the
dedicated account. Do not put private keys, tokens, or a generated inventory in
the repository.

The transfer algorithm writes into `.partial`, resumes matching files, resets
contradictory partial files, verifies every checksum, and atomically renames a
complete directory. An acknowledgement is a separate record, so retries never
mutate the bundle. Over SFTP or rsync, upload to a temporary name, verify with
`bundle-verify`, and rename only after completion. The same library algorithm
is used by local/simulation tests.

## Experience bundles

`BundleBuilder` wraps the actual JSONL ledger through `JsonlLedgerAdapter` and
accepts deterministic graph, vector, sensor, and blob-reference exports through
small source adapters. It retains full frame records rather than flattening
away sensor artifacts, vector provenance, preprocessing labels, calibration,
or observations needed to reconstruct examples.

Each manifest records the Pete/source identity, time range, event cursors,
checkpoint identities, software/build identity, store schemas, active models,
configuration/calibration identities, typed content paths, sizes, and SHA-256
checksums. Audio, images, point clouds, and other large media can be included or
referenced with their hashes and preprocessing/calibration identity.

Construction happens in a staging directory. Files are synced, the manifest
and matching `.complete` marker are written, and the directory is atomically
renamed. The bundle ID is derived from its request and content. A range index
returns the already validated bundle when the identical source range is
exported again; it cannot silently assign contradictory content to the same
range identity. A corrected source must use a new checkpoint/configuration
identity.

The current graph and vectors remain on the motherbrain. This milestone does
not implement live Neo4j/Qdrant replication. Their deterministic export files
are bundle inputs, which keeps database-specific snapshot mechanics behind an
adapter.

## Jobs and candidates

The versioned job envelope declares a job class, exact input bundle IDs,
resource requirements, required runtimes, parameters, submitter identity, and
training build identity. Forebrain-suitable classes include dataset building,
long replay, graph analysis, representation/perception training, substantial
fine-tuning, searches, consolidation, and full evaluation. Motherbrain live
learning has a different bounded configuration: 20% CPU, 256 MiB, 25 ms steps
by default, with persistence/network-pressure pauses.

Every job is an atomically persisted JSON record. States are `queued`,
`running`, `succeeded`, `failed`, `cancelled`, and `interrupted`, with concise
progress, detail, and transition history. A worker restart turns a durable
`running` record into `interrupted`; an authorized retry can requeue it. Invalid
or unavailable resources fail before execution.

`dataset_construction` is the useful physical-experience job. It accepts only a
complete WorldLab capture whose source is `RealRobot`, retains every capture
file and raw asset in the checksummed experience bundle, and deterministically
builds a provenance-bearing frame/modality index. Its evaluation reports
required-modality frame coverage, mean modality coverage, temporal
monotonicity, asset-reference count, and capture-writer drops. The candidate
declares `physical_experience_dataset/1` output compatibility and targets only
the motherbrain dataset library.

`fixture_digest` remains a small deterministic infrastructure smoke test. It
verifies source bundles, derives a digest/count/byte artifact, evaluates its
deterministic properties, and emits a complete model candidate. Candidate manifests include
algorithm family, learned artifacts, preprocessing and I/O schemas, training
build, exact bundle IDs, parameters, evaluation, hardware/runtime needs,
deployment target, rollback compatibility, and checksums.

The motherbrain treats returned candidates as untrusted. Its lifecycle is
`received -> validated -> staged -> activated`, with `rejected`, `rolled_back`,
and `superseded` recorded separately. Staging copies a verified immutable
candidate into a library. Activation atomically replaces an `active/current`
symlink only after compatibility validation and policy approval, retaining
`active/previous`. Rollback atomically swaps them. An interrupted copy or failed
validation never touches `current`. Arbitrary learned artifacts targeting
`brainstem` are rejected.

## Docked consolidation

`ConsolidationCoordinator` persists one phase at a time. Its production
`start_with_power_assessment` path refuses to start
unless Create is stopped, docked, charging, has no active motion authority, and
fresh independent X1202 GPIO6/GPIO16/fuel-gauge and Create OI evidence satisfy
the battery policy. The compatibility `start` interface remains for older
callers, but it cannot represent evidence age and is not the physical acceptance
path. `tick_with_power_assessment` rechecks evidence before every expensive
phase and pauses in place when external power or charging evidence disappears.

The cycle checkpoints an epoch, discovers an authorized compatible forebrain,
transfers complete bundles, submits and waits for jobs, retrieves candidates,
validates/stages them, and waits at an explicit activation boundary. Every
network/training phase is retryable in place. State survives a motherbrain
restart. The coordinator reads bodily readiness but exposes no brainstem
command interface, so loss of the data plane, worker, or motherbrain cannot
block brainstem safety or Create charging.

## Provision a forebrain

Use a local untracked copy of the inventory example and pin a reviewed commit
or tag. Add only public SSH keys.

```bash
cp provisioning/forebrain/inventory.example.yml /tmp/forebrains.yml
$EDITOR /tmp/forebrains.yml
just forebrain-provision-check /tmp/forebrains.yml
just forebrain-provision /tmp/forebrains.yml
```

The idempotent playbook installs build, OpenSSH, rsync, networking, and
diagnostic packages; creates the locked-password `netherwick-forebrain`
account; creates explicit experience/dataset/job/candidate/log/temp/workspace
directories; detects GPU capability without requiring one; checks out and
builds the selected revision; installs a config only when none exists; enrolls
restricted public keys; and enables worker/discovery systemd services.

Validate and inspect the node:

```bash
ssh operator@FOREBRAIN sudo /usr/local/bin/pete-higher-brain validate-node \
  --config /etc/netherwick/forebrain.toml
ssh operator@FOREBRAIN sudo -u netherwick-forebrain \
  /usr/local/bin/pete-higher-brain capabilities \
  --node-id forebrain-example-01 \
  --workspace /var/lib/netherwick/forebrain/workspace
systemctl status netherwick-forebrain-worker netherwick-forebrain-discovery
```

A CPU-only machine is ready for the fixture/replay/dataset/evaluation skeleton;
GPU-only jobs are advertised only when a GPU runtime is detected.

Recovery is idempotent: rerun `site.yml` after repairing configuration or
connectivity. It preserves an existing `/etc/netherwick/forebrain.toml`.
Removal preserves received/training data by default:

```bash
just forebrain-remove /tmp/forebrains.yml
```

To explicitly erase data and the account:

```bash
ansible-playbook -i /tmp/forebrains.yml provisioning/forebrain/remove.yml \
  -e netherwick_purge_data=true
```

## Local end-to-end demonstration

This demonstration uses no Pico, Create, X1202, GPU, or specific interface.

```bash
ROOT="$(mktemp -d)"
mkdir -p "$ROOT/mother" "$ROOT/fore"/{experience,datasets,jobs,candidates,workspace,tmp} "$ROOT/logs"
printf '%s\n' '{"t_ms":100,"observation":"fixture","preprocessing":"fixture/1"}' > "$ROOT/sensors.jsonl"
cat > "$ROOT/request.json" <<'JSON'
{"pete_id":"pete","source_node_id":"motherbrain-sim","begin_timestamp_ms":100,"end_timestamp_ms":100,"event_range":{"first_cursor":"1","last_cursor":"1"},"source_checkpoints":{"ledger":"fixture-1"},"software_identity":"demo-build","schema_versions":{"ledger":"1"},"active_model_versions":{},"configuration_identity":"demo-config","calibration_identity":"demo-calibration"}
JSON
cargo run -q -p pete-higher-brain -- bundle-create \
  --request "$ROOT/request.json" --output "$ROOT/mother" --sensors "$ROOT/sensors.jsonl"
BUNDLE_PATH="$(find "$ROOT/mother" -maxdepth 1 -name '*.bundle' -print -quit)"
BUNDLE_ID="$(cargo run -q -p pete-higher-brain -- bundle-verify "$BUNDLE_PATH" | jq -r .bundle_id)"
cargo run -q -p pete-higher-brain -- bundle-transfer "$BUNDLE_PATH" \
  --destination "$ROOT/fore/experience" --acknowledgements "$ROOT/mother/acks" --receiver forebrain-sim
cargo run -q -p pete-higher-brain -- job-create --bundle-id "$BUNDLE_ID" \
  --output "$ROOT/job.json" --software-identity demo-build
cat > "$ROOT/forebrain.toml" <<EOF
node_id = "forebrain-sim"
workspace = "$ROOT/fore/workspace"
bundles = "$ROOT/fore/experience"
datasets = "$ROOT/fore/datasets"
jobs = "$ROOT/fore/jobs"
candidates = "$ROOT/fore/candidates"
logs = "$ROOT/logs"
temporary = "$ROOT/fore/tmp"
poll_interval_seconds = 1
EOF
cargo run -q -p pete-higher-brain -- job-submit --config "$ROOT/forebrain.toml" --envelope "$ROOT/job.json"
cargo run -q -p pete-higher-brain -- worker --config "$ROOT/forebrain.toml" --once
JOB_ID="$(jq -r .job_id "$ROOT/job.json")"
CANDIDATE_ID="$(cargo run -q -p pete-higher-brain -- job-status --config "$ROOT/forebrain.toml" "$JOB_ID" | jq -r .candidate_id)"
CANDIDATE_PATH="$ROOT/fore/candidates/$CANDIDATE_ID.candidate"
cargo run -q -p pete-higher-brain -- candidate-validate "$CANDIDATE_PATH"
cargo run -q -p pete-higher-brain -- candidate-receive --store "$ROOT/mother/models" "$CANDIDATE_PATH"
cargo run -q -p pete-higher-brain -- candidate-stage --store "$ROOT/mother/models" "$CANDIDATE_ID"
cargo run -q -p pete-higher-brain -- candidate-activate --store "$ROOT/mother/models" --approve "$CANDIDATE_ID"
readlink "$ROOT/mother/models/active/current"
```

## Physical-experience dataset lifecycle

Start from a completed capture produced by `capture-real` or a physical
possession session. `bundle-create --capture` refuses `Sim`, `Replay`, and
`Unknown` sources and checks that the manifest frame count matches
`frames.jsonl` before checksumming the complete capture tree.

```bash
ROOT="$(mktemp -d)"
mkdir -p "$ROOT/mother" "$ROOT/fore"/{experience,jobs,candidates,workspace} "$ROOT/acks"
cargo run -q -p pete-higher-brain -- bundle-create \
  --request physical-experience-request.json \
  --output "$ROOT/mother" \
  --capture data/captures/real/rpi5-smoke
BUNDLE_PATH="$(find "$ROOT/mother" -maxdepth 1 -name '*.bundle' -print -quit)"
BUNDLE_ID="$(cargo run -q -p pete-higher-brain -- bundle-verify "$BUNDLE_PATH" | jq -r .bundle_id)"
```

Exercise an interrupted transfer, then resume it using the same command without
the byte budget:

```bash
cargo run -q -p pete-higher-brain -- bundle-transfer "$BUNDLE_PATH" \
  --destination "$ROOT/fore/experience" --acknowledgements "$ROOT/acks" \
  --receiver forebrain-physical --byte-budget 65536
cargo run -q -p pete-higher-brain -- bundle-transfer "$BUNDLE_PATH" \
  --destination "$ROOT/fore/experience" --acknowledgements "$ROOT/acks" \
  --receiver forebrain-physical
```

Create and submit the useful job. Choose required modalities that the capture
was meant to provide; the minimum ratio makes coverage a deterministic
candidate-production gate rather than an informational count.

```bash
cargo run -q -p pete-higher-brain -- job-create \
  --job-class dataset-construction --bundle-id "$BUNDLE_ID" \
  --required-modality body --required-modality range \
  --minimum-required-modality-frame-ratio 0.90 \
  --software-identity "$(git rev-parse HEAD)" --output "$ROOT/job.json"
cargo run -q -p pete-higher-brain -- job-submit \
  --config "$ROOT/forebrain.toml" --envelope "$ROOT/job.json"
cargo run -q -p pete-higher-brain -- worker --config "$ROOT/forebrain.toml" --once
```

If the worker was stopped while the durable record was running, reopening it
marks the job interrupted. Requeue it explicitly and run the worker again:

```bash
JOB_ID="$(jq -r .job_id "$ROOT/job.json")"
cargo run -q -p pete-higher-brain -- job-retry \
  --config "$ROOT/forebrain.toml" "$JOB_ID"
cargo run -q -p pete-higher-brain -- worker --config "$ROOT/forebrain.toml" --once
```

Use the existing `candidate-receive`, `candidate-stage`,
`candidate-activate --approve`, and `candidate-rollback` commands for the
explicit lifecycle. The focused integration test exercises interrupted
transfer, worker restart/retry, incompatible-candidate rejection without an
active-link change, approval-gated activation, rollback, and the forbidden
brainstem target in one path:

```bash
cargo test -p pete-higher-brain \
  physical_dataset_exercises_resumption_restart_rejection_activation_and_rollback
```

Run the flow again with a changed fixture/checkpoint identity to create a second
candidate, activate it, then exercise rollback:

```bash
cargo run -q -p pete-higher-brain -- candidate-rollback --store "$ROOT/mother/models"
```

The only hardware-dependent work left is validating X1202 thresholds/GPIO on
the incoming board, deriving trustworthy dock state from the real Create OI
status path, validating interface and multicast behavior on the chosen Pi/GPU
adapters, and exercising transfer interruption on the physical links.
