# netherwick

`netherwick` is a Rust workspace for Pete Netherwick, an embodied self-training robot architecture.

Pete gathers raw sensors, memory recalls, internal drives, model predictions, surprise, and LLM guidance into `Now`. `Now` is compressed into an `ExperienceLatent`, used to imagine futures, choose actions, and train from consequences.

Pete acts through high-level action primitives. A hard-coded autonomic layer keeps the body safe. The LLM may command, reflect, critique, and teach, but cannot bypass safety.

Every hard-coded behavior is replaceable. It can run directly, shadow-train a model, compare with a model, promote a model, or fall back to safe hand-written logic.

Pete Netherwick is an embodied predictive organism: a robot body with reflexes, an experience ledger, compact learned present, imagined futures, memory returning as sensation, swappable learned behaviors, and an LLM consciousness that commands and teaches while safety protects the body.

## Setup

On Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install -y just
just setup
```

`just setup` installs the Linux build dependencies plus Kinect 1 userspace support through `libfreenect` when distro packages are available.

Useful commands:

```bash
just check
just test
just sim
just inspect-ledger
just hardware-env
```

## Docker services

Copy `.env.example` to `.env` if you want to override ports or passwords. Then:

```bash
just servers      # Neo4j + Qdrant
just live-server  # Neo4j + Qdrant + live sim server
```

The live view is at `http://localhost:8787/view`. Neo4j Browser is at `http://localhost:7474` with the default `.env.example` credentials.
