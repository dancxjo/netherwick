# netherwick

`netherwick` is a Rust workspace for Pete Netherwick, an embodied self-training robot architecture.

Pete gathers raw sensors, memory recalls, internal drives, model predictions, surprise, and LLM guidance into `Now`. `Now` is compressed into an `ExperienceLatent`, used to imagine futures, choose actions, and train from consequences.

Pete acts through high-level action primitives. A hard-coded autonomic layer keeps the body safe. The LLM may command, reflect, critique, and teach, but cannot bypass safety.

Every hard-coded behavior is replaceable. It can run directly, shadow-train a model, compare with a model, promote a model, or fall back to safe hand-written logic.

Pete Netherwick is an embodied predictive organism: a robot body with reflexes, an experience ledger, compact learned present, imagined futures, memory returning as sensation, swappable learned behaviors, and an LLM consciousness that commands and teaches while safety protects the body.
