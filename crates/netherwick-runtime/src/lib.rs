pub const CORE_LOOP: &[&str] = &[
    "Now_t",
    "encode to latent z_t",
    "predict futures",
    "choose action",
    "safety-filter",
    "act",
    "observe Now_t+1",
    "compute surprise/reward",
    "write ledger",
    "train",
];

pub const CADENCES: &[(&str, &str)] = &[
    ("fast", "sensors/safety/motors"),
    ("medium", "Now/prediction/conductor"),
    ("slow", "LLM/reflection/memory summaries"),
    ("idle", "replay/dream/model comparison"),
];
