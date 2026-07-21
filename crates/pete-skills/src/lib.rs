//! Runtime-loaded, sandboxed motherbrain skills.
//!
//! Lua owns semantic sequencing. Rust owns bodily resources, bounded command
//! renewal, authority, numerical controllers, and every physical safety check.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::future::Future;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll, Wake, Waker};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use mlua::{
    AsyncThread, DebugEvent, Error as LuaError, Function, HookTriggers, Lua, LuaOptions,
    LuaSerdeExt, MultiValue, StdLib, Table, UserData, UserDataFields, Value as LuaValue, Variadic,
    VmState,
};
use pete_cockpit::{CockpitEventKind, EventBatch, SafetyLatchKind};
use pete_conductor::{SkillId, SkillOutcome, SkillPhase, SkillRequest, SkillStatus};
use pete_now::{Now, ObjectClass};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

include!("skills/types.rs");
include!("skills/runtime.rs");
include!("skills/loading.rs");
include!("skills/api.rs");
include!("skills/operations.rs");
include!("skills/values.rs");

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
