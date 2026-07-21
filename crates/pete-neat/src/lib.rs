//! A deliberately small NEAT nervous system for PETE locomotion.
//!
//! This crate owns policy proposals, never physical authority. Its output is
//! still expected to pass through `pete-autonomic`, the cockpit lease and the
//! brainstem safety/reflex layer.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use pete_behaviors::{FunctionBehavior, OutputDistance};
use pete_body::BodySense;
use pete_core::Pose2;
use pete_now::RangeSense;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

include!("neat/locomotion.rs");
include!("neat/genome.rs");
include!("neat/population.rs");
include!("neat/evaluation.rs");
include!("neat/selection.rs");
include!("neat/shadow.rs");
include!("neat/helpers.rs");

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
