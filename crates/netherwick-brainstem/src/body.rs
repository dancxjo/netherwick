#![allow(dead_code)]

use crate::commands::CreateOiMode;

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum BodyKind {
    CreateOpenInterface,
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum DriveKind {
    Differential,
}

include!(concat!(env!("OUT_DIR"), "/body_config.rs"));
