use core::fmt::{self, Write as _};

use heapless::String;

use crate::body;

#[derive(Clone, Copy)]
pub struct FirmwareBuildIdentity {
    pub firmware_version: &'static str,
    pub git_commit: &'static str,
    pub git_commit_short: &'static str,
    pub git_dirty: bool,
    pub build_timestamp: &'static str,
    pub build_profile: &'static str,
    pub build_target: &'static str,
    pub build_backend: &'static str,
    pub build_id: &'static str,
}

pub const CURRENT: FirmwareBuildIdentity = FirmwareBuildIdentity {
    firmware_version: env!("CARGO_PKG_VERSION"),
    git_commit: body::BUILD_GIT_COMMIT,
    git_commit_short: body::BUILD_GIT_COMMIT_SHORT,
    git_dirty: body::BUILD_GIT_DIRTY,
    build_timestamp: body::BUILD_TIMESTAMP,
    build_profile: body::BUILD_PROFILE,
    build_target: body::BUILD_TARGET,
    build_backend: body::BUILD_BACKEND,
    build_id: body::BUILD_ID,
};

pub fn write_json<const N: usize>(
    response: &mut String<N>,
    identity: FirmwareBuildIdentity,
) -> fmt::Result {
    write!(
        response,
        "\"firmware_version\":\"{}\",\"git_commit\":\"{}\",\"git_commit_short\":\"{}\",\"git_dirty\":{},\"build_timestamp\":\"{}\",\"build_profile\":\"{}\",\"build_target\":\"{}\",\"build_backend\":\"{}\",\"build_id\":\"{}\"",
        identity.firmware_version,
        identity.git_commit,
        identity.git_commit_short,
        identity.git_dirty,
        identity.build_timestamp,
        identity.build_profile,
        identity.build_target,
        identity.build_backend,
        identity.build_id,
    )
}

pub fn write_compact<const N: usize>(
    response: &mut String<N>,
    identity: FirmwareBuildIdentity,
) -> fmt::Result {
    write!(
        response,
        " firmware_version={} git_commit={} git_dirty={} build_id={}",
        identity.firmware_version, identity.git_commit, identity.git_dirty, identity.build_id
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(version: &str, short: &str, dirty: bool) -> String<64> {
        let mut value = String::new();
        write!(
            value,
            "{}+g{}{}",
            version,
            short,
            if dirty { ".dirty" } else { "" }
        )
        .unwrap();
        value
    }

    #[test]
    fn renders_clean_identity() {
        assert_eq!(render("0.1.7", "1a2b3c4d", false), "0.1.7+g1a2b3c4d");
    }

    #[test]
    fn renders_dirty_identity() {
        assert_eq!(render("0.1.7", "1a2b3c4d", true), "0.1.7+g1a2b3c4d.dirty");
    }

    #[test]
    fn renders_override_identity() {
        assert_eq!(render("0.1.7", "cioverride", false), "0.1.7+gcioverride");
    }

    #[test]
    fn renders_unavailable_identity() {
        assert_eq!(render("0.1.7", "unknown", false), "0.1.7+gunknown");
    }
}
