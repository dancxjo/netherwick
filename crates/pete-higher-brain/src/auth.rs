use crate::{atomic_write_json, sha256_bytes};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    Discover,
    TransferExperience,
    SubmitJob,
    CancelJob,
    ReturnCandidate,
    StageCandidate,
    ActivateCandidate,
    RollbackModel,
    ProvisionSoftware,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Principal {
    pub id: String,
    pub scopes: BTreeSet<Scope>,
}

impl Principal {
    pub fn require(&self, scope: Scope) -> Result<()> {
        if !self.scopes.contains(&scope) {
            anyhow::bail!("principal {:?} lacks {:?} authority", self.id, scope);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AuthorizationFile {
    pub schema_version: u32,
    /// SHA-256 token fingerprints. Plain tokens are never persisted here.
    pub tokens: BTreeMap<String, AuthorizedToken>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthorizedToken {
    pub principal_id: String,
    pub scopes: BTreeSet<Scope>,
    #[serde(default)]
    pub disabled: bool,
}

impl AuthorizationFile {
    pub fn authorize(&self, token: &str) -> Result<Principal> {
        let fingerprint = sha256_bytes(token.as_bytes());
        let entry = self
            .tokens
            .get(&fingerprint)
            .filter(|entry| !entry.disabled)
            .ok_or_else(|| anyhow::anyhow!("unknown or disabled higher-brain credential"))?;
        Ok(Principal {
            id: entry.principal_id.clone(),
            scopes: entry.scopes.clone(),
        })
    }

    pub fn enroll_token(
        &mut self,
        token: &str,
        principal_id: impl Into<String>,
        scopes: impl IntoIterator<Item = Scope>,
    ) -> String {
        let fingerprint = sha256_bytes(token.as_bytes());
        self.tokens.insert(
            fingerprint.clone(),
            AuthorizedToken {
                principal_id: principal_id.into(),
                scopes: scopes.into_iter().collect(),
                disabled: false,
            },
        );
        fingerprint
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        atomic_write_json(path, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scopes_are_independent() {
        let mut auth = AuthorizationFile {
            schema_version: 1,
            ..Default::default()
        };
        auth.enroll_token("secret", "trainer", [Scope::SubmitJob]);
        let principal = auth.authorize("secret").unwrap();
        assert!(principal.require(Scope::SubmitJob).is_ok());
        assert!(principal.require(Scope::ActivateCandidate).is_err());
        assert!(!serde_json::to_string(&auth).unwrap().contains("secret"));
    }
}
