use std::collections::BTreeMap;

use anyhow::{anyhow, Result};
use pete_actions::{
    ActionPrimitive, ApproachTarget, ExploreStyle, InspectTarget, ReignMode, TurnDir,
};
use pete_now::{
    ClockDomain, DriveSelfSummary, DriveSense, EntityId, EpistemicActionKind, EpistemicAffordance,
    EpistemicAttempt, EpistemicQuestionFamily, EvidenceRef, Freshness, GoalStatusBelief,
    PendingTemporalExpectation, QuestionId, SemanticBehaviorId, SemanticConceptId,
    SemanticExplanation, SemanticNodeRef, SemanticPredicate, SemanticRelationId,
    SocialAcknowledgmentKind, TimeInterval, WorldEntity, WorldEntityKind, WorldModelSnapshot,
    WorldModelUpdateContext,
};
use serde::{Deserialize, Serialize};

// Goal architecture domains share this namespace to preserve its public API.
include!("goal/types.rs");
include!("goal/module.rs");
include!("goal/evaluation.rs");
include!("goal/arbiter.rs");
include!("goal/system.rs");

#[cfg(test)]
#[path = "goal_architecture_tests.rs"]
mod tests;
