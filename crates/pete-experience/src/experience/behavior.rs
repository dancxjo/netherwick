#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecalledExperience {
    pub score: f32,
    pub experience: Experience,
    pub sensation: Sensation,
    #[serde(default)]
    pub original_frame_id: Option<Uuid>,
    #[serde(default)]
    pub original_vector_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperienceBehaviorInput {
    pub now: Now,
    pub sense_vectors: Vec<Vec<f32>>,
}

impl ExperienceBehaviorInput {
    pub fn from_now(now: &Now) -> Self {
        let encode_input = experience_encode_input_from_now(now);
        Self {
            now: now.clone(),
            sense_vectors: encode_input.sense_vectors,
        }
    }

    pub fn from_instant(now: &Now, instant: &ExperienceInstant) -> Self {
        let encode_input = ExperienceEncodeInput::from_instant(instant);
        Self {
            now: now.clone(),
            sense_vectors: encode_input.sense_vectors,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperienceBehaviorOutput {
    pub latent: ExperienceLatent,
    pub reconstruction: Option<ExperienceDecodeOutput>,
    pub reconstruction_loss: Option<f32>,
    pub confidence: f32,
}
