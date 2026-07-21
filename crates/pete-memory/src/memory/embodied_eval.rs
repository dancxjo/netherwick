#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EmbodiedEvalOmission {
    PrimarySensations,
    Descendants,
    Vectors,
    Impressions,
    FusedExperience,
    SummaryImpression,
    Predictions,
    MemoryPersistence,
    MemoryLinks,
    Recall,
}

impl EmbodiedEvalOmission {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PrimarySensations => "primary-sensations",
            Self::Descendants => "descendants",
            Self::Vectors => "vectors",
            Self::Impressions => "impressions",
            Self::FusedExperience => "fused-experience",
            Self::SummaryImpression => "summary-impression",
            Self::Predictions => "predictions",
            Self::MemoryPersistence => "memory-persistence",
            Self::MemoryLinks => "memory-links",
            Self::Recall => "recall",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedPipelineCoverageReport {
    pub schema_version: u32,
    pub fixture: String,
    pub placeholder: bool,
    pub placeholder_vector_count: usize,
    pub frame_count: usize,
    pub instant_count: usize,
    pub instant_teacher_vector_count: usize,
    pub instant_missing_modality_count: usize,
    pub primary_sensation_count: usize,
    pub descendant_sensation_count: usize,
    pub vectorized_sensation_count: usize,
    pub impression_count: usize,
    pub summary_impression_count: usize,
    pub experience_latent_count: usize,
    pub prediction_count: usize,
    pub memory_link_count: usize,
    pub recall_sensation_count: usize,
    pub recall_impression_count: usize,
    pub place_recognition_candidate_count: usize,
    pub lineage_edge_count: usize,
    pub input_modalities: Vec<String>,
    #[serde(default)]
    pub instant_coverage: Vec<InstantCoverage>,
    pub vector_coverage: EmbodiedVectorCoverage,
    pub warnings: Vec<String>,
    pub failures: Vec<String>,
}

impl EmbodiedPipelineCoverageReport {
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }
}

pub async fn deterministic_embodied_eval_report() -> Result<EmbodiedPipelineCoverageReport> {
    deterministic_embodied_eval_report_with_omissions(&[]).await
}

pub async fn deterministic_embodied_eval_report_with_omissions(
    omissions: &[EmbodiedEvalOmission],
) -> Result<EmbodiedPipelineCoverageReport> {
    let store = InMemoryExperienceStore::new();
    let prior_now = deterministic_embodied_fixture_now(1_000, 0.0);
    let mut prior = build_embodied_eval_frame(prior_now, None, omissions).await?;
    if !omitted(omissions, EmbodiedEvalOmission::MemoryLinks) {
        attach_memory_links_to_frame(&mut prior);
    }
    if !omitted(omissions, EmbodiedEvalOmission::MemoryPersistence) {
        store.store(&prior).await?;
        store.observe_frame(&prior).await?;
    }

    let current_now = deterministic_embodied_fixture_now(1_750, 0.08);
    let recall = if omitted(omissions, EmbodiedEvalOmission::Recall)
        || omitted(omissions, EmbodiedEvalOmission::MemoryPersistence)
    {
        None
    } else {
        Some(store.recall(RecallQuery::from_now(&current_now)).await?)
    };
    let mut current = build_embodied_eval_frame(current_now, recall.as_ref(), omissions).await?;
    if !omitted(omissions, EmbodiedEvalOmission::MemoryLinks) {
        attach_memory_links_to_frame(&mut current);
    }
    if !omitted(omissions, EmbodiedEvalOmission::MemoryPersistence) {
        store.store(&current).await?;
        store.observe_frame(&current).await?;
    }

    let persisted_frame_count = store.snapshot().len();
    let mut frames = vec![prior, current];
    let mut report = coverage_report_from_frames("deterministic", &frames);
    report.place_recognition_candidate_count = recall
        .as_ref()
        .map(|recall| recall.place_recognition_candidates.len())
        .unwrap_or_default();
    report.frame_count = persisted_frame_count.max(frames.len());
    if omitted(omissions, EmbodiedEvalOmission::MemoryPersistence) {
        report.frame_count = persisted_frame_count;
    }
    report.warnings.extend(
        omissions
            .iter()
            .map(|stage| format!("omitted {}", stage.as_str())),
    );
    evaluate_required_embodied_coverage(&mut report);
    frames.clear();
    Ok(report)
}

pub fn deterministic_embodied_fixture_now(t_ms: u64, pose_offset_m: f32) -> Now {
    let mut body = BodySense {
        battery_level: 0.72,
        charging: false,
        flags: BodyFlags {
            wall: true,
            ..BodyFlags::default()
        },
        odometry: Pose2 {
            x_m: 1.25 + pose_offset_m,
            y_m: -0.35,
            heading_rad: 0.18,
        },
        velocity: Velocity {
            forward_m_s: 0.06,
            turn_rad_s: 0.01,
        },
        last_update_ms: t_ms,
        ..BodySense::default()
    };
    body.cliff_sensors.front_left = 0.08;

    let mut rgb = vec![9_u8; 12 * 8 * 3];
    for y in 2..6 {
        for x in 4..8 {
            let idx = (y * 12 + x) * 3;
            rgb[idx] = 210;
            rgb[idx + 1] = 160;
            rgb[idx + 2] = 80;
        }
    }

    let mut now = Now::blank(t_ms, body);
    now.eye_frame = Some(EyeFrame {
        captured_at_ms: t_ms.saturating_sub(12),
        width: 12,
        height: 8,
        format: EyeFrameFormat::Rgb8,
        bytes: rgb,
        source: Some("fixture.synthetic_camera".to_string()),
    });
    now.eye.scene_vectors.push(
        VectorArtifact::new(
            SCENE_VECTOR_COLLECTION,
            "fixture-scene",
            vec![1.0, 0.0, 0.25, 0.5],
        )
        .with_model("fixture.scene.vector.v1")
        .with_source_frame_id("fixture-frame")
        .with_occurred_at_ms(t_ms),
    );
    now.range = RangeSense {
        schema_version: 1,
        beams: vec![0.42, 0.55, 1.2, 0.9, 0.48],
        nearest_m: Some(0.42),
        ..RangeSense::default()
    };
    now.kinect = KinectSense {
        schema_version: 1,
        depth_m: vec![0.72, 0.74, 0.81, 0.92, 1.05, 1.1],
        depth_width: 3,
        depth_height: 2,
        min_depth_m: 0.72,
        max_depth_m: 1.1,
        depth_coordinate_system: Some("fixture-depth-camera".to_string()),
        skeletons: vec![KinectSkeletonSense {
            tracking_id: 7,
            lean_xy: [0.02, -0.01],
            joints: vec![KinectJointSense {
                joint_name: "head".to_string(),
                position_m: [0.4, 0.1, 1.2],
                tracking_confidence: 0.8,
                tracked: true,
            }],
        }],
        ..KinectSense::default()
    };
    now.ear = EarSense {
        schema_version: 1,
        features: vec![vec![0.1, 0.2, 0.15, 0.05]],
        transcript: Some("fixture voice says remember the charger alcove".to_string()),
        transcript_vectors: vec![pete_now::VectorArtifact::new(
            "transcripts",
            "fixture-asr-transcript",
            vec![0.21, 0.34, 0.55, 0.89],
        )
        .with_model("pete.text.hashing.v1")
        .with_source_id("fixture-asr")
        .with_occurred_at_ms(t_ms)],
        asr: AsrSense {
            transcript: Some("fixture voice says remember the charger alcove".to_string()),
            is_final: true,
            confidence: 0.91,
            start_ms: Some(t_ms.saturating_sub(360)),
            end_ms: Some(t_ms),
            duration_ms: Some(360),
            sample_rate_hz: Some(16_000),
            word_count: Some(7),
            speaker_confidence: Some(0.77),
            ..AsrSense::default()
        },
    };
    now.voice.vectors.push(
        VectorArtifact::new(
            VOICE_VECTOR_COLLECTION,
            "fixture-voice",
            vec![0.2, 0.4, 0.6, 0.8],
        )
        .with_model("fixture.voice.vector.v1")
        .with_source_id("speaker:fixture")
        .with_occurred_at_ms(t_ms),
    );
    now.face.vectors.push(
        VectorArtifact::new(
            FACE_VECTOR_COLLECTION,
            "fixture-face",
            vec![0.8, 0.6, 0.4, 0.2],
        )
        .with_model("fixture.face.vector.v1")
        .with_source_id("person:fixture")
        .with_source_frame_id("fixture-frame")
        .with_occurred_at_ms(t_ms),
    );
    now.objects.observations.push(ObjectObservation {
        label: "charger alcove".to_string(),
        class: ObjectClass::Charger,
        bearing_rad: 0.15,
        distance_m: Some(0.9),
        confidence: 0.82,
        source: ObjectObservationSource::Captioner,
    });
    now
}

async fn build_embodied_eval_frame(
    now: Now,
    recall: Option<&RecallBundle>,
    omissions: &[EmbodiedEvalOmission],
) -> Result<ExperienceFrame> {
    let pipeline = EmbodiedPipeline::new();
    let mut sensations = Vec::new();
    let mut impressions = Vec::new();

    if !omitted(omissions, EmbodiedEvalOmission::PrimarySensations) {
        for primary in pete_experience::primary_sensations_from_now(&now) {
            let batch = pipeline.ingest_primary(primary).await?;
            sensations.extend(batch.sensations);
            impressions.extend(batch.impressions);
        }
    }
    if omitted(omissions, EmbodiedEvalOmission::Descendants) {
        let retained_ids = sensations
            .iter()
            .filter(|sensation| sensation.parent_id.is_none())
            .map(|sensation| sensation.id)
            .collect::<BTreeSet<_>>();
        sensations.retain(|sensation| sensation.parent_id.is_none());
        impressions.retain(|impression| {
            impression
                .sensation_id
                .map(|id| retained_ids.contains(&id))
                .unwrap_or(false)
        });
    }
    if omitted(omissions, EmbodiedEvalOmission::Vectors) {
        for sensation in &mut sensations {
            sensation.vector = None;
            if let Some(impression) = &mut sensation.impression {
                impression.vector = None;
            }
        }
        for impression in &mut impressions {
            impression.vector = None;
        }
    }
    if omitted(omissions, EmbodiedEvalOmission::Impressions) {
        for sensation in &mut sensations {
            sensation.impression = None;
        }
        impressions.clear();
    }

    let mut experience = if omitted(omissions, EmbodiedEvalOmission::FusedExperience)
        || sensations.is_empty()
        || impressions.is_empty()
    {
        None
    } else {
        let mut fused = ExperienceFuser::new(750).fuse(&sensations, &impressions)?;
        if omitted(omissions, EmbodiedEvalOmission::SummaryImpression) {
            fused.summary_impression = None;
        }
        if omitted(omissions, EmbodiedEvalOmission::Predictions) {
            fused.predictions.clear();
        }
        Some(fused)
    };

    if let (Some(experience), Some(recall)) = (&mut experience, recall) {
        for recollection in &recall.recollections {
            sensations.push(recollection.sensation.clone());
            if let Some(impression) = recollection.sensation.impression.clone() {
                impressions.push(impression.clone());
                experience.impression_ids.push(impression.id);
            }
            experience.sensation_ids.push(recollection.sensation.id);
        }
    }

    let summary_impression = experience
        .as_ref()
        .and_then(|experience| experience.summary_impression.clone());
    if let Some(summary) = summary_impression {
        impressions.push(summary);
    }
    let latent = pete_experience::ExperienceLatent {
        t_ms: now.t_ms,
        z: vec![
            (now.t_ms as f32 / 1_000.0).sin(),
            now.body.battery_level,
            now.body.odometry.x_m,
            now.body.odometry.y_m,
        ],
        reconstruction_error: 0.0,
        prediction_error: 0.0,
        confidence: 0.5,
    };

    Ok(ExperienceFrame {
        id: uuid::Uuid::new_v4(),
        t_ms: now.t_ms,
        now,
        sensations,
        impressions,
        experiences: experience.into_iter().collect(),
        z: Some(latent),
        chosen_action: Some(ActionPrimitive::Inspect {
            target: pete_actions::InspectTarget::Novelty,
        }),
        conscious_command: None,
        reign_input: None,
        reign_outcome: None,
        predicted_futures: vec![FuturePrediction {
            offset_ms: 750,
            predicted_z: vec![0.1, 0.2, 0.3, 0.4],
            confidence: 0.31,
            summary: Some("fallback latent future remains near the charger alcove".to_string()),
        }],
        behavior_runs: Vec::new(),
        actual_next: None,
        reward: Reward::default(),
        surprise: SurpriseSense::default(),
        memory_recall: recall.map(|recall| recall.hits.clone()).unwrap_or_default(),
        recollections: recall
            .map(|recall| recall.recollections.clone())
            .unwrap_or_default(),
        llm_teaching: Vec::new(),
        counterfactuals: Vec::new(),
        notes: vec!["deterministic embodied eval fixture".to_string()],
    })
}

fn coverage_report_from_frames(
    fixture: impl Into<String>,
    frames: &[ExperienceFrame],
) -> EmbodiedPipelineCoverageReport {
    let mut report = EmbodiedPipelineCoverageReport {
        schema_version: 1,
        fixture: fixture.into(),
        frame_count: frames.len(),
        ..EmbodiedPipelineCoverageReport::default()
    };
    let mut modalities = BTreeSet::new();
    for frame in frames {
        let instant = frame.experience_instant();
        let instant_coverage = instant.coverage();
        report.instant_count += 1;
        report.instant_teacher_vector_count += instant.teacher_vectors.len();
        report.instant_missing_modality_count += instant.missing_modalities.len();
        report.primary_sensation_count += instant.primary_sensations.len();
        report.descendant_sensation_count += instant.descendant_sensations.len();
        report.vectorized_sensation_count += frame
            .sensations
            .iter()
            .filter(|sensation| sensation.vector.is_some())
            .count();
        report.placeholder_vector_count += frame
            .sensations
            .iter()
            .filter_map(|sensation| sensation.vector.as_ref())
            .filter(|vector| vector.is_fallback)
            .count()
            + frame
                .impressions
                .iter()
                .filter_map(|impression| impression.vector.as_ref())
                .filter(|vector| vector.is_fallback)
                .count();
        report.impression_count += frame
            .impressions
            .iter()
            .filter(|impression| impression.sensation_id.is_some() || !impression.about.is_empty())
            .count();
        report.summary_impression_count += frame
            .experiences
            .iter()
            .filter(|experience| experience.summary_impression.is_some())
            .count();
        report.experience_latent_count += usize::from(frame.z.is_some());
        report.prediction_count += instant.predictions.len();
        report.memory_link_count += instant.memory_links.len();
        report.recall_sensation_count += frame
            .sensations
            .iter()
            .filter(|sensation| {
                sensation.modality == Modality::Memory
                    && sensation.payload_kind == SensationPayloadKind::MemoryRecall
            })
            .count();
        report.recall_impression_count += frame
            .impressions
            .iter()
            .filter(|impression| impression.kind == "memory.recall.impression")
            .count();
        report.lineage_edge_count += instant.lineage.len();
        modalities.extend(instant_coverage.present_modalities.iter().cloned());
        report.instant_coverage.push(instant_coverage);
        let coverage = EmbodiedVectorCoverage::from_parts(
            &frame.sensations,
            &frame.impressions,
            frame.experiences.last(),
        );
        merge_vector_coverage(&mut report.vector_coverage, coverage);
    }
    report.input_modalities = modalities.into_iter().collect();
    report.placeholder = report.placeholder_vector_count > 0;
    report
}

fn merge_vector_coverage(target: &mut EmbodiedVectorCoverage, incoming: EmbodiedVectorCoverage) {
    target.image += incoming.image;
    target.face += incoming.face;
    target.voice += incoming.voice;
    target.transcript += incoming.transcript;
    target.impression += incoming.impression;
    target.experience += incoming.experience;
    target.fallback_count += incoming.fallback_count;
}

fn evaluate_required_embodied_coverage(report: &mut EmbodiedPipelineCoverageReport) {
    required_stage(report.instant_count, "no instants", &mut report.failures);
    required_stage(
        report.instant_teacher_vector_count,
        "no instant teacher vectors",
        &mut report.failures,
    );
    required_stage(
        report.primary_sensation_count,
        "no primary sensations",
        &mut report.failures,
    );
    required_stage(
        report.descendant_sensation_count,
        "no descendants",
        &mut report.failures,
    );
    required_stage(
        report.vectorized_sensation_count,
        "no vectors",
        &mut report.failures,
    );
    required_stage(
        report.impression_count,
        "no impressions",
        &mut report.failures,
    );
    required_stage(
        report.experience_latent_count,
        "no learned experience latent",
        &mut report.failures,
    );
    required_stage(
        report.summary_impression_count,
        "no summary impression",
        &mut report.failures,
    );
    required_stage(
        report.prediction_count,
        "no prediction",
        &mut report.failures,
    );
    required_stage(
        report.memory_link_count,
        "no memory persistence/link",
        &mut report.failures,
    );
    required_stage(
        report.frame_count,
        "no memory persistence/link",
        &mut report.failures,
    );
    required_stage(
        report
            .recall_sensation_count
            .min(report.recall_impression_count),
        "no recall",
        &mut report.failures,
    );
    required_stage(
        report.place_recognition_candidate_count,
        "no place recognition",
        &mut report.failures,
    );
    required_stage(
        report.lineage_edge_count,
        "no lineage",
        &mut report.warnings,
    );
}

fn required_stage(count: usize, message: &str, messages: &mut Vec<String>) {
    if count == 0 && !messages.iter().any(|existing| existing == message) {
        messages.push(message.to_string());
    }
}

fn omitted(omissions: &[EmbodiedEvalOmission], stage: EmbodiedEvalOmission) -> bool {
    omissions.iter().any(|candidate| *candidate == stage)
}

fn danger_signal(now: &Now) -> f32 {
    let body = &now.body;
    let bumper = body.flags.bump_left || body.flags.bump_right;
    let cliff = body.flags.cliff_left
        || body.flags.cliff_front_left
        || body.flags.cliff_front_right
        || body.flags.cliff_right;
    let cliff_sensor = body.cliff_sensors.max();
    let nearest = now.range.nearest_m.unwrap_or(10.0);
    let range_risk = (1.0 - nearest / 0.7).clamp(0.0, 1.0);
    [
        if bumper { 1.0 } else { 0.0 },
        if body.flags.wall { 0.85 } else { 0.0 },
        if cliff {
            1.0
        } else {
            cliff_sensor.clamp(0.0, 1.0)
        },
        range_risk,
    ]
    .into_iter()
    .fold(0.0, f32::max)
}

fn charge_signal(now: &Now) -> f32 {
    let sim_score = now
        .extensions
        .get("sim.world")
        .and_then(|value| value.get("values"))
        .and_then(|value| value.as_array())
        .and_then(|values| {
            let near = values
                .get(3)
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0) as f32;
            let visible = values
                .get(4)
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0) as f32;
            Some(near.max(visible))
        })
        .unwrap_or(0.0);
    [if now.body.charging { 1.0 } else { 0.0 }, sim_score]
        .into_iter()
        .fold(0.0, f32::max)
}

fn social_signal(now: &Now) -> f32 {
    let visual = !now.face.vectors.is_empty() as u8 as f32;
    let voice = !now.voice.vectors.is_empty() as u8 as f32;
    let skeleton = (!now.kinect.skeletons.is_empty()) as u8 as f32;
    let transcript = now
        .ear
        .transcript
        .as_ref()
        .map(|text| (!text.trim().is_empty()) as u8 as f32)
        .unwrap_or(0.0);
    visual.max(voice).max(skeleton).max(transcript)
}

fn observed_object_summary(now: &Now) -> Vec<String> {
    let mut objects = now
        .objects
        .observations
        .iter()
        .filter(|observation| observation.confidence >= 0.3)
        .map(|observation| observation.label.clone())
        .collect::<Vec<_>>();
    if danger_signal(now) >= 0.5 {
        push_unique_object(&mut objects, "danger");
    }
    if charge_signal(now) >= 0.5 {
        push_unique_object(&mut objects, "charger");
    }
    if social_signal(now) >= 0.5 {
        push_unique_object(&mut objects, "person_or_speaker");
    }
    objects
}

fn push_unique_object(objects: &mut Vec<String>, value: &str) {
    if !objects.iter().any(|object| object == value) {
        objects.push(value.to_string());
    }
}

fn scene_vectors_with_frame_id(
    artifacts: &[VectorArtifact],
    frame_id: Option<&str>,
) -> Vec<VectorArtifact> {
    let Some(frame_id) = frame_id else {
        return artifacts.to_vec();
    };
    artifacts
        .iter()
        .cloned()
        .map(|mut artifact| {
            if artifact.source_frame_id.is_none() {
                artifact.source_frame_id = Some(frame_id.to_string());
            }
            artifact
        })
        .collect()
}

fn merge_vector_ids(target: &mut Vec<String>, artifacts: &[VectorArtifact]) {
    for artifact in artifacts {
        if artifact.point_id.trim().is_empty() {
            continue;
        }
        if !target.iter().any(|existing| existing == &artifact.point_id) {
            target.push(artifact.point_id.clone());
        }
    }
    const MAX_ASSOCIATED_VECTORS: usize = 12;
    if target.len() > MAX_ASSOCIATED_VECTORS {
        target.drain(0..target.len() - MAX_ASSOCIATED_VECTORS);
    }
}

fn update_action_outcome(
    outcomes: &mut Vec<ActionOutcomeSummary>,
    action: &ActionPrimitive,
    reward: f32,
    t_ms: u64,
) {
    if let Some(existing) = outcomes
        .iter_mut()
        .find(|candidate| candidate.action == *action)
    {
        let prior_total = existing.mean_reward * existing.count as f32;
        existing.count = existing.count.saturating_add(1);
        existing.mean_reward = (prior_total + reward) / existing.count.max(1) as f32;
        existing.last_seen_tick = t_ms;
    } else {
        outcomes.push(ActionOutcomeSummary {
            action: action.clone(),
            count: 1,
            mean_reward: reward,
            last_seen_tick: t_ms,
        });
    }
    outcomes.sort_by(|left, right| {
        right
            .mean_reward
            .abs()
            .partial_cmp(&left.mean_reward.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| right.last_seen_tick.cmp(&left.last_seen_tick))
    });
    const MAX_ACTION_OUTCOMES: usize = 8;
    outcomes.truncate(MAX_ACTION_OUTCOMES);
}

