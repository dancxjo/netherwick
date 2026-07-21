#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct VisionFixtureManifest {
    schema_version: u32,
    frames: Vec<VisionFixtureFrame>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct VisionFixtureFrame {
    id: String,
    path: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    sequence: Option<String>,
    #[serde(default)]
    timestamp_ms: u64,
    #[serde(default)]
    annotations: Vec<VisionFixtureAnnotation>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct VisionFixtureAnnotation {
    label: String,
    bbox: pete_now::VisionBoundingBox,
    #[serde(default)]
    track_id: Option<String>,
}

#[derive(Clone, Debug)]
struct VisionEvalFrame {
    id: String,
    tags: Vec<String>,
    sequence: Option<String>,
    annotations: Vec<VisionFixtureAnnotation>,
    frame: EyeFrame,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct VisionEvalDetection {
    frame_id: String,
    bbox: pete_now::VisionBoundingBox,
    labels: Vec<pete_now::VisionLabelHypothesis>,
    track_id: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct VisionEvalMetrics {
    frames: usize,
    tag_counts: BTreeMap<String, usize>,
    annotated_objects: usize,
    detections: usize,
    true_positives: usize,
    false_positives: usize,
    false_negatives: usize,
    label_precision: Option<f32>,
    label_recall: Option<f32>,
    track_fragmentations: usize,
    duplicate_tracks: usize,
    inference_p50_ms: Option<u64>,
    inference_p95_ms: Option<u64>,
    inference_p50_us: Option<u64>,
    inference_p95_us: Option<u64>,
    throughput_fps: f32,
    failures: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct VisionBackendEvaluation {
    identity: pete_now::VisionModelIdentity,
    state: pete_now::VisionBackendState,
    resource_profile: pete_sensors::VisionPipelineConfig,
    detections: Vec<VisionEvalDetection>,
    metrics: VisionEvalMetrics,
    failure_reasons: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct VisionComparison {
    candidate_minus_baseline_detections: i64,
    candidate_minus_baseline_true_positives: i64,
    candidate_minus_baseline_fragmentations: i64,
    candidate_minus_baseline_p95_ms: Option<i64>,
    candidate_minus_baseline_p95_us: Option<i64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct VisionEvalReport {
    schema_version: u32,
    source: String,
    frame_ids: Vec<String>,
    candidate: VisionBackendEvaluation,
    baseline: VisionBackendEvaluation,
    comparison: VisionComparison,
}

async fn run_vision_eval(args: VisionEvalArgs) -> Result<()> {
    let (source, frames) = if let Some(capture) = args.capture.as_deref() {
        (
            format!("capture:{capture}"),
            load_vision_capture_frames(capture).await?,
        )
    } else {
        (
            format!("fixtures:{}", args.fixtures),
            load_vision_fixture_frames(Path::new(&args.fixtures))?,
        )
    };
    if frames.is_empty() {
        anyhow::bail!("vision evaluation source contains no RGB frames");
    }
    let config = pete_sensors::VisionPipelineConfig::raspberry_pi_5();
    let candidate = evaluate_vision_backend(
        vision_backend_from_name(&args.candidate)?,
        config.clone(),
        &frames,
    );
    let baseline =
        evaluate_vision_backend(vision_backend_from_name(&args.baseline)?, config, &frames);
    let report = VisionEvalReport {
        schema_version: 1,
        source,
        frame_ids: frames.iter().map(|frame| frame.id.clone()).collect(),
        comparison: VisionComparison {
            candidate_minus_baseline_detections: candidate.metrics.detections as i64
                - baseline.metrics.detections as i64,
            candidate_minus_baseline_true_positives: candidate.metrics.true_positives as i64
                - baseline.metrics.true_positives as i64,
            candidate_minus_baseline_fragmentations: candidate.metrics.track_fragmentations as i64
                - baseline.metrics.track_fragmentations as i64,
            candidate_minus_baseline_p95_ms: candidate
                .metrics
                .inference_p95_ms
                .zip(baseline.metrics.inference_p95_ms)
                .map(|(candidate, baseline)| candidate as i64 - baseline as i64),
            candidate_minus_baseline_p95_us: candidate
                .metrics
                .inference_p95_us
                .zip(baseline.metrics.inference_p95_us)
                .map(|(candidate, baseline)| candidate as i64 - baseline as i64),
        },
        candidate,
        baseline,
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(out) = args.out {
        std::fs::write(&out, format!("{json}\n"))
            .with_context(|| format!("writing vision evaluation report {out}"))?;
    }
    println!("{json}");
    Ok(())
}

fn vision_backend_from_name(name: &str) -> Result<std::sync::Arc<dyn pete_sensors::VisionBackend>> {
    match name {
        "classical" | "saliency" => Ok(std::sync::Arc::new(pete_sensors::ClassicalSaliencyBackend)),
        "unavailable" | "none" | "missing" => Ok(std::sync::Arc::new(
            pete_sensors::UnavailableVisionBackend::new("model/backend not installed"),
        )),
        other => anyhow::bail!(
            "unknown vision backend {other}; supported backends: classical, unavailable"
        ),
    }
}

fn load_vision_fixture_frames(path: &Path) -> Result<Vec<VisionEvalFrame>> {
    let manifest: VisionFixtureManifest = serde_json::from_slice(
        &std::fs::read(path).with_context(|| format!("reading {}", path.display()))?,
    )
    .with_context(|| format!("parsing {}", path.display()))?;
    if manifest.schema_version != 1 {
        anyhow::bail!(
            "unsupported vision fixture schema {}; expected 1",
            manifest.schema_version
        );
    }
    let root = path.parent().unwrap_or_else(|| Path::new("."));
    manifest
        .frames
        .into_iter()
        .map(|fixture| {
            let image_path = root.join(&fixture.path);
            let image = image::open(&image_path)
                .with_context(|| format!("decoding fixture {}", image_path.display()))?
                .into_rgb8();
            let (width, height) = image.dimensions();
            Ok(VisionEvalFrame {
                id: fixture.id,
                tags: fixture.tags,
                sequence: fixture.sequence,
                annotations: fixture.annotations,
                frame: EyeFrame {
                    captured_at_ms: fixture.timestamp_ms,
                    rgbd_frame_id: None,
                    device_timestamp_ms: None,
                    width,
                    height,
                    format: EyeFrameFormat::Rgb8,
                    bytes: image.into_raw(),
                    source: Some("vision_fixture".to_string()),
                },
            })
        })
        .collect()
}

async fn load_vision_capture_frames(path: &str) -> Result<Vec<VisionEvalFrame>> {
    let reader = CaptureReader::open(path).await?;
    Ok(reader
        .read_frames()
        .await?
        .into_iter()
        .filter_map(|record| {
            let frame = record
                .snapshot
                .kinect
                .color_frame
                .clone()
                .or(record.snapshot.eye_frame.clone())?;
            Some(VisionEvalFrame {
                id: format!("capture-frame-{}", record.index),
                tags: vec!["capture".to_string()],
                sequence: Some("capture".to_string()),
                annotations: Vec::new(),
                frame,
            })
        })
        .collect())
}

#[derive(Clone, Debug, Default)]
struct EvalTrack {
    id: String,
    bbox: pete_now::VisionBoundingBox,
    sequence: Option<String>,
}

fn evaluate_vision_backend(
    backend: std::sync::Arc<dyn pete_sensors::VisionBackend>,
    config: pete_sensors::VisionPipelineConfig,
    frames: &[VisionEvalFrame],
) -> VisionBackendEvaluation {
    let started = Instant::now();
    let mut inference_us = Vec::new();
    let mut detections = Vec::new();
    let mut failures = Vec::new();
    let mut tracks = Vec::<EvalTrack>::new();
    let mut next_track_id = 0_u64;
    let mut metrics = VisionEvalMetrics {
        frames: frames.len(),
        tag_counts: frames.iter().flat_map(|frame| frame.tags.iter()).fold(
            BTreeMap::new(),
            |mut counts, tag| {
                *counts.entry(tag.clone()).or_default() += 1;
                counts
            },
        ),
        annotated_objects: frames.iter().map(|frame| frame.annotations.len()).sum(),
        ..VisionEvalMetrics::default()
    };
    let mut annotation_tracks = BTreeMap::<String, BTreeSet<String>>::new();

    for frame in frames {
        let inference_started = Instant::now();
        let proposals = backend
            .preprocess(&frame.frame, &config)
            .and_then(|prepared| backend.detect(&prepared, config.maximum_detections));
        inference_us.push(inference_started.elapsed().as_micros() as u64);
        let proposals = match proposals {
            Ok(proposals) => proposals,
            Err(error) => {
                failures.push(format!("{}: {error}", frame.id));
                metrics.false_negatives += frame.annotations.len();
                continue;
            }
        };
        if tracks
            .first()
            .is_some_and(|track| track.sequence != frame.sequence)
        {
            tracks.clear();
        }
        let mut used_tracks = BTreeSet::new();
        let mut frame_detections = Vec::new();
        for proposal in proposals {
            let matched = tracks
                .iter()
                .enumerate()
                .filter(|(index, _)| !used_tracks.contains(index))
                .map(|(index, track)| (index, vision_bbox_iou(proposal.bbox, track.bbox)))
                .filter(|(_, iou)| *iou >= config.track_iou_threshold)
                .max_by(|left, right| {
                    left.1
                        .partial_cmp(&right.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            let track_id = if let Some((index, _)) = matched {
                used_tracks.insert(index);
                tracks[index].bbox = proposal.bbox;
                tracks[index].id.clone()
            } else {
                next_track_id += 1;
                let id = format!("eval-track-{next_track_id}");
                tracks.push(EvalTrack {
                    id: id.clone(),
                    bbox: proposal.bbox,
                    sequence: frame.sequence.clone(),
                });
                id
            };
            frame_detections.push(VisionEvalDetection {
                frame_id: frame.id.clone(),
                bbox: proposal.bbox,
                labels: proposal.labels,
                track_id,
            });
        }
        score_vision_frame(
            frame,
            &frame_detections,
            &mut metrics,
            &mut annotation_tracks,
        );
        detections.extend(frame_detections);
    }
    metrics.detections = detections.len();
    metrics.failures = failures.len();
    metrics.label_precision = (metrics.true_positives + metrics.false_positives > 0).then(|| {
        metrics.true_positives as f32 / (metrics.true_positives + metrics.false_positives) as f32
    });
    metrics.label_recall = (metrics.annotated_objects > 0)
        .then(|| metrics.true_positives as f32 / metrics.annotated_objects as f32);
    metrics.track_fragmentations = annotation_tracks
        .values()
        .map(|tracks| tracks.len().saturating_sub(1))
        .sum();
    inference_us.sort_unstable();
    metrics.inference_p50_us = vision_percentile(&inference_us, 0.5);
    metrics.inference_p95_us = vision_percentile(&inference_us, 0.95);
    metrics.inference_p50_ms = metrics.inference_p50_us.map(|value| value / 1_000);
    metrics.inference_p95_ms = metrics.inference_p95_us.map(|value| value / 1_000);
    metrics.throughput_fps = if started.elapsed().as_secs_f32() > 0.0 {
        frames.len() as f32 / started.elapsed().as_secs_f32()
    } else {
        0.0
    };
    VisionBackendEvaluation {
        identity: backend.identity(),
        state: backend.state(),
        resource_profile: config,
        detections,
        metrics,
        failure_reasons: failures,
    }
}

fn score_vision_frame(
    frame: &VisionEvalFrame,
    detections: &[VisionEvalDetection],
    metrics: &mut VisionEvalMetrics,
    annotation_tracks: &mut BTreeMap<String, BTreeSet<String>>,
) {
    let mut matched_detections = BTreeSet::new();
    for annotation in &frame.annotations {
        let mut matches = detections
            .iter()
            .enumerate()
            .filter(|(_, detection)| {
                vision_bbox_iou(annotation.bbox, detection.bbox) >= 0.3
                    && detection
                        .labels
                        .iter()
                        .any(|label| label.label == annotation.label)
            })
            .collect::<Vec<_>>();
        matches.sort_by(|left, right| {
            vision_bbox_iou(annotation.bbox, right.1.bbox)
                .partial_cmp(&vision_bbox_iou(annotation.bbox, left.1.bbox))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some((index, detection)) = matches.first() {
            metrics.true_positives += 1;
            matched_detections.insert(*index);
            if matches.len() > 1 {
                metrics.duplicate_tracks += matches.len() - 1;
            }
            if let Some(track_id) = annotation.track_id.as_ref() {
                annotation_tracks
                    .entry(track_id.clone())
                    .or_default()
                    .insert(detection.track_id.clone());
            }
        } else {
            metrics.false_negatives += 1;
        }
    }
    metrics.false_positives += detections.len().saturating_sub(matched_detections.len());
}

fn vision_bbox_iou(left: pete_now::VisionBoundingBox, right: pete_now::VisionBoundingBox) -> f32 {
    let x0 = left.x.max(right.x);
    let y0 = left.y.max(right.y);
    let x1 = left
        .x
        .saturating_add(left.width)
        .min(right.x.saturating_add(right.width));
    let y1 = left
        .y
        .saturating_add(left.height)
        .min(right.y.saturating_add(right.height));
    let intersection = x1.saturating_sub(x0) as u64 * y1.saturating_sub(y0) as u64;
    let union = left.width as u64 * left.height as u64 + right.width as u64 * right.height as u64
        - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

fn vision_percentile(samples: &[u64], fraction: f32) -> Option<u64> {
    (!samples.is_empty()).then(|| {
        let index = ((samples.len() - 1) as f32 * fraction).round() as usize;
        samples[index]
    })
}
