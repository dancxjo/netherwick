pub trait FaceDetector: Send + Sync {
    fn detect_faces(&self, frame: &EyeFrame) -> Result<Vec<FaceDetection>>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct FaceDetection {
    pub face_id: String,
    pub source_frame_id: Option<String>,
    pub embedding: Vec<f32>,
    pub model: String,
}

pub trait ObjectDetector: Send + Sync {
    fn detect_objects(&self, frame: &EyeFrame) -> Result<Vec<ObjectDetection>>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObjectDetection {
    pub object_id: String,
    pub label: String,
    pub class: ObjectClass,
    pub bearing_rad: f32,
    pub distance_m: Option<f32>,
    pub confidence: f32,
    pub source: ObjectObservationSource,
    pub source_frame_id: Option<String>,
    pub embedding: Vec<f32>,
    pub model: String,
}

#[cfg(feature = "face")]
pub struct FaceIdDetector {
    analyzer: Arc<Mutex<face_id::analyzer::FaceAnalyzer>>,
}

#[cfg(feature = "face")]
impl FaceIdDetector {
    pub async fn from_hf() -> Result<Self> {
        let analyzer = face_id::analyzer::FaceAnalyzer::from_hf()
            .build()
            .await
            .context("failed to initialize face_id analyzer")?;
        Ok(Self {
            analyzer: Arc::new(Mutex::new(analyzer)),
        })
    }
}

#[cfg(feature = "face")]
impl FaceDetector for FaceIdDetector {
    fn detect_faces(&self, frame: &EyeFrame) -> Result<Vec<FaceDetection>> {
        let image = dynamic_image_from_eye_frame(frame)?;
        let faces = self
            .analyzer
            .lock()
            .map_err(|_| anyhow::anyhow!("face analyzer lock poisoned"))?
            .analyze(&image)
            .context("face_id analysis failed")?;
        Ok(faces
            .into_iter()
            .enumerate()
            .map(|(index, face)| FaceDetection {
                face_id: face_detection_id(frame, index, &face.embedding),
                source_frame_id: None,
                embedding: face.embedding,
                model: "face_id.0.4.1".to_string(),
            })
            .collect())
    }
}

fn process_eye_frame(
    t_ms: TimeMs,
    frame: &EyeFrame,
    face_detector: Option<&dyn FaceDetector>,
    object_detector: Option<&dyn ObjectDetector>,
) -> ProcessedFrame {
    let source_frame_id = format!(
        "eye-{}-{}x{}-{}",
        frame.captured_at_ms,
        frame.width,
        frame.height,
        frame.bytes.len()
    );
    let signal = bytes_to_unit_signal(&frame.bytes);
    let mut eye = EyeSense {
        schema_version: 1,
        frames: vec![signal.clone()],
        ..EyeSense::default()
    };
    eye.image_vectors.push(
        VectorArtifact::new(
            IMAGE_VECTOR_COLLECTION,
            source_frame_id.clone(),
            signal.clone(),
        )
        .with_model("raw-byte-unit-signal-v0")
        .with_source_frame_id(source_frame_id.clone())
        .with_occurred_at_ms(t_ms),
    );
    eye.image_description_vectors.push(
        VectorArtifact::new(
            IMAGE_DESCRIPTION_VECTOR_COLLECTION,
            format!("{source_frame_id}-summary"),
            frame_summary_vector(frame, &signal),
        )
        .with_model("frame-summary-v0")
        .with_source_frame_id(source_frame_id.clone())
        .with_occurred_at_ms(t_ms),
    );
    eye.scene_vectors.push(
        VectorArtifact::new(
            SCENE_VECTOR_COLLECTION,
            format!("{source_frame_id}-scene"),
            frame_summary_vector(frame, &signal),
        )
        .with_model("scene-summary-v0")
        .with_source_frame_id(source_frame_id.clone())
        .with_occurred_at_ms(t_ms),
    );

    let face = match face_detector {
        Some(detector) => detected_face_sense(t_ms, frame, &source_frame_id, detector),
        None => Ok(FaceSense {
            schema_version: 1,
            ..FaceSense::default()
        }),
    }
    .unwrap_or_else(|_| FaceSense {
        schema_version: 1,
        ..FaceSense::default()
    });

    let objects = match object_detector {
        Some(detector) => detected_object_sense(t_ms, frame, &source_frame_id, detector),
        None => Ok(ObjectSense {
            schema_version: 1,
            ..ObjectSense::default()
        }),
    }
    .unwrap_or_else(|_| ObjectSense {
        schema_version: 1,
        ..ObjectSense::default()
    });

    ProcessedFrame {
        eye,
        face,
        objects,
        summary: format!(
            "{:?} frame {}x{}, {} bytes",
            frame.format,
            frame.width,
            frame.height,
            frame.bytes.len()
        ),
        source_frame_id,
    }
}

fn detected_face_sense(
    t_ms: TimeMs,
    frame: &EyeFrame,
    source_frame_id: &str,
    detector: &dyn FaceDetector,
) -> Result<FaceSense> {
    let detections = detector.detect_faces(frame)?;
    let mut face = FaceSense {
        schema_version: 1,
        ..FaceSense::default()
    };
    for detection in detections {
        if detection.embedding.is_empty() {
            continue;
        }
        face.vectors.push(
            VectorArtifact::new(
                FACE_VECTOR_COLLECTION,
                detection.face_id,
                detection.embedding,
            )
            .with_model(detection.model)
            .with_source_frame_id(
                detection
                    .source_frame_id
                    .unwrap_or_else(|| source_frame_id.to_string()),
            )
            .with_occurred_at_ms(t_ms),
        );
    }
    Ok(face)
}

fn detected_object_sense(
    t_ms: TimeMs,
    frame: &EyeFrame,
    source_frame_id: &str,
    detector: &dyn ObjectDetector,
) -> Result<ObjectSense> {
    let detections = detector.detect_objects(frame)?;
    let mut objects = ObjectSense {
        schema_version: 1,
        ..ObjectSense::default()
    };
    for detection in detections {
        let source_frame_id = detection
            .source_frame_id
            .clone()
            .unwrap_or_else(|| source_frame_id.to_string());
        objects.observations.push(ObjectObservation {
            label: detection.label,
            class: detection.class,
            bearing_rad: detection.bearing_rad,
            distance_m: detection.distance_m,
            confidence: detection.confidence,
            source: detection.source,
        });
        if detection.embedding.is_empty() {
            continue;
        }
        objects.vectors.push(
            VectorArtifact::new(
                OBJECT_VECTOR_COLLECTION,
                detection.object_id,
                detection.embedding,
            )
            .with_model(detection.model)
            .with_source_frame_id(source_frame_id)
            .with_occurred_at_ms(t_ms),
        );
    }
    Ok(objects)
}

fn face_detection_id(frame: &EyeFrame, index: usize, embedding: &[f32]) -> String {
    let mut hash = stable_hash64(&frame.captured_at_ms.to_le_bytes());
    hash ^= stable_hash64(&frame.width.to_le_bytes()).rotate_left(7);
    hash ^= stable_hash64(&frame.height.to_le_bytes()).rotate_left(13);
    hash ^= stable_hash64(&index.to_le_bytes()).rotate_left(19);
    for value in embedding.iter().take(32) {
        hash ^= stable_hash64(&value.to_bits().to_le_bytes()).rotate_left(3);
    }
    format!("face-{hash:016x}-{index}")
}

fn stable_hash64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    bytes.iter().fold(FNV_OFFSET, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
    })
}

#[cfg(feature = "face")]
fn dynamic_image_from_eye_frame(frame: &EyeFrame) -> Result<image::DynamicImage> {
    match frame.format {
        EyeFrameFormat::Mjpeg => {
            image::load_from_memory(&frame.bytes).context("failed to decode MJPEG eye frame")
        }
        EyeFrameFormat::Rgb8 => {
            image::RgbImage::from_raw(frame.width, frame.height, frame.bytes.clone())
                .map(image::DynamicImage::ImageRgb8)
                .context("RGB eye frame byte length did not match dimensions")
        }
        EyeFrameFormat::Bgr8 => {
            let mut rgb = frame.bytes.clone();
            for pixel in rgb.chunks_exact_mut(3) {
                pixel.swap(0, 2);
            }
            image::RgbImage::from_raw(frame.width, frame.height, rgb)
                .map(image::DynamicImage::ImageRgb8)
                .context("BGR eye frame byte length did not match dimensions")
        }
        EyeFrameFormat::Gray8 => {
            image::GrayImage::from_raw(frame.width, frame.height, frame.bytes.clone())
                .map(image::DynamicImage::ImageLuma8)
                .context("gray eye frame byte length did not match dimensions")
        }
        _ => anyhow::bail!(
            "unsupported eye frame format for face detection: {:?}",
            frame.format
        ),
    }
}

fn summary_extension_values(processed: &ProcessedFrame) -> Vec<f32> {
    let signal = processed
        .eye
        .frames
        .first()
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let mean = if signal.is_empty() {
        0.0
    } else {
        signal.iter().sum::<f32>() / signal.len() as f32
    };
    vec![
        signal.len() as f32,
        mean,
        processed.eye.image_vectors.len() as f32,
        processed.face.vectors.len() as f32,
    ]
}

fn frame_summary_vector(frame: &EyeFrame, signal: &[f32]) -> Vec<f32> {
    let mean = if signal.is_empty() {
        0.0
    } else {
        signal.iter().sum::<f32>() / signal.len() as f32
    };
    vec![
        frame.width as f32,
        frame.height as f32,
        frame.bytes.len() as f32,
        mean,
    ]
}

impl SensorUpdateTimes {
    fn age_ms(&self, t_ms: TimeMs) -> serde_json::Value {
        serde_json::json!({
            "body": self.body.map(|value| t_ms.saturating_sub(value)),
            "eye": self.eye.map(|value| t_ms.saturating_sub(value)),
            "ear": self.ear.map(|value| t_ms.saturating_sub(value)),
            "range": self.range.map(|value| t_ms.saturating_sub(value)),
            "imu": self.imu.map(|value| t_ms.saturating_sub(value)),
            "gps": self.gps.map(|value| t_ms.saturating_sub(value)),
            "kinect": self.kinect.map(|value| t_ms.saturating_sub(value)),
            "face": self.face.map(|value| t_ms.saturating_sub(value)),
            "voice": self.voice.map(|value| t_ms.saturating_sub(value)),
        })
    }
}

pub use pete_now::{EyeFrame, EyeFrameFormat};
