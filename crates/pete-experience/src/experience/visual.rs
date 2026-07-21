pub trait DescendantExtractor {
    fn extract(&self, sensation: &Sensation) -> Result<Vec<Sensation>>;
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisualDetectionKind {
    Face,
    Object,
    SalientRegion,
}

impl VisualDetectionKind {
    fn label(&self) -> &'static str {
        match self {
            Self::Face => "face",
            Self::Object => "object-shaped region",
            Self::SalientRegion => "salient visual region",
        }
    }

    fn stage(&self) -> &'static str {
        match self {
            Self::Face => "descendant.face_crop",
            Self::Object => "descendant.object_crop",
            Self::SalientRegion => "descendant.salient_region_crop",
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::Face => "vision.face_crop",
            Self::Object => "vision.object_crop",
            Self::SalientRegion => "vision.salient_region_crop",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DetectedRegion {
    pub kind: VisualDetectionKind,
    pub bbox: BoundingBox,
    pub confidence: f32,
    pub labels: Vec<String>,
}

pub struct VisualDescendantExtractor;

pub trait VisualDetector {
    fn detect(&self, sensation: &Sensation) -> Result<Vec<DetectedRegion>>;
}

impl VisualDescendantExtractor {
    pub fn detect_regions(&self, sensation: &Sensation) -> Vec<DetectedRegion> {
        self.detect(sensation).unwrap_or_default()
    }

    fn extract_visual(&self, sensation: &Sensation) -> Vec<Sensation> {
        let frame = VisualFrame::from_sensation(sensation);
        let regions = frame
            .as_ref()
            .map(detect_salient_regions)
            .unwrap_or_default();
        let mut descendants = regions
            .iter()
            .map(|region| visual_crop_sensation(sensation, frame.as_ref(), region))
            .collect::<Vec<_>>();
        if descendants.is_empty() {
            if let Some(crop) = deterministic_center_crop(sensation, frame.as_ref()) {
                descendants.push(crop);
            }
        }
        descendants
    }
}

impl VisualDetector for VisualDescendantExtractor {
    fn detect(&self, sensation: &Sensation) -> Result<Vec<DetectedRegion>> {
        let Some(frame) = VisualFrame::from_sensation(sensation) else {
            return Ok(Vec::new());
        };
        Ok(detect_salient_regions(&frame))
    }
}

impl DescendantExtractor for VisualDescendantExtractor {
    fn extract(&self, sensation: &Sensation) -> Result<Vec<Sensation>> {
        if sensation.modality == Modality::Vision
            && sensation.payload_kind == SensationPayloadKind::ImageBytes
        {
            Ok(self.extract_visual(sensation))
        } else {
            Ok(Vec::new())
        }
    }
}

#[derive(Clone, Debug)]
struct VisualFrame {
    width: u32,
    height: u32,
    format: String,
    rgb: Vec<u8>,
}

impl VisualFrame {
    fn from_sensation(sensation: &Sensation) -> Option<Self> {
        let width = payload_u32(&sensation.payload, "width")?;
        let height = payload_u32(&sensation.payload, "height")?;
        if width == 0 || height == 0 {
            return None;
        }
        let bytes = sensation
            .payload
            .get("raw_bytes_b64")
            .and_then(Value::as_str)
            .and_then(|encoded| {
                base64::engine::general_purpose::STANDARD
                    .decode(encoded)
                    .ok()
            })?;
        let format = sensation
            .payload
            .get("format")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let pixel_count = width as usize * height as usize;
        let rgb = match normalized_visual_format(&format).as_str() {
            "rgb8" if bytes.len() >= pixel_count * 3 => bytes[..pixel_count * 3].to_vec(),
            "bgr8" if bytes.len() >= pixel_count * 3 => {
                let mut rgb = Vec::with_capacity(pixel_count * 3);
                for pixel in bytes.chunks_exact(3).take(pixel_count) {
                    rgb.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
                }
                rgb
            }
            "gray8" | "grey8" if bytes.len() >= pixel_count => {
                let mut rgb = Vec::with_capacity(pixel_count * 3);
                for value in bytes.iter().take(pixel_count) {
                    rgb.extend_from_slice(&[*value, *value, *value]);
                }
                rgb
            }
            _ if bytes.len() >= pixel_count * 3 => bytes[..pixel_count * 3].to_vec(),
            _ => return None,
        };
        Some(Self {
            width,
            height,
            format,
            rgb,
        })
    }
}

fn normalized_visual_format(format: &str) -> String {
    format
        .trim_matches('"')
        .trim()
        .trim_start_matches("EyeFrameFormat::")
        .to_ascii_lowercase()
}

fn payload_u32(payload: &Value, key: &str) -> Option<u32> {
    payload
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

fn detect_salient_regions(frame: &VisualFrame) -> Vec<DetectedRegion> {
    let width = frame.width as usize;
    let height = frame.height as usize;
    let pixels = width.saturating_mul(height);
    if pixels < 16 || frame.rgb.len() < pixels * 3 {
        return Vec::new();
    }

    let mut luma = Vec::with_capacity(pixels);
    let mut mean = 0.0_f32;
    for pixel in frame.rgb.chunks_exact(3).take(pixels) {
        let value =
            (0.2126 * pixel[0] as f32 + 0.7152 * pixel[1] as f32 + 0.0722 * pixel[2] as f32)
                / 255.0;
        mean += value;
        luma.push(value);
    }
    mean /= pixels as f32;
    let threshold = (mean + 0.18).clamp(0.12, 0.82);
    let mut visited = vec![false; pixels];
    let mut regions = Vec::new();

    for start in 0..pixels {
        if visited[start] || luma[start] < threshold {
            continue;
        }
        let mut stack = vec![start];
        visited[start] = true;
        let mut min_x = width;
        let mut max_x = 0_usize;
        let mut min_y = height;
        let mut max_y = 0_usize;
        let mut count = 0_usize;
        let mut luma_sum = 0.0_f32;
        let mut skin_like = 0_usize;

        while let Some(index) = stack.pop() {
            let x = index % width;
            let y = index / width;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            count += 1;
            luma_sum += luma[index];
            let base = index * 3;
            let r = frame.rgb[base];
            let g = frame.rgb[base + 1];
            let b = frame.rgb[base + 2];
            if is_skin_like_rgb(r, g, b) {
                skin_like += 1;
            }

            for neighbor in neighbors4(index, x, y, width, height) {
                if !visited[neighbor] && luma[neighbor] >= threshold {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }

        let bbox_width = max_x.saturating_sub(min_x) + 1;
        let bbox_height = max_y.saturating_sub(min_y) + 1;
        let area_ratio = count as f32 / pixels as f32;
        if count < 8 || area_ratio < 0.01 || bbox_width < 3 || bbox_height < 3 {
            continue;
        }
        let fill_ratio = count as f32 / (bbox_width * bbox_height) as f32;
        let mean_region_luma = luma_sum / count as f32;
        let aspect = bbox_width as f32 / bbox_height as f32;
        let skin_ratio = skin_like as f32 / count as f32;
        let kind = if skin_ratio > 0.45 && (0.55..=1.45).contains(&aspect) {
            VisualDetectionKind::Face
        } else if fill_ratio > 0.25 && area_ratio > 0.025 {
            VisualDetectionKind::Object
        } else {
            VisualDetectionKind::SalientRegion
        };
        let confidence =
            (0.28 + area_ratio.sqrt() * 0.55 + (mean_region_luma - mean).max(0.0) * 0.4)
                .clamp(0.05, 0.92);
        let mut labels = vec![kind.label().to_string()];
        labels.push("visual crop".to_string());
        regions.push(DetectedRegion {
            kind,
            bbox: BoundingBox {
                x: min_x as u32,
                y: min_y as u32,
                width: bbox_width as u32,
                height: bbox_height as u32,
            },
            confidence,
            labels,
        });
    }

    regions.sort_by(|left, right| {
        right
            .confidence
            .partial_cmp(&left.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    regions.truncate(3);
    regions
}

fn neighbors4(index: usize, x: usize, y: usize, width: usize, height: usize) -> [usize; 4] {
    [
        if x > 0 { index - 1 } else { index },
        if x + 1 < width { index + 1 } else { index },
        if y > 0 { index - width } else { index },
        if y + 1 < height { index + width } else { index },
    ]
}

fn is_skin_like_rgb(r: u8, g: u8, b: u8) -> bool {
    r > 95 && g > 40 && b > 20 && r > g && g >= b && r.saturating_sub(b) > 35
}

fn visual_crop_sensation(
    parent: &Sensation,
    frame: Option<&VisualFrame>,
    region: &DetectedRegion,
) -> Sensation {
    let mut metadata = parent.metadata.clone();
    metadata.bbox = Some(region.bbox);
    metadata.confidence = Some(region.confidence);
    for label in &region.labels {
        if !metadata.labels.contains(label) {
            metadata.labels.push(label.clone());
        }
    }
    metadata.properties.insert(
        "detection_kind".to_string(),
        serde_json::to_value(&region.kind).unwrap_or(Value::Null),
    );
    if let Some(frame) = frame {
        metadata.properties.insert(
            "source_format".to_string(),
            Value::String(frame.format.clone()),
        );
    }
    let crop_bytes_b64 = frame
        .and_then(|frame| crop_rgb_bytes(frame, region.bbox))
        .map(|bytes| base64::engine::general_purpose::STANDARD.encode(bytes));
    let crop_content_id = crop_bytes_b64
        .as_deref()
        .map(|encoded| format!("crop:{:04}", (stable_unit(encoded) * 10_000.0) as u32));
    let mut payload = json!({
        "parent_image": parent.id,
        "bbox": region.bbox,
        "width": region.bbox.width,
        "height": region.bbox.height,
        "method": "visual_region_proposal_v0",
        "detection_kind": &region.kind,
        "confidence": region.confidence,
        "labels": &region.labels,
    });
    if let Some(content_id) = crop_content_id {
        payload["crop_content_id"] = Value::String(content_id);
    }
    if let Some(encoded) = crop_bytes_b64 {
        payload["raw_bytes_b64"] = Value::String(encoded);
        payload["format"] = Value::String("rgb8".to_string());
    }
    Sensation::descendant(
        parent,
        region.kind.kind(),
        SensationPayloadKind::Crop,
        payload,
        metadata,
        region.kind.stage(),
    )
    .with_summary(match &region.kind {
        VisualDetectionKind::Face => "I see a face close to me.",
        VisualDetectionKind::Object => "I notice an object-shaped region ahead.",
        VisualDetectionKind::SalientRegion => "I notice a salient patch of the scene.",
    })
}

fn crop_rgb_bytes(frame: &VisualFrame, bbox: BoundingBox) -> Option<Vec<u8>> {
    let frame_width = frame.width as usize;
    let frame_height = frame.height as usize;
    let x0 = bbox.x as usize;
    let y0 = bbox.y as usize;
    let width = bbox.width as usize;
    let height = bbox.height as usize;
    if width == 0 || height == 0 || x0 >= frame_width || y0 >= frame_height {
        return None;
    }
    let x1 = (x0 + width).min(frame_width);
    let y1 = (y0 + height).min(frame_height);
    let mut crop = Vec::with_capacity((x1 - x0) * (y1 - y0) * 3);
    for y in y0..y1 {
        let start = (y * frame_width + x0) * 3;
        let end = (y * frame_width + x1) * 3;
        crop.extend_from_slice(&frame.rgb[start..end]);
    }
    Some(crop)
}

fn deterministic_center_crop(parent: &Sensation, frame: Option<&VisualFrame>) -> Option<Sensation> {
    let width = payload_u32(&parent.payload, "width").unwrap_or(0);
    let height = payload_u32(&parent.payload, "height").unwrap_or(0);
    if width < 16 || height < 16 {
        return None;
    }
    let bbox = BoundingBox {
        x: width / 4,
        y: height / 4,
        width: (width / 2).max(1),
        height: (height / 2).max(1),
    };
    let mut metadata = parent.metadata.clone();
    metadata.bbox = Some(bbox);
    metadata.labels.push("central visual crop".to_string());
    metadata.confidence = Some(0.35);
    let crop_bytes_b64 = frame
        .and_then(|frame| crop_rgb_bytes(frame, bbox))
        .map(|bytes| base64::engine::general_purpose::STANDARD.encode(bytes));
    let mut payload = json!({
        "parent_image": parent.id,
        "bbox": bbox,
        "width": bbox.width,
        "height": bbox.height,
        "method": "deterministic_center_crop",
    });
    if let Some(encoded) = crop_bytes_b64 {
        payload["raw_bytes_b64"] = Value::String(encoded);
        payload["format"] = Value::String("rgb8".to_string());
    }
    Some(
        Sensation::descendant(
            parent,
            "vision.crop",
            SensationPayloadKind::Crop,
            payload,
            metadata,
            "descendant.center_crop",
        )
        .with_summary("I narrow my sight toward the middle of the frame."),
    )
}
