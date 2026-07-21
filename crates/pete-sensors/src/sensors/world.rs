#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorldSnapshot {
    pub body: BodySense,
    pub final_selected_action: Option<ActionPrimitive>,
    pub llm_action_proposal: Option<LlmActionProposal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action_debug: Option<serde_json::Value>,
    pub eye_frame: Option<EyeFrame>,
    pub ear_pcm: Option<PcmAudioFrame>,
    pub eye: EyeSense,
    pub ear: EarSense,
    pub range: RangeSense,
    pub imu: ImuSense,
    pub gps: Option<GpsSense>,
    pub kinect: KinectSense,
    pub objects: ObjectSense,
    pub face: FaceSense,
    pub voice: VoiceSense,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub latency_calibration: BTreeMap<String, pete_now::StreamLatencyCalibration>,
    #[serde(default)]
    pub locomotion_calibration: pete_now::LocomotionCalibrationState,
    pub extensions: Vec<ExtensionSense>,
}

impl Default for WorldSnapshot {
    fn default() -> Self {
        Self {
            body: BodySense::default(),
            final_selected_action: None,
            llm_action_proposal: None,
            action_debug: None,
            eye_frame: None,
            ear_pcm: None,
            eye: EyeSense {
                schema_version: 1,
                ..EyeSense::default()
            },
            ear: EarSense {
                schema_version: 1,
                ..EarSense::default()
            },
            range: RangeSense {
                schema_version: 1,
                ..RangeSense::default()
            },
            imu: ImuSense {
                schema_version: 1,
                ..ImuSense::default()
            },
            gps: None,
            kinect: KinectSense {
                schema_version: 1,
                ..KinectSense::default()
            },
            objects: ObjectSense {
                schema_version: 1,
                ..ObjectSense::default()
            },
            face: FaceSense {
                schema_version: 1,
                ..FaceSense::default()
            },
            voice: VoiceSense {
                schema_version: 1,
                ..VoiceSense::default()
            },
            latency_calibration: BTreeMap::new(),
            locomotion_calibration: pete_now::LocomotionCalibrationState::default(),
            extensions: Vec::new(),
        }
    }
}

impl WorldSnapshot {
    pub fn to_now(&self, t_ms: u64) -> Now {
        let mut now = Now::blank(t_ms, self.body.clone());
        now.eye = self.eye.clone();
        now.eye_frame = self.eye_frame.clone();
        now.ear = self.ear.clone();
        now.face = self.face.clone();
        now.voice = self.voice.clone();
        now.range = self.range.clone();
        now.imu = self.imu.clone();
        now.gps = self.gps.clone();
        now.kinect = self.kinect.clone();
        now.objects = self.objects.clone();
        if !self.latency_calibration.is_empty() {
            now.extensions.insert(
                "sensor.latency_calibration".to_string(),
                serde_json::to_value(&self.latency_calibration)
                    .unwrap_or_else(|error| serde_json::json!({"error": error.to_string()})),
            );
        }
        now.extensions.insert(
            "calibration.locomotion".to_string(),
            serde_json::to_value(&self.locomotion_calibration)
                .unwrap_or_else(|error| serde_json::json!({"error": error.to_string()})),
        );
        now.predictions = PredictionSense {
            schema_version: 1,
            ..PredictionSense::default()
        };
        now.surprise = SurpriseSense {
            schema_version: 1,
            ..SurpriseSense::default()
        };
        for extension in &self.extensions {
            now.extensions.insert(
                extension.name.clone(),
                serde_json::json!({
                    "schema_version": extension.schema_version,
                    "values": extension.values,
                }),
            );
        }
        now
    }
}

impl From<&EyeFrame> for FrameKey {
    fn from(frame: &EyeFrame) -> Self {
        Self {
            captured_at_ms: frame.captured_at_ms,
            width: frame.width,
            height: frame.height,
            format: format!("{:?}", frame.format),
            byte_len: frame.bytes.len(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WorldUpdate {
    pub body: Option<BodySense>,
    pub eye_frame: Option<EyeFrame>,
    pub ear_pcm: Option<PcmAudioFrame>,
    pub eye: Option<EyeSense>,
    pub ear: Option<EarSense>,
    pub range: Option<RangeSense>,
    pub imu: Option<ImuSense>,
    pub gps: Option<GpsSense>,
    pub kinect: Option<KinectSense>,
    pub objects: Option<ObjectSense>,
    pub face: Option<FaceSense>,
    pub voice: Option<VoiceSense>,
    pub extensions: Option<Vec<ExtensionSense>>,
}

impl WorldUpdate {
    pub fn apply_to(self, snapshot: &mut WorldSnapshot) {
        if let Some(body) = self.body {
            snapshot.body = body;
        }
        if let Some(frame) = self.eye_frame {
            snapshot.eye_frame = Some(frame);
        }
        if let Some(frame) = self.ear_pcm {
            snapshot.ear_pcm = Some(frame);
        }
        if let Some(eye) = self.eye {
            snapshot.eye = eye;
        }
        if let Some(ear) = self.ear {
            snapshot.ear = ear;
        }
        if let Some(range) = self.range {
            snapshot.range = range;
        }
        if let Some(imu) = self.imu {
            snapshot.imu = imu;
        }
        if self.gps.is_some() {
            snapshot.gps = self.gps;
        }
        if let Some(kinect) = self.kinect {
            snapshot.kinect = kinect;
        }
        if let Some(objects) = self.objects {
            snapshot.objects = objects;
        }
        if let Some(face) = self.face {
            snapshot.face = face;
        }
        if let Some(voice) = self.voice {
            snapshot.voice = voice;
        }
        if let Some(extensions) = self.extensions {
            snapshot.extensions = extensions;
        }
    }
}
